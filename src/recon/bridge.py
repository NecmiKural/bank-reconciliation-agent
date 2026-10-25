"""The host side of code mode: the only way model-written code reaches the tools.

Code running in our sandbox, or in Anthropic's code execution container, calls four tools by
name with a line id. The bridge checks every call (the tool is allowed, the arguments match
its schema and size limits, the line is an open line of this statement and still undecided,
the line's call budget is not spent), then sends the two read tools through the read-only
MCP server and the two decisions to that line's own Toolbox, whose verifier is the agent's.

Nothing the code prints or returns is a decision. Only an accepted propose_match or escalate
is, and the first one on a line is final.
"""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from mcp import Client

from recon import mcp_server
from recon.agent import Defences, finalize
from recon.score import Decision
from recon.synth import BankLine
from recon.tools import MAX_COMBINATION_IDS, TOOL_SCHEMAS, Search, Toolbox

READ_TOOLS = ("search_candidates", "find_combinations")
TERMINAL_TOOLS = ("propose_match", "escalate")
MAX_CHARS = {"line_id": 40, "payee": 140, "invoice_number": 40, "rationale": 500, "reason": 500}
MAX_ID_CHARS = 40


@dataclass(frozen=True)
class Caps:
    """Per-line budgets for code that calls tools in a loop.

    NOTE: fixed numbers set just above what the per-line agent has needed: across 3,198
    line-runs in local traces it made at most 11 search or combination calls and had at most
    1 proposal rejected on a line. They stop a brute-force search for a proposal the verifier
    accepts; a larger statement does not earn a larger per-line budget.
    """

    calls_per_line: int = 12
    rejected_per_line: int = 2


CAPS = Caps()


def _terminal_schema(name: str) -> dict[str, Any]:
    function = next(t["function"] for t in TOOL_SCHEMAS if t["function"]["name"] == name)
    parameters = function["parameters"]
    properties = {"line_id": {"type": "string", "description": "The bank line this is for."}}
    return {
        "name": name,
        "description": function["description"],
        "input_schema": {
            "type": "object",
            "properties": properties | parameters["properties"],
            "required": ["line_id", *parameters["required"]],
            "additionalProperties": False,
        },
    }


class Bridge:
    """One statement's tools. Use as a context manager: it holds an MCP client session."""

    # Every call with its arguments and result, for the replay export (src/recon/export.py);
    # trace rows stay metadata-only. NOTE: one list shared by every bridge in the process,
    # set only while the export replays a slice with one worker; an instance field returned
    # with each Statement is the upgrade if anything else needs it.
    log: list[dict[str, Any]] | None = None

    def __init__(
        self,
        lines: list[BankLine],
        search: Search,
        defences: Defences,
        source: str,
        caps: Caps = CAPS,
    ) -> None:
        self.boxes = {line.id: Toolbox(line, search, defences.payer_check) for line in lines}
        self.defences, self.source, self.caps = defences, source, caps
        self.decisions: dict[str, Decision] = {}
        self.calls: dict[str, int] = dict.fromkeys(self.boxes, 0)
        self.rejected: dict[str, int] = dict.fromkeys(self.boxes, 0)
        self.trace: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.specs: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> Bridge:
        # The MCP session lives in one task on its own event loop thread (anyio needs it
        # opened and closed in the same task); calls reach it through a queue.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._jobs: asyncio.Queue = asyncio.Queue()
        ready: Future = Future()
        server = mcp_server.build(toolboxes=self.boxes)
        self._session = asyncio.run_coroutine_threadsafe(self._serve(server, ready), self._loop)
        try:
            listed = ready.result()
        except BaseException:
            self._stop()
            raise
        for tool in listed:
            if tool.name in READ_TOOLS:
                schema = {**tool.input_schema, "additionalProperties": False}
                self.specs[tool.name] = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": schema,
                }
        for name in TERMINAL_TOOLS:
            self.specs[name] = _terminal_schema(name)
        return self

    def __exit__(self, *exc: object) -> None:
        self._loop.call_soon_threadsafe(self._jobs.put_nowait, None)
        try:
            self._session.result()
        finally:
            self._stop()

    def _stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    async def _serve(self, server: Any, ready: Future) -> None:
        try:
            async with Client(server) as client:
                ready.set_result((await client.list_tools()).tools)
                while (job := await self._jobs.get()) is not None:
                    name, args, done = job
                    try:
                        done.set_result(await client.call_tool(name, args))
                    except Exception as exc:
                        done.set_exception(exc)
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    def api(self) -> list[dict[str, Any]]:
        """The four tools as code sees them: name, description and JSON schema, the read
        tools exactly as the MCP server lists them."""
        return [self.specs[name] for name in (*READ_TOOLS, *TERMINAL_TOOLS)]

    def handle(self, name: str, args: Any, caller: str = "sandbox") -> str:
        """Run one call from code and return its result as a JSON string. Never raises for
        anything the code sent."""
        with self._lock:
            refusal, outcome = self._handle(name, args)
            line_id = args.get("line_id") if isinstance(args, dict) else None
            ids = args.get("invoice_ids") if isinstance(args, dict) else None
            self.trace.append(
                {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": str(name)[:40],
                    "recon.ok": refusal is None and "error" not in outcome,
                    "recon.line_id": line_id if isinstance(line_id, str) else None,
                    "recon.caller": caller,
                    "recon.refusal": refusal,
                    "recon.invoice_ids": [i[:40] for i in ids[:25] if isinstance(i, str)]
                    if isinstance(ids, list)
                    else None,
                }
            )
            if Bridge.log is not None:
                Bridge.log.append(
                    {
                        "line_id": line_id,
                        "tool": name,
                        "caller": caller,
                        "args": args,
                        "refusal": refusal,
                        "result": outcome,
                    }
                )
            return json.dumps(outcome, sort_keys=True)

    def _handle(self, name: Any, args: Any) -> tuple[str | None, dict[str, Any]]:
        if name not in self.specs:
            return "unknown_method", {"error": f"unknown tool {str(name)[:40]!r}"}
        if not isinstance(args, dict):
            return "schema", {"error": "arguments must be an object"}
        if "line_id" not in args:  # said plainly, so the model can add it and call again
            return "schema", {"error": "missing arguments ['line_id']"}
        line_id = args.get("line_id")
        if not isinstance(line_id, str) or line_id not in self.boxes:
            return "scope", {"error": "line_id is not an open line of this statement"}
        if line_id in self.decisions:
            return "decided", {"error": f"line {line_id} is already decided"}
        if problem := self._check(name, args):
            return problem
        rest = {k: v for k, v in args.items() if k != "line_id"}
        if name in READ_TOOLS:
            if self.calls[line_id] >= self.caps.calls_per_line:
                return "call_cap", {"error": f"line {line_id} has used its tool calls"}
            self.calls[line_id] += 1
            return None, self._read(name, args)
        outcome, decision = self.boxes[line_id].call(name, json.dumps(rest))
        if decision is not None:
            self.decisions[line_id] = finalize(decision, self.source, self.defences)
        elif name == "propose_match":
            self.rejected[line_id] += 1
            if self.rejected[line_id] >= self.caps.rejected_per_line:
                reason = f"proposal limit reached; last refusal: {outcome.get('error', '')}"
                self.decisions[line_id] = finalize(
                    Decision(line_id, "escalate", rationale=reason), self.source, self.defences
                )
        return None, outcome

    def _check(self, name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        schema = self.specs[name]["input_schema"]
        properties = schema["properties"]
        if extra := sorted(set(args) - set(properties)):
            return "schema", {"error": f"unexpected arguments {extra}"}
        if missing := sorted(set(schema.get("required", [])) - set(args)):
            return "schema", {"error": f"missing arguments {missing}"}
        for key, value in args.items():
            if properties[key].get("type") == "array":
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    return "schema", {"error": f"{key} must be a list of strings"}
                if len(value) > MAX_COMBINATION_IDS or any(len(v) > MAX_ID_CHARS for v in value):
                    return "size", {"error": f"{key} is too long"}
            elif not isinstance(value, str):
                return "schema", {"error": f"{key} must be a string"}
            elif len(value) > MAX_CHARS.get(key, 0):
                return "size", {"error": f"{key} is too long"}
        return None

    def _read(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        done: Future = Future()
        self._loop.call_soon_threadsafe(self._jobs.put_nowait, (name, args, done))
        result = done.result()
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            return {"error": text[:500] or f"{name} failed"}
        return result.structured_content or {}

    def results(self, reason: str = "code mode: no decision") -> dict[str, Decision]:
        """Every line's decision; a line the code never decided goes to a human, with the
        reason the statement ended."""
        return {
            line_id: self.decisions.get(line_id)
            or finalize(
                Decision(line_id, "escalate", rationale=reason),
                self.source,
                self.defences,
            )
            for line_id in self.boxes
        }
