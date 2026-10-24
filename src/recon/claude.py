"""Claude through Anthropic's own Messages API, in two arms that share one loop:

- direct: one conversation per line, the tools called by the model as ordinary tool calls;
- ptc: one conversation per statement with programmatic tool calling, where Claude writes
  Python in Anthropic's code execution container and that code calls the tools.

Both reach the tools through the bridge, so the tool definitions are the ones the MCP server
lists (plus the two host-side decisions), and every call is checked per line with the agent's
verifier. Programmatic tool calling cannot call MCP connector tools directly, which is why the
same schemas are sent as custom tools and run here. `allowed_callers` is not a security
boundary, so a direct call in the ptc arm goes through the same bridge and is counted.

Record and replay work as in llm.LLM: the key is the sha256 of the canonical request body,
which holds no header or key.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recon import agent, codemode, tools
from recon.agent import DEFENCES, Defences, Limits, LineResult
from recon.bridge import CAPS, Bridge, Caps
from recon.codemode import Statement
from recon.llm import ModelSpec, ReplayMiss, request_key
from recon.synth import BankLine

PROGRAMMATIC = "code_execution_20260120"
CODE_EXECUTION = {"type": PROGRAMMATIC, "name": "code_execution"}
MAX_TOKENS = 16_000
LIMITS = Limits()


@dataclass(frozen=True)
class Usage:
    fresh: int  # uncached input tokens
    cache_read: int
    cache_write: int
    output: int


def usage_of(response: dict[str, Any]) -> Usage:
    u = response.get("usage") or {}
    return Usage(
        u.get("input_tokens") or 0,
        u.get("cache_read_input_tokens") or 0,
        u.get("cache_creation_input_tokens") or 0,
        u.get("output_tokens") or 0,
    )


def cost_usd(spec: ModelSpec, usage: Usage) -> float:
    """Every input category counts towards the prompt length that picks the price card."""
    prompt = usage.fresh + usage.cache_read + usage.cache_write
    scale = spec.long_multiplier if spec.long_prompt and prompt > spec.long_prompt else 1.0
    return (
        scale
        * (
            usage.fresh * spec.usd_in
            + usage.cache_read * spec.usd_cached
            + usage.cache_write * spec.usd_cache_write
            + usage.output * spec.usd_out
        )
        / 1e6
    )


@dataclass(frozen=True)
class Reply:
    response: dict[str, Any]
    latency_ms: float
    replayed: bool


def transient_errors() -> tuple[type[Exception], ...]:
    """Worth escalating for, as in agent.TRANSIENT_ERRORS; the SDK has already retried."""
    import anthropic  # an optional dependency: `uv sync --extra claude`

    return (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
    )


class ClaudeLLM:
    """One Messages API client with the record and replay of llm.LLM."""

    def __init__(
        self,
        spec: ModelSpec,
        mode: str = "live",
        cassette: Path | None = None,
        client: Any = None,
        salt: str = "",
        on_request: Callable[[list[dict]], None] | None = None,
    ) -> None:
        if mode not in ("live", "record", "replay"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode != "live" and cassette is None:
            raise ValueError(f"{mode} mode needs a cassette path")
        self.spec, self.mode, self.cassette = spec, mode, cassette
        self.recorded: dict[str, dict] = {}
        if cassette is not None and cassette.exists():
            for line in cassette.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.recorded[entry["key"]] = entry
        self._client = client
        self.salt = salt
        self.on_request = on_request
        self._lock = threading.Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            if not os.environ.get(self.spec.key_env or ""):
                raise RuntimeError(
                    f"{self.spec.key_env} is not set: add it to .env and run with"
                    " `uv run --extra claude --env-file .env ...`"
                )
            import anthropic

            self._client = anthropic.Anthropic(timeout=300)
        return self._client

    def create(self, body: dict[str, Any]) -> Reply:
        if self.on_request is not None:
            self.on_request(body["messages"])
        key = request_key({**body, "salt": self.salt} if self.salt else body)
        if self.mode == "replay":
            entry = self.recorded.get(key)
            if entry is None:
                raise ReplayMiss(f"no recorded response for request {key[:12]}")
            return Reply(entry["response"], entry["latency_ms"], replayed=True)
        start = time.perf_counter()
        raw = self.client.messages.create(**body)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        response = raw if isinstance(raw, dict) else raw.to_dict(mode="json")
        if self.mode == "record":
            entry = {"key": key, "latency_ms": latency_ms, "response": response}
            with self._lock:
                self.cassette.parent.mkdir(parents=True, exist_ok=True)
                with self.cassette.open("a") as f:
                    f.write(json.dumps(entry, sort_keys=True) + "\n")
        return Reply(response, latency_ms, replayed=False)


def tool_definitions(specs: list[dict[str, Any]], programmatic: bool) -> list[dict[str, Any]]:
    """The bridge's tools as Anthropic custom tools; with programmatic calling, callable
    only from code, plus the code execution tool. No `strict`, which programmatic calling
    does not accept: the bridge validates every call instead."""
    defined = [
        {
            "name": spec["name"],
            "description": spec["description"],
            "input_schema": spec["input_schema"],
            **({"allowed_callers": [PROGRAMMATIC]} if programmatic else {}),
        }
        for spec in specs
    ]
    return [*defined, CODE_EXECUTION] if programmatic else defined


def reconcile_statement(
    lines: list[BankLine],
    model: ClaudeLLM,
    search: tools.Search,
    defences: Defences = DEFENCES["on"],
    programmatic: bool = False,
    limits: Limits = LIMITS,
    caps: Caps = CAPS,
) -> tuple[dict[str, LineResult], Statement]:
    """Decide a statement in one conversation, with the budget of codemode: N lines get N
    times the per-line steps, tokens and dollars. A paused turn is sent back as it is."""
    began = time.perf_counter()
    n = len(lines)
    stats = Statement(lines=n)
    transient = transient_errors()
    with Bridge(lines, search, defences, model.spec.name, caps) as bridge:
        system = (
            codemode.system_prompt(bridge.api(), defences) if programmatic else defences.prompt()
        )
        one_line = not programmatic and n == 1  # the per-line agent's own message
        user = (
            agent.user_message(lines[0], defences)
            if one_line
            else codemode.statement_message(lines, defences)
        )
        definitions = tool_definitions(bridge.api(), programmatic)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        container = None
        for step in range(1, limits.steps * n + 1):
            body = {
                "model": model.spec.api_model,
                "max_tokens": MAX_TOKENS,
                "system": system,
                "tools": definitions,
                "messages": messages,
                **model.spec.extra_body,
            }
            if container is not None:
                body["container"] = container
            try:
                reply = model.create(body)
            except transient as exc:
                stats.ended = f"provider error: {type(exc).__name__}"
                break
            response = reply.response
            usage = usage_of(response)
            cost = cost_usd(model.spec, usage)
            stats.steps = step
            stats.input_tokens += usage.fresh + usage.cache_read + usage.cache_write
            stats.cached_tokens += usage.cache_read
            stats.cache_write_tokens += usage.cache_write
            stats.output_tokens += usage.output
            stats.cost_usd += cost
            stats.model_ms += reply.latency_ms
            stop = response.get("stop_reason")
            stats.trace.append(
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "anthropic",
                    "gen_ai.request.model": model.spec.api_model,
                    "gen_ai.response.model": response.get("model"),
                    "gen_ai.response.finish_reasons": [stop],
                    "gen_ai.usage.input_tokens": usage.fresh + usage.cache_read + usage.cache_write,
                    "gen_ai.usage.output_tokens": usage.output,
                    "recon.cached_tokens": usage.cache_read,
                    "recon.cost_usd": round(cost, 6),
                    "recon.latency_ms": reply.latency_ms,
                    "recon.replayed": reply.replayed,
                    "recon.prompt_sha": agent.sha(system),
                    "recon.statement_lines": n,
                    "recon.step": step,
                    "recon.container": (response.get("container") or {}).get("id"),
                    "recon.server_tool_use": (response.get("usage") or {}).get("server_tool_use"),
                }
            )
            if response.get("container"):
                if container != response["container"]["id"]:
                    stats.containers += 1
                container = response["container"]["id"]
            # The turn goes back unchanged, thinking blocks included.
            messages.append({"role": "assistant", "content": response.get("content", [])})
            if stop == "pause_turn":
                continue  # the server resumes the paused turn; no new user message
            if stop != "tool_use":
                stats.ended = f"model stopped ({stop})"
                break
            results = []
            for block in response.get("content", []):
                if block.get("type") != "tool_use":
                    continue
                caller = (block.get("caller") or {}).get("type", "direct")
                stats.programmatic_calls += caller != "direct"
                content = bridge.handle(block["name"], block.get("input"), caller)
                results.append(
                    {"type": "tool_result", "tool_use_id": block["id"], "content": content}
                )
            # Only tool results, as programmatic tool calling requires of a continuation.
            messages.append({"role": "user", "content": results})
            if len(bridge.decisions) == n:
                stats.ended = "every line decided"
                break
            spent = stats.input_tokens + stats.output_tokens
            if spent > limits.max_tokens * n or stats.cost_usd > limits.max_usd * n:
                stats.ended = "budget limit reached"
                break
        else:
            stats.ended = "step limit reached"
        stats.trace += bridge.trace
        provider_failed = stats.ended.startswith("provider error")
        decisions = bridge.results(stats.ended if provider_failed else f"claude: {stats.ended}")
    stats.wall_ms = round((time.perf_counter() - began) * 1000, 1)
    return {i: LineResult(decision=d) for i, d in decisions.items()}, stats


def reconcile_open(
    lines: list[BankLine],
    model: ClaudeLLM,
    arm: str,
    statement_size: str,
    workers: int,
    database_url: str | None,
    defences: Defences = DEFENCES["on"],
) -> tuple[dict[str, LineResult], list[Statement]]:
    """direct: one conversation per line; ptc: a statement per S."""
    programmatic = arm == "ptc"
    size = statement_size if programmatic else "1"

    def decide(statement: list[BankLine], search: tools.Search):
        return reconcile_statement(statement, model, search, defences, programmatic)

    return codemode.run_statements(lines, size, workers, database_url, decide)
