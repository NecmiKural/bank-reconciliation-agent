"""Code mode: one conversation per statement, in which the model writes Python that calls
the tools.

The model has one tool, run_python. Each cell runs in the sandbox, and every tool call the
code makes goes through the bridge: the four tools as the MCP server and the agent's Toolbox
define them, checked per line, with the agent's verifier. Like the per-line loop, this one
ends only in decisions the code accepted; a line it leaves undecided, a provider error or a
spent budget sends the remaining lines to a human.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

from recon import db, tools
from recon.agent import (
    DEFENCES,
    TRANSIENT_ERRORS,
    Defences,
    Limits,
    LineResult,
    assistant_message,
    sha,
    user_message,
)
from recon.bridge import CAPS, Bridge, Caps
from recon.llm import LLM, cost_usd, usage_of
from recon.redact import redact
from recon.sandbox import CellResult, Sandbox
from recon.synth import BankLine

PROMPT = files("recon").joinpath("prompts", "code_mode.md").read_text()
RUN_PYTHON = tools._tool(
    "run_python",
    "Run one cell of Python in the sandbox. Variables persist between cells; only what the"
    " cell prints comes back.",
    {"code": tools._string("Python code. Await the tool functions; print what you need.")},
)
STATEMENT_SIZES = ("all", "1")
LIMITS = Limits()


def render_api(specs: list[dict[str, Any]]) -> str:
    """The tools as Python signatures, from the specs the bridge took from the MCP server."""
    blocks = []
    for spec in specs:
        schema = spec["input_schema"]
        required = set(schema.get("required", []))
        params = ", ".join(
            f"{name} ({'list of str' if prop.get('type') == 'array' else 'str'}"
            f"{'' if name in required else ', optional'})"
            for name, prop in schema["properties"].items()
        )
        blocks.append(
            f"async def {spec['name']}(params: dict) -> str\n"
            f"    {spec['description']}\n"
            f"    params: {params}"
        )
    return "\n\n".join(blocks)


def system_prompt(specs: list[dict[str, Any]], defences: Defences) -> str:
    return defences.prompt(PROMPT.replace("{api}", render_api(specs)))


def statement_message(lines: list[BankLine], defences: Defences) -> str:
    parts = [f"Statement of {len(lines)} open bank lines. Decide every one."]
    parts += [user_message(line, defences) for line in lines]
    return "\n\n".join(parts)


def cell_message(cell: CellResult, restarted: bool, defences: Defences) -> str:
    """What a cell returns to the model: its output and error, redacted like everything else
    the provider sees."""
    clean = redact if defences.redact else str
    body: dict[str, Any] = {"stdout": clean(cell.stdout), "error": clean(cell.error or "") or None}
    if cell.truncated:
        body["truncated"] = True
    if restarted:
        body["sandbox"] = "restarted after the cell was stopped; earlier variables are gone"
    return json.dumps(body, sort_keys=True)


@dataclass
class Statement:
    """Totals for one statement; per-line results carry only their decisions."""

    lines: int
    steps: int = 0  # provider requests
    sampling_passes: int = 0  # requests whose response the model wrote (output tokens > 0)
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_ms: float = 0.0
    sandbox_ms: float = 0.0
    wall_ms: float = 0.0
    cells: int = 0
    code_failures: int = 0
    programmatic_calls: int = 0  # Claude: tool calls made from code in Anthropic's container
    containers: int = 0
    cache_write_tokens: int = 0
    ended: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)


def reconcile_statement(
    lines: list[BankLine],
    model: LLM,
    search: tools.Search,
    defences: Defences = DEFENCES["on"],
    limits: Limits = LIMITS,
    caps: Caps = CAPS,
    sandbox: Callable[[], Sandbox] = Sandbox,
) -> tuple[dict[str, LineResult], Statement]:
    """Decide a statement in one conversation. The budget scales with its size: N lines get
    N times the per-line steps, tokens and dollars."""
    began = time.perf_counter()
    n = len(lines)
    stats = Statement(lines=n)
    provider = model.spec.name.split(":")[0].split("-")[0]
    with Bridge(lines, search, defences, model.spec.name, caps) as bridge:
        names = [spec["name"] for spec in bridge.api()]
        prompt = system_prompt(bridge.api(), defences)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": statement_message(lines, defences)},
        ]
        box = sandbox()
        try:
            box.start(names)
            stats.sandbox_ms += box.start_ms
            for step in range(1, limits.steps * n + 1):
                try:
                    reply = model.chat(messages, [RUN_PYTHON])
                except TRANSIENT_ERRORS as exc:
                    stats.ended = f"provider error: {type(exc).__name__}"
                    break
                response = reply.response
                if not response.choices:
                    stats.ended = "provider returned no choices"
                    break
                usage = usage_of(response)
                cost = cost_usd(model.spec, usage)
                choice = response.choices[0]
                stats.steps = step
                stats.sampling_passes += usage.output_tokens > 0
                stats.input_tokens += usage.input_tokens
                stats.cached_tokens += usage.cached_tokens
                stats.output_tokens += usage.output_tokens
                stats.cost_usd += cost
                stats.model_ms += reply.latency_ms
                stats.trace.append(
                    {
                        "gen_ai.operation.name": "chat",
                        "gen_ai.provider.name": provider,
                        "gen_ai.request.model": model.spec.api_model,
                        "gen_ai.response.model": response.model,
                        "gen_ai.response.finish_reasons": [choice.finish_reason],
                        "gen_ai.usage.input_tokens": usage.input_tokens,
                        "gen_ai.usage.output_tokens": usage.output_tokens,
                        "recon.cached_tokens": usage.cached_tokens,
                        "recon.cost_usd": round(cost, 6),
                        "recon.latency_ms": reply.latency_ms,
                        "recon.replayed": reply.replayed,
                        "recon.prompt_sha": sha(prompt),
                        "recon.statement_lines": n,
                        "recon.step": step,
                    }
                )
                messages.append(assistant_message(choice.message))
                calls = choice.message.tool_calls or []
                if choice.finish_reason != "tool_calls" or not calls:
                    stats.ended = f"model stopped ({choice.finish_reason})"
                    break
                for call in calls:
                    content, box = run_call(call, box, sandbox, bridge, names, defences, stats)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                if len(bridge.decisions) == n:
                    stats.ended = "every line decided"
                    break
                spent = stats.input_tokens + stats.output_tokens
                if spent > limits.max_tokens * n or stats.cost_usd > limits.max_usd * n:
                    stats.ended = "budget limit reached"
                    break
            else:
                stats.ended = "step limit reached"
        finally:
            box.close()
        stats.trace += bridge.trace
        # a provider error keeps its prefix, which makes the whole run invalid
        provider_failed = stats.ended.startswith("provider error")
        decisions = bridge.results(stats.ended if provider_failed else f"code mode: {stats.ended}")
    stats.wall_ms = round((time.perf_counter() - began) * 1000, 1)
    return {i: LineResult(decision=d) for i, d in decisions.items()}, stats


def code_row(
    code: str, where: str, error: str | None = None, killed: bool = False, truncated: bool = False
) -> dict[str, Any]:
    """The code a model ran, for the local trace and the probes. Traces stay in runs/, which
    is not committed; the code is in the recorded model replies anyway."""
    return {
        "gen_ai.operation.name": "execute_code",
        "recon.where": where,
        "recon.code": code[:8000],
        "recon.error": (error or "")[:500] or None,
        "recon.killed": killed,
        "recon.truncated": truncated,
    }


def run_call(
    call: Any,
    box: Sandbox,
    fresh: Callable[[], Sandbox],
    bridge: Bridge,
    names: list[str],
    defences: Defences,
    stats: Statement,
) -> tuple[str, Sandbox]:
    """One run_python call: the cell's message, and the sandbox to use next (a fresh one if
    the cell killed it)."""
    if call.function.name != "run_python":
        return json.dumps({"error": f"unknown tool {call.function.name[:40]!r}"}), box
    try:
        code = json.loads(call.function.arguments)["code"]
        if not isinstance(code, str):
            raise TypeError
    except (ValueError, KeyError, TypeError, RecursionError):
        return json.dumps({"error": "run_python takes one string argument, code"}), box
    cell = box.run(code, lambda method, params: bridge.handle(method, params, "sandbox"))
    stats.cells += 1
    stats.sandbox_ms += cell.exec_ms
    stats.trace.append(code_row(code, "sandbox", cell.error, cell.killed, cell.truncated))
    restarted = False
    if cell.error:
        stats.code_failures += 1
    if cell.killed:
        box.close()
        box = fresh()
        box.start(names)
        stats.sandbox_ms += box.start_ms
        restarted = True
    return cell_message(cell, restarted, defences), box


Decide = Callable[[list[BankLine], tools.Search], tuple[dict[str, LineResult], Statement]]


def reconcile_open(
    lines: list[BankLine],
    model: LLM,
    statement_size: str,
    workers: int,
    database_url: str | None,
    defences: Defences = DEFENCES["on"],
    sandbox: Callable[[], Sandbox] = Sandbox,
) -> tuple[dict[str, LineResult], list[Statement]]:
    def decide(statement: list[BankLine], search: tools.Search):
        return reconcile_statement(statement, model, search, defences, sandbox=sandbox)

    return run_statements(lines, statement_size, workers, database_url, decide)


def run_statements(
    lines: list[BankLine],
    statement_size: str,
    workers: int,
    database_url: str | None,
    decide: Decide,
) -> tuple[dict[str, LineResult], list[Statement]]:
    """The open lines as one statement (S=all, serial) or one statement per line (S=1, in
    parallel, one database connection per worker thread)."""
    if not lines:
        return {}, []
    statements = [lines] if statement_size == "all" else [[line] for line in lines]
    local = threading.local()
    lock = threading.Lock()
    connections = []

    def search(line: BankLine, payee: str, number: str) -> list[tools.Row]:
        conn = getattr(local, "conn", None)
        if conn is None:
            conn = local.conn = db.connect(database_url)
            with lock:
                connections.append(conn)
        return tools.db_search(conn)(line, payee, number)

    pool = ThreadPoolExecutor(max_workers=1 if statement_size == "all" else workers)
    try:
        done = list(pool.map(lambda statement: decide(statement, search), statements))
    finally:
        pool.shutdown(cancel_futures=True)
        for conn in connections:
            conn.close()
    results = {i: r for per_line, _ in done for i, r in per_line.items()}
    return results, [stats for _, stats in done]


def statement_summary(statements: list[Statement]) -> dict[str, Any]:
    """Totals over a run's statements, and how the bridge answered the code."""
    rows = [row for s in statements for row in s.trace]
    tool_rows = [r for r in rows if r["gen_ai.operation.name"] == "execute_tool"]
    input_tokens = sum(s.input_tokens for s in statements)
    cached = sum(s.cached_tokens for s in statements)
    return {
        "statements": len(statements),
        "model_requests": sum(s.steps for s in statements),
        "sampling_passes": sum(s.sampling_passes for s in statements),
        "cells": sum(s.cells for s in statements),
        "code_failures": sum(s.code_failures for s in statements),
        "programmatic_calls": sum(s.programmatic_calls for s in statements),
        "containers": sum(s.containers for s in statements),
        "cache_write_tokens": sum(s.cache_write_tokens for s in statements),
        "input_tokens": input_tokens,
        "cached_tokens": cached,
        "output_tokens": sum(s.output_tokens for s in statements),
        "cache_share": round(cached / input_tokens, 4) if input_tokens else 0.0,
        "cost_usd": round(sum(s.cost_usd for s in statements), 6),
        "model_ms": round(sum(s.model_ms for s in statements), 1),
        "sandbox_ms": round(sum(s.sandbox_ms for s in statements), 1),
        "ended": dict(Counter(s.ended for s in statements)),
        "bridge_calls": dict(Counter(r["gen_ai.tool.name"] for r in tool_rows)),
        "bridge_refusals": dict(
            Counter(r["recon.refusal"] for r in tool_rows if r["recon.refusal"])
        ),
    }
