"""The committed CI slices as JSON for the static replay viewer in docs/replay/.

Each export replays a slice through the whole pipeline, as tests/test_gate.py does, and keeps
what the run already holds in memory: every conversation the model was sent (through the
model's on_request hook, whose message lists the loops append each reply and tool result to),
every call code made through the bridge (Bridge.log), and each line's decision, policy verdict
and label. Nothing here calls a model, and nothing carries a run id or a timestamp, so the same
code and cassette always give the same file.
"""

from __future__ import annotations

import bisect
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from recon import agent, evaluate, llm, score, synth, tools
from recon.bridge import Bridge
from recon.report import START

SLICES = ("A-slice", "A-slice-sandbox", "A-slice-claude", "A-slice-ptc")
CASSETTES = Path("tests/cassettes")
OUT = Path("docs/replay")
README = Path("README.md")
SCHEMA = 1


def tidy(value: Any) -> Any:
    """Every string as the viewer shows it: em and en dashes become commas, as the command
    line prints them, and control characters go, except newlines and tabs, which code needs."""
    if isinstance(value, str):
        text = value.replace(" \u2014 ", ", ").replace("\u2014", ", ").replace("\u2013", ", ")
        return "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    if isinstance(value, dict):
        return {key: tidy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [tidy(item) for item in value]
    return value


def parse(text: Any) -> Any:
    """A tool's JSON arguments or result as an object; anything else as it came."""
    try:
        return json.loads(text)
    except (TypeError, ValueError, RecursionError):
        return text


def headline(readme: Path) -> str:
    """The first paragraph of the README results block, the page's only aggregate claim."""
    block = readme.read_text().split(START, 1)[1]
    paragraphs = [p.strip() for p in block.split("\n\n")]
    return next(p for p in paragraphs if p and not p.startswith("<!--"))


def openai_turns(messages: list[dict]) -> list[dict[str, Any]]:
    results = {m["tool_call_id"]: parse(m["content"]) for m in messages if m["role"] == "tool"}
    return [
        {
            "text": m.get("content") or "",
            "reasoning": m.get("reasoning_content") or "",
            "calls": [
                {
                    "tool": c["function"]["name"],
                    "args": parse(c["function"]["arguments"]),
                    "result": results.get(c["id"]),
                }
                for c in m.get("tool_calls") or []
            ],
            "code": [],
        }
        for m in messages
        if m["role"] == "assistant"
    ]


def code_output(block: dict | None) -> dict[str, Any] | None:
    """What a code execution in Anthropic's container printed, in the sandbox's shape."""
    if block is None:
        return None
    content = block.get("content") or {}
    error = content.get("stderr") or content.get("error_code") or None
    return {"stdout": content.get("stdout", ""), "error": error}


def anthropic_turns(messages: list[dict]) -> list[dict[str, Any]]:
    blocks = [b for m in messages if isinstance(m["content"], list) for b in m["content"]]
    results = {b["tool_use_id"]: parse(b["content"]) for b in blocks if b["type"] == "tool_result"}
    outputs = {b["tool_use_id"]: b for b in blocks if b["type"] == "code_execution_tool_result"}
    turns = []
    for m in messages:
        if m["role"] != "assistant":
            continue
        content = m["content"]
        turns.append(
            {
                "text": "\n\n".join(b["text"] for b in content if b["type"] == "text"),
                "reasoning": "\n\n".join(
                    b["thinking"] for b in content if b["type"] == "thinking" and b["thinking"]
                ),
                "calls": [
                    {"tool": b["name"], "args": b.get("input"), "result": results.get(b["id"])}
                    for b in content
                    if b["type"] == "tool_use"
                ],
                "code": [
                    {"code": (b.get("input") or {}).get("code", ""), "output": code_output(o)}
                    for b in content
                    if b["type"] == "server_tool_use"
                    for o in [outputs.get(b["id"])]
                ],
            }
        )
    return turns


def statement(turns: list[dict[str, Any]], calls: list[dict], marks: list[int]) -> list[dict]:
    """A code-mode conversation, turn by turn: the code each turn ran with its output, and the
    tool calls that code made, attributed to a turn by where the bridge log stood at each
    request. In our sandbox the code is run_python's argument and the output its result."""
    by_turn: dict[int, list[dict[str, str]]] = {}
    for index, call in enumerate(calls):
        turn = bisect.bisect_right(marks, index)
        by_turn.setdefault(turn, []).append({"tool": call["tool"], "line_id": call["line_id"]})
    shaped = []
    for number, turn in enumerate(turns, 1):
        cells = [
            {"code": (c["args"] or {}).get("code", ""), "output": c["result"]}
            for c in turn["calls"]
            if c["tool"] == "run_python" and isinstance(c["args"], dict)
        ]
        shaped.append(
            {
                "text": turn["text"],
                "reasoning": turn["reasoning"],
                "code": turn["code"] + cells,
                "calls": by_turn.get(number, []),
            }
        )
    return shaped


def outcome(label: synth.Label, decision: score.Decision) -> str:
    """The agent decides a match or an escalation, never a bank rule."""
    if score.is_correct(label, decision):
        return "correct"
    return "sent_to_person" if decision.decision == "escalate" else "wrong_match"


def build(cassette: Path, recording: dict[str, Any], readme: Path = README) -> dict[str, Any]:
    """Replay one slice with one worker, so the bridge log reads in the order calls were made."""
    arm = recording.get("arm", "direct")
    size = recording.get("statement_size") or "all"
    spec = next(s for s in synth.SPECS if s.name == recording["business"])
    dataset = synth.generate(spec)
    model_spec = llm.MODELS[recording["model"]]
    model = evaluate.make_model(model_spec, "replay", cassette)
    conversations: dict[int, list[dict]] = {}
    marks: dict[int, list[int]] = {}
    calls: list[dict[str, Any]] = []

    def capture(messages: list[dict]) -> None:
        conversations.setdefault(id(messages), messages)
        marks.setdefault(id(messages), []).append(len(calls))

    model.on_request = capture
    Bridge.log = calls
    try:
        run = evaluate.run_once(
            dataset,
            model,
            "export",
            recording["limit"],
            workers=1,
            trace_dir=None,
            arm=arm,
            statement_size=size,
        )
    finally:
        Bridge.log = None

    turns_of = anthropic_turns if model_spec.api == "anthropic" else openai_turns
    defences = agent.DEFENCES["on"]
    lines = [line for line in dataset.lines if line.id in run.agent_results]
    by_message = {agent.user_message(line, defences): line.id for line in lines}
    per_line: dict[str, list[dict]] = {}
    shaped, statement_marks = None, []
    for key, messages in conversations.items():
        first = next(m["content"] for m in messages if m["role"] == "user")
        if first in by_message:
            per_line[by_message[first]] = turns_of(messages)
        else:  # one conversation for the whole statement
            statement_marks = marks[key]
            shaped = statement(turns_of(messages), calls, statement_marks)

    def calls_of(line_id: str) -> list[dict[str, Any]] | None:
        """In code mode, the line's calls as the bridge answered them."""
        if shaped is None:
            return None
        return [
            {
                "turn": bisect.bisect_right(statement_marks, index),
                "tool": call["tool"],
                "args": call["args"],
                "result": call["result"],
                "refusal": call["refusal"],
                "from_code": call["caller"] != "direct",
            }
            for index, call in enumerate(calls)
            if call["line_id"] == line_id
        ]

    labels = {label.line_id: label for label in dataset.labels}
    exported = []
    for line in lines:
        label, decision = labels[line.id], run.decisions[line.id]
        verdict = run.verdicts[line.id]
        exported.append(
            {
                "id": line.id,
                "amount": tools.dollars(line.amount_cents),
                "message": agent.user_message(line, defences),
                "turns": per_line.get(line.id),
                "calls": calls_of(line.id),
                "decision": {
                    "decision": decision.decision,
                    "invoice_ids": list(decision.invoice_ids),
                    "rationale": decision.rationale,
                },
                "policy": {"auto": verdict.auto, "reason": verdict.reason},
                "label": {
                    "case": label.case,
                    "expected": label.expected,
                    "invoice_ids": list(label.invoice_ids),
                },
                "attack": {
                    "goal": label.attack_goal,
                    "targets": list(label.attack_targets),
                    "succeeded": evaluate.attack_hit(label, decision, verdict),
                }
                if label.attack_goal
                else None,
                "outcome": outcome(label, decision),
            }
        )
    return tidy(
        {
            "schema": SCHEMA,
            "slice": cassette.stem,
            "business": recording["business"],
            "dataset_fingerprint": recording["dataset_fingerprint"],
            "model": recording["model"],
            "arm": arm,
            "statement_size": None if arm == "direct" else size,
            "recorded_at": recording["generated_at"],
            "recorded_commit": recording["commit"],
            "results_at": f"v{version('bank-reconciliation-agent')}",
            "readme_headline": headline(readme),
            "statement": shaped,
            "lines": exported,
        }
    )


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write(out_dir: Path = OUT, cassettes: Path = CASSETTES, check: bool = False) -> list[str]:
    """Rebuild every slice's file; returns the slices whose file was missing or different.
    With check, nothing is written."""
    stale = []
    for name in SLICES:
        recording = json.loads((cassettes / f"{name}.json").read_text())
        text = render(build(cassettes / f"{name}.jsonl", recording))
        path = out_dir / f"{name}.json"
        if not path.exists() or path.read_text() != text:
            stale.append(name)
            if not check:
                out_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
    return stale
