"""Replay each committed slice through the whole pipeline and hold it to the gate: the per-line
agent's slice, code mode's, which also runs the recorded code in the sandbox again, and Claude's
two arms, per line and with programmatic tool calling."""

import json
import os
import re
import subprocess
from pathlib import Path

import psycopg
import pytest

from recon import db, evaluate, export, gate, llm, sandbox, synth

ROOT = Path(__file__).resolve().parents[1]
CASSETTES = Path(__file__).parent / "cassettes"
REPLAY = ROOT / "docs" / "replay"
SLICES = list(export.SLICES)


def sandbox_ready() -> bool:
    try:
        found = subprocess.run(
            ["docker", "image", "inspect", sandbox.IMAGE],
            env=sandbox.docker_env(),
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return found.returncode == 0


@pytest.fixture(scope="module", params=SLICES)
def recorded(request):
    try:
        db.connect().close()
    except psycopg.OperationalError:
        if os.environ.get("CI"):
            raise
        pytest.skip("Postgres not reachable; run `docker compose up -d --wait`")
    recording = json.loads((CASSETTES / f"{request.param}.json").read_text())
    if llm.MODELS[recording["model"]].api == "anthropic":
        pytest.importorskip("anthropic")  # an optional dependency: `uv sync --extra claude`
    if recording.get("arm", "direct") == "sandbox" and not sandbox_ready():
        if os.environ.get("CI"):
            pytest.fail(f"{sandbox.IMAGE} missing in CI")
        pytest.skip(f"Docker or {sandbox.IMAGE} not available; run `make sandbox-image`")
    return CASSETTES / f"{request.param}.jsonl", recording


def replay(cassette, recording):
    return evaluate.evaluate(
        "A",
        recording["model"],
        1,
        "replay",
        cassette,
        recording["limit"],
        4,
        trace_dir=None,
        arm=recording.get("arm", "direct"),
        statement_size=recording.get("statement_size") or "all",
    )


def test_the_replayed_slice_passes_the_gate(recorded):
    cassette, recording = recorded
    assert gate.check(replay(cassette, recording), recording) == []


def doctor(entry):
    """Turn a recorded match proposal into an escalation: in a tool call for the per-line
    agent, in the code a code-mode cell runs, and in a Claude tool call, made directly or
    from code."""
    count = 0
    for block in entry["response"].get("content") or []:  # Anthropic's Messages API
        if block.get("type") == "tool_use" and block.get("name") == "propose_match":
            block["name"] = "escalate"
            block["input"] = {"line_id": block["input"].get("line_id"), "reason": "doctored"}
            count += 1
    choices = entry["response"].get("choices") or [{"message": {}}]
    for call in choices[0]["message"].get("tool_calls") or []:
        function = call["function"]
        if function["name"] == "propose_match":
            call["function"] = {"name": "escalate", "arguments": '{"reason": "doctored"}'}
            count += 1
        elif function["name"] == "run_python" and "propose_match" in function["arguments"]:
            call["function"] = {**function, "arguments": json.dumps({"code": "pass"})}
            count += 1
    return count


def test_a_doctored_cassette_fails_the_gate(recorded, tmp_path):
    cassette, recording = recorded
    entries = [json.loads(line) for line in cassette.read_text().splitlines()]
    if not sum(doctor(entry) for entry in entries):
        pytest.fail("the slice has no recorded match to doctor")
    doctored = tmp_path / "doctored.jsonl"
    doctored.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
    try:
        problems = gate.check(replay(doctored, recording), recording)
    except llm.ReplayMiss:  # code mode: the doctored cell's output changed the next request
        problems = ["replay stopped: a request the recording does not hold"]
    assert problems, "a doctored recording passed the gate"


@pytest.mark.parametrize("name", SLICES)
def test_committed_recordings_hold_no_secrets_or_personal_data(name):
    planted = evaluate.planted_pii(synth.generate(synth.SPECS[0]))
    for path in (CASSETTES / f"{name}.jsonl", CASSETTES / f"{name}.json", REPLAY / f"{name}.json"):
        text = path.read_text()
        secret = r"\bsk-[A-Za-z0-9_-]{16,}|x-api-key|Bearer |[Aa]uthorization"
        assert not re.search(secret, text), path
        assert not [value for value in planted if value in text], path


def test_recon_run_replays_the_slice_end_to_end(recorded, tmp_path, monkeypatch, capsys):
    """`make demo` runs this; it must reach the end and print the run id for review."""
    from recon import cli

    cassette, recording = recorded
    monkeypatch.chdir(tmp_path)  # traces go to a scratch folder
    argv = ["run", "--mode", "replay", "--cassette", str(cassette), "--limit", "40"]
    argv += ["--workers", "4", "--arm", recording.get("arm", "direct")]
    argv += ["--model", recording["model"]]
    argv += ["--statement-size", recording.get("statement_size") or "all"]
    cli.main(argv)
    out = capsys.readouterr().out
    run_id = f"A-{evaluate.safe_name(recording['model'])}-"
    assert f"review with: recon review list --run {run_id}" in out
    agent = recording["runs"][0]["summary"]["agent"]
    bad = sum(line.startswith("BAD") for line in out.splitlines())
    assert bad == agent["n"] - agent["correct"]  # the lines the recording got wrong, no more


def test_the_committed_replay_export_matches_a_fresh_replay(recorded):
    cassette, recording = recorded
    committed = (REPLAY / f"{cassette.stem}.json").read_text()
    fresh = export.render(export.build(cassette, recording, ROOT / "README.md"))
    assert fresh == committed, "docs/replay is stale: run `make replay-export`"


@pytest.mark.parametrize("name", SLICES)
def test_the_replay_export_counts_what_the_recording_counted(name):
    """The viewer shows outcomes per line; together they must say what the recording said."""
    lines = json.loads((REPLAY / f"{name}.json").read_text())["lines"]
    summary = json.loads((CASSETTES / f"{name}.json").read_text())["runs"][0]["summary"]
    agent = summary["agent"]
    assert len(lines) == agent["n"]
    assert sum(line["outcome"] == "correct" for line in lines) == agent["correct"]
    assert sum(line["outcome"] == "wrong_match" for line in lines) == agent["wrong_matches"]
    assert sum(line["decision"]["decision"] == "escalate" for line in lines) == agent["escalated"]
    hits = [line for line in lines if line["attack"] and line["attack"]["succeeded"]]
    model_hits = [line for line in hits if line["attack"]["goal"] in evaluate.MODEL_GOALS]
    assert len(model_hits) == summary["attacks"]["model_decided"]["all"]["success"]


@pytest.mark.parametrize("name", SLICES)
def test_each_replay_file_stays_small(name):
    assert (REPLAY / f"{name}.json").stat().st_size < 400_000


def test_the_viewer_never_parses_a_string_as_markup():
    """Recorded model text and planted payloads are hostile by design: text nodes only."""
    code = (REPLAY / "app.js").read_text()
    for sink in ("innerHTML", "insertAdjacentHTML", "outerHTML"):
        assert sink not in code, sink


def test_the_viewer_copy_avoids_overclaiming_phrases():
    banned = ("production", "injection-proof", "robust to prompt injection", "worse model")
    for name in ("index.html", "app.js", "README.md"):
        text = (REPLAY / name).read_text().lower()
        assert not [phrase for phrase in banned if phrase in text], name


def test_payer_text_is_printed_without_control_characters():
    from recon.cli import printable

    assert printable("ACME\x1b[2J LTD\x07") == "ACME[2J LTD"
