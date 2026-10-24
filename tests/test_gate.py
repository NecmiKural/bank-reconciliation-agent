"""Replay each committed slice through the whole pipeline and hold it to the gate: the per-line
agent's slice, and code mode's, which also runs the recorded code in the sandbox again."""

import json
import os
import re
import subprocess
from pathlib import Path

import psycopg
import pytest

from recon import db, evaluate, gate, llm, sandbox, synth

CASSETTES = Path(__file__).parent / "cassettes"
SLICES = ["A-slice", "A-slice-sandbox"]


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
    if recording.get("arm", "direct") == "sandbox" and not sandbox_ready():
        if os.environ.get("CI"):
            pytest.fail(f"{sandbox.IMAGE} missing in CI")
        pytest.skip(f"Docker or {sandbox.IMAGE} not available; run `make sandbox-image`")
    return CASSETTES / f"{request.param}.jsonl", recording


def replay(cassette, recording):
    return evaluate.evaluate(
        "A",
        "deepseek-flash",
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
    agent, and in the code a code-mode cell runs."""
    count = 0
    for call in entry["response"]["choices"][0]["message"].get("tool_calls") or []:
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
    for path in (CASSETTES / f"{name}.jsonl", CASSETTES / f"{name}.json"):
        text = path.read_text()
        assert not re.search(r"\bsk-[A-Za-z0-9]{16,}|Bearer |[Aa]uthorization", text), path
        assert not [value for value in planted if value in text], path


def test_recon_run_replays_the_slice_end_to_end(recorded, tmp_path, monkeypatch, capsys):
    """`make demo` runs this; it must reach the end and print the run id for review."""
    from recon import cli

    cassette, recording = recorded
    monkeypatch.chdir(tmp_path)  # traces go to a scratch folder
    argv = ["run", "--mode", "replay", "--cassette", str(cassette), "--limit", "40"]
    argv += ["--workers", "4", "--arm", recording.get("arm", "direct")]
    cli.main(argv)
    out = capsys.readouterr().out
    assert "review with: recon review list --run A-deepseek-flash-" in out
    assert "BAD" not in out


def test_payer_text_is_printed_without_control_characters():
    from recon.cli import printable

    assert printable("ACME\x1b[2J LTD\x07") == "ACME[2J LTD"
