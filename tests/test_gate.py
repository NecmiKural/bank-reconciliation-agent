"""Replay the committed slice through the whole pipeline and hold it to the gate."""

import json
import os
import re
from pathlib import Path

import psycopg
import pytest

from recon import db, evaluate, gate, synth

SLICE = Path(__file__).parent / "cassettes" / "A-slice.jsonl"
RECORDING = Path(__file__).parent / "cassettes" / "A-slice.json"


@pytest.fixture(scope="module")
def recording():
    try:
        db.connect().close()
    except psycopg.OperationalError:
        if os.environ.get("CI"):
            raise
        pytest.skip("Postgres not reachable; run `docker compose up -d --wait`")
    return json.loads(RECORDING.read_text())


def replay(cassette, recording):
    return evaluate.evaluate(
        "A", "deepseek-flash", 1, "replay", cassette, recording["limit"], 4, trace_dir=None
    )


def test_the_replayed_slice_passes_the_gate(recording):
    assert gate.check(replay(SLICE, recording), recording) == []


def test_a_doctored_cassette_fails_the_gate(recording, tmp_path):
    """Turn every recorded match proposal into an escalation; the gate must notice."""
    entries = [json.loads(line) for line in SLICE.read_text().splitlines()]
    doctored_calls = 0
    for entry in entries:
        for call in entry["response"]["choices"][0]["message"].get("tool_calls") or []:
            if call["function"]["name"] == "propose_match":
                call["function"] = {"name": "escalate", "arguments": '{"reason": "doctored"}'}
                doctored_calls += 1
    if not doctored_calls:
        pytest.fail("the slice has no recorded match to doctor")
    doctored = tmp_path / "doctored.jsonl"
    doctored.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
    problems = gate.check(replay(doctored, recording), recording)
    assert any("decisions_agent differ from the recording" in p for p in problems), problems


def test_committed_recordings_hold_no_secrets_or_personal_data():
    planted = evaluate.planted_pii(synth.generate(synth.SPECS[0]))
    for path in (SLICE, RECORDING):
        text = path.read_text()
        assert not re.search(r"\bsk-[A-Za-z0-9]{16,}|Bearer |[Aa]uthorization", text), path
        assert not [value for value in planted if value in text], path


def test_recon_run_replays_the_slice_end_to_end(recording, tmp_path, monkeypatch, capsys):
    """`make demo` runs this; it must reach the end and print the run id for review."""
    from recon import cli

    monkeypatch.chdir(tmp_path)  # traces go to a scratch folder
    cli.main(
        ["run", "--mode", "replay", "--cassette", str(SLICE), "--limit", "40", "--workers", "4"]
    )
    out = capsys.readouterr().out
    assert "review with: recon review list --run A-deepseek-flash-" in out
    assert "BAD" not in out


def test_payer_text_is_printed_without_control_characters():
    from recon.cli import printable

    assert printable("ACME\x1b[2J LTD\x07") == "ACME[2J LTD"
