"""README numbers come only from results files, and CI holds them to that."""

from pathlib import Path

import pytest

from recon import report
from recon.score import Decision

ROOT = Path(__file__).resolve().parents[1]


def test_readme_results_match_the_results_files():
    stale = report.sync(ROOT / "README.md", ROOT / "results", check=True)
    assert not stale, "README results are stale: run `uv run recon report`"


@pytest.mark.parametrize(
    ("compact", "expected"),
    [
        ("match:INV-1,INV-2", Decision("L", "match", ("INV-1", "INV-2"))),
        ("bank_rule:bank_fee", Decision("L", "bank_rule", rule="bank_fee")),
        ("escalate", Decision("L", "escalate")),
    ],
)
def test_compact_decisions_round_trip(compact, expected):
    assert report.decision("L", compact) == expected


def test_only_full_valid_live_runs_feed_the_readme():
    base = {
        "business": "A",
        "model": report.PRIMARY_MODEL,
        "defences": {"name": "on"},
        "mode": "record",
        "limit": None,
    }
    runs = [
        {**base, "valid": True, "generated_at": "2026-10-18T01:00:00+00:00"},
        {**base, "valid": True, "generated_at": "2026-10-19T01:00:00+00:00", "mode": "replay"},
        {**base, "valid": True, "generated_at": "2026-10-20T01:00:00+00:00", "limit": 40},
        {**base, "valid": False, "generated_at": "2026-10-21T01:00:00+00:00"},
        {**base, "valid": True, "generated_at": "2026-10-22T01:00:00+00:00", "model": "other"},
    ]
    assert report.pick(runs, "A", "on")["generated_at"].startswith("2026-10-18")
