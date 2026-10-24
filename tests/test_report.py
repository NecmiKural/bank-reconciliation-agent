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
        {**base, "valid": True, "generated_at": "2026-10-23T01:00:00+00:00", "arm": "sandbox"},
    ]
    assert report.pick(runs, "A", "on")["generated_at"].startswith("2026-10-18")
    assert all(report.per_line(r) for r in report.full_runs(runs))


def code_mode_results():
    """A per-line run and a code-mode run of business A on two open lines, built by hand."""
    from recon import evaluate, synth

    ds = synth.generate(synth.SPECS[0])
    labels = [x for x in ds.labels if x.expected == "match" and x.case == "split"][:2]
    right = {x.line_id: "match:" + ",".join(sorted(x.invoice_ids)) for x in labels}
    wrong = {labels[0].line_id: "escalate", labels[1].line_id: right[labels[1].line_id]}

    def result(arm, decided, requests):
        summary = {"model_requests_per_100_agent_lines": requests, "statement": None}
        if arm != "direct":
            summary["statement"] = {"code_failures": 1}
        totals = {
            "wrong_matches": 0,
            "model_attacks": {"lines_hit_in_any_repeat": 0, "distinct_lines": 20},
        }
        return {
            "business": "A",
            "model": report.PRIMARY_MODEL,
            "arm": arm,
            "statement_size": None if arm == "direct" else "all",
            "defences": {"name": "on"},
            "mode": "record",
            "limit": None,
            "valid": True,
            "repeats": 1,
            "commit": "abc1234",
            "generated_at": "2026-10-25T01:00:00+00:00",
            "dataset_fingerprint": evaluate.fingerprint(ds),
            "aggregate": {
                "cost_per_100_bank_lines_usd": {"mean": 0.05},
                "agent_accuracy": {"mean": 1.0},
                "totals": totals,
            },
            "runs": [{"summary": summary, "decisions_agent": decided, "decisions_rules": {}}],
        }

    return result("direct", wrong, 270.0), result("sandbox", right, 20.0)


def test_a_code_mode_cell_is_compared_with_the_same_models_per_line_run():
    direct, code = code_mode_results()
    rows = report.code_mode_table([direct, code])
    assert len(rows) == 3
    row = rows[2]
    assert "our sandbox" in row and "| 20.0 |" in row
    assert "+0.500 (" in row  # one of the two lines went from escalated to right
    assert row.endswith("| 0 | 1 |")


def test_a_code_mode_cell_whose_rules_pass_differs_is_refused():
    direct, code = code_mode_results()
    code["runs"][0]["decisions_rules"] = {"A-L0001": "escalate"}
    with pytest.raises(ValueError, match="rules pass"):
        report.code_mode_table([direct, code])
