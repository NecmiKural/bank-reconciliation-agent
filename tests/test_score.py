from datetime import date

import pytest

from recon.score import Decision, score, wilson
from recon.synth import Label


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [(5, 10, 0.2366, 0.7634), (0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (0, 0, 0.0, 1.0)],
)
def test_wilson_interval(k, n, low, high):
    assert wilson(k, n) == pytest.approx((low, high), abs=1e-4)


def label(line_id, case, expected, invoice_ids=(), rule=""):
    return Label(line_id, case, expected, invoice_ids, rule, date(2026, 1, 5), "", (), False, ())


LABELS = (
    label("L1", "exact", "match", ("I1",)),
    label("L2", "split", "match", ("I2", "I3")),
    label("L3", "partial", "match", ("I4",)),
    label("L4", "no_match", "escalate"),
    label("L5", "bank_rule", "bank_rule", rule="bank_fee"),
)


def test_score_counts_matches_escalations_and_wrong_matches():
    decisions = {
        "L1": Decision("L1", "match", ("I1",)),
        "L2": Decision("L2", "match", ("I3", "I2")),  # order does not matter
        "L3": Decision("L3", "match", ("I9",)),  # wrong invoice
        "L5": Decision("L5", "bank_rule", rule="bank_fee"),
    }  # L4 has no decision, which counts as an escalation
    result = score(LABELS, decisions)
    assert result["n"] == 5 and result["correct"] == 4 and result["wrong_matches"] == 1
    assert result["escalated"] == 1
    assert result["match"]["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["match"]["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["cases"]["partial"] == {"n": 1, "correct": 0, "rate": 0.0, "ci95": [0.0, 0.7935]}


def test_matching_an_unmatched_line_is_a_wrong_match():
    result = score(LABELS[3:4], {"L4": Decision("L4", "match", ("I1",))})
    assert result["wrong_matches"] == 1 and result["correct"] == 0
