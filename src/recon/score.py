"""Score decisions against ground-truth labels, with Wilson intervals on every rate."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from recon.synth import CASES, Label

Z95 = 1.959964


@dataclass(frozen=True)
class Decision:
    """What the rules pass or the agent decided for one bank line."""

    line_id: str
    decision: str  # "match", "bank_rule" or "escalate"
    invoice_ids: tuple[str, ...] = ()
    rule: str = ""
    source: str = "rules"
    rationale: str = ""


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_correct(label: Label, decision: Decision) -> bool:
    if decision.decision != label.expected:
        return False
    if label.expected == "match":
        return set(decision.invoice_ids) == set(label.invoice_ids)
    if label.expected == "bank_rule":
        return decision.rule == label.rule
    return True


def _r(x: float) -> float:
    return round(x, 4)


def score(labels: Iterable[Label], decisions: dict[str, Decision]) -> dict:
    """A missing decision counts as an escalation to a human."""
    per_case: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct = wrong_matches = escalated = proposed = proposed_right = expected_matches = 0
    n = 0
    for label in labels:
        decision = decisions.get(label.line_id) or Decision(label.line_id, "escalate")
        ok = is_correct(label, decision)
        n += 1
        correct += ok
        per_case[label.case][0] += 1
        per_case[label.case][1] += ok
        escalated += decision.decision == "escalate"
        expected_matches += label.expected == "match"
        if decision.decision == "match":
            proposed += 1
            proposed_right += ok
            wrong_matches += not ok
    precision = proposed_right / proposed if proposed else 0.0
    recall = proposed_right / expected_matches if expected_matches else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": n,
        "correct": correct,
        "accuracy": _r(correct / n) if n else 0.0,
        "escalated": escalated,
        "wrong_matches": wrong_matches,
        "match": {
            "proposed": proposed,
            "precision": _r(precision),
            "recall": _r(recall),
            "f1": _r(f1),
            "precision_ci95": [_r(x) for x in wilson(proposed_right, proposed)],
            "recall_ci95": [_r(x) for x in wilson(proposed_right, expected_matches)],
        },
        "cases": {
            case: {
                "n": per_case[case][0],
                "correct": per_case[case][1],
                "rate": _r(per_case[case][1] / per_case[case][0]),
                "ci95": [_r(x) for x in wilson(per_case[case][1], per_case[case][0])],
            }
            for case in CASES
            if per_case[case][0]
        },
    }


def paired_bootstrap(
    a: dict[str, list[bool]],
    b: dict[str, list[bool]],
    samples: int = 10_000,
    seed: int = 20261024,
) -> dict:
    """How much more often b was right than a, with a 95% interval: each line's correctness
    is averaged over its repeats, and lines, not line-runs, are resampled, so a line's repeats
    move together. Only lines both decided count."""
    lines = sorted(set(a) & set(b))
    diffs = [statistics.fmean(b[i]) - statistics.fmean(a[i]) for i in lines]
    if not diffs:
        return {"lines": 0, "difference": None, "ci95": [None, None]}
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(samples))
    low, high = means[int(0.025 * samples)], means[math.ceil(0.975 * samples) - 1]
    return {
        "lines": len(diffs),
        "difference": _r(statistics.fmean(diffs)),
        "ci95": [_r(low), _r(high)],
    }
