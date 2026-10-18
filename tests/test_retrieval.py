"""The retrieval ablation's ranking and scoring, and its SQL against a real pgvector."""

import hashlib
import os

import psycopg
import pytest

from recon import db, retrieval, synth


def fake_embed(texts):
    """Deterministic 768-d vectors from character trigrams, so similar names stay close."""
    out = {}
    for text in texts:
        v = [0.0] * 768
        padded = f"  {text.lower()} "
        for i in range(len(padded) - 2):
            v[int(hashlib.md5(padded[i : i + 3].encode()).hexdigest(), 16) % 768] += 1.0
        out[text] = v
    return out


@pytest.mark.parametrize("payee", ["SMITH AND SONS LTD", "PAUL REID", "ACME", "ROBSON GROUP"])
def test_renaming_is_deterministic_and_changes_the_name(payee):
    assert retrieval.rename(payee, "L1") == retrieval.rename(payee, "L1")
    assert retrieval.rename(payee, "L1") != payee


def test_rrf_fuses_trigram_and_vector_ranks():
    rows = [  # id, amount, issued, trigram, fts, vector
        ("I1", 100, "2026-01-01", 0.9, 0.0, 0.9),
        ("I2", 100, "2026-01-02", 0.5, 0.0, 0.1),
        ("I3", 100, "2026-01-03", 0.1, 0.0, 0.5),
        ("I4", 999, "2026-01-04", 0.0, 0.0, 0.0),
    ]
    ranked = retrieval.ranks(rows, amount=999)
    assert ranked["trigram"] == ["I4", "I1", "I2", "I3"]  # an exact amount comes first
    assert ranked["vector"] == ["I4", "I1", "I3", "I2"]
    assert ranked["rrf"] == ["I4", "I1", "I2", "I3"]  # equal fused scores tie-break by id


def test_metrics_score_the_last_and_first_labelled_invoice():
    places = [retrieval.score_ranks(["A", "B", "C"], {"A", "C"}), (1, 1)]
    assert places[0] == (3, 1)
    result = retrieval.metrics(places)
    assert result["recall_at_5"] == 1.0 and result["mrr"] == 1.0


def test_ablation_runs_against_pgvector():
    try:
        conn = db.connect()
    except psycopg.OperationalError:
        if os.environ.get("CI"):
            raise
        pytest.skip("Postgres not reachable; run `docker compose up -d --wait`")
    with conn:
        result = retrieval.ablate(conn, synth.generate(synth.SPECS[0]), fake_embed, 768)
    for queries in ("as_written", "renamed"):
        for method in retrieval.METHODS:
            assert result[queries][method]["n"] == 215
            assert result[queries][method]["recall_at_25"] > 0.5
