"""Least privilege lives in Postgres, not in prompt text, so test it against a real database."""

import os

import psycopg
import pytest

from recon import db, rules, score, synth


@pytest.fixture(scope="module")
def conn():
    try:
        c = db.connect()
    except psycopg.OperationalError:
        if os.environ.get("CI"):
            raise
        pytest.skip("Postgres not reachable; run `docker compose up -d --wait`")
    db.reset(c, synth.generate(synth.SPECS[0]))
    yield c
    c.close()


def count(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_load_matches_the_dataset(conn):
    ds = synth.generate(synth.SPECS[0])
    assert count(conn, "ledger.contact") == len(ds.contacts)
    assert count(conn, "ledger.invoice") == len(ds.invoices)
    assert count(conn, "ledger.bank_line") == count(conn, "eval.label") == len(ds.lines)


def test_agent_reads_the_ledger_and_queues_proposals(conn):
    with db.as_role(conn, "recon_agent"):
        assert count(conn, "ledger.bank_line") == 250
        conn.execute(
            "INSERT INTO queue.proposal (run_id, bank_line_id, decision, source)"
            " VALUES ('test', 'A-L0001', 'escalate', 'test')"
        )


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE ledger.invoice SET status = 'paid'",
        "DELETE FROM ledger.bank_line",
        "INSERT INTO ledger.allocation (bank_line_id, rule) VALUES ('A-L0001', 'x')",
        "SELECT count(*) FROM eval.label",
        "SELECT count(*) FROM audit.approval",
        "UPDATE queue.proposal SET decision = 'match'",
    ],
)
def test_agent_cannot_write_the_ledger_or_read_labels(conn, statement):
    with pytest.raises(psycopg.errors.InsufficientPrivilege), db.as_role(conn, "recon_agent"):
        conn.execute(statement)


def test_reviewer_posts_and_audit_is_append_only(conn):
    with db.as_role(conn, "recon_reviewer"):
        proposal = conn.execute("SELECT min(id) FROM queue.proposal").fetchone()[0]
        conn.execute(
            "INSERT INTO audit.approval (proposal_id, decision) VALUES (%s, 'approved')",
            [proposal],
        )
        conn.execute("UPDATE ledger.invoice SET status = 'paid' WHERE id = 'INV-1001'")
    for statement in ("UPDATE audit.approval SET note = 'x'", "DELETE FROM audit.approval"):
        with pytest.raises(psycopg.errors.RaiseException), conn.transaction():
            conn.execute(statement)  # even the owner cannot rewrite history
    with pytest.raises(psycopg.errors.RaiseException), conn.transaction():
        conn.execute("TRUNCATE audit.approval CASCADE")
    assert count(conn, "audit.approval") == 1


def test_role_switch_rejects_unknown_roles(conn):
    with pytest.raises(ValueError), db.as_role(conn, "postgres"):
        pass


@pytest.mark.parametrize("spec", synth.SPECS, ids=lambda s: s.name)
def test_rules_baseline_is_never_wrong(conn, spec):
    ds = synth.generate(spec)
    db.reset(conn, ds)
    decisions = rules.run(conn)
    result = score.score(ds.labels, decisions)
    assert result["wrong_matches"] == 0
    assert result["cases"]["exact"]["correct"] == result["cases"]["exact"]["n"]
    assert result["cases"]["bank_rule"]["correct"] == result["cases"]["bank_rule"]["n"]
    attacked = {label.line_id for label in ds.labels if label.attack_goal}
    assert not attacked & decisions.keys(), "an attack never reaches the model"
