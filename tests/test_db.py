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


@pytest.mark.parametrize("spec", synth.SPECS, ids=lambda s: s.name)
def test_every_labelled_answer_passes_the_payer_check(conn, spec):
    """The verifier's payer threshold must never reject a correct answer."""
    from recon.tools import PAYER_THRESHOLD

    ds = synth.generate(spec)
    lines = {line.id: line for line in ds.lines}
    invoices = {i.id: i for i in ds.invoices}
    payees = {c.id: c.payee for c in ds.contacts}
    for label in ds.labels:
        if not label.invoice_ids:
            continue
        line = lines[label.line_id]
        payer = payees[invoices[label.invoice_ids[0]].contact_id]
        sim = conn.execute("SELECT similarity(%s, %s)", [payer, line.payee]).fetchone()[0]
        assert sim >= PAYER_THRESHOLD or line.reference in label.invoice_ids, (label, sim)


def test_review_posts_approved_matches_and_audits_every_decision(conn):
    from recon.policy import Verdict
    from recon.score import Decision

    ds = synth.generate(synth.SPECS[0])
    db.reset(conn, ds)
    match = next(label for label in ds.labels if label.invoice_ids)
    other, auto = [line.id for line in ds.lines if line.id != match.line_id][:2]
    db.queue(
        conn,
        "r1",
        [
            (Decision(match.line_id, "match", match.invoice_ids), Verdict(False, "first payer")),
            (Decision(other, "escalate", rationale="unsure"), Verdict(False, "escalated")),
            (Decision(auto, "bank_rule", rule="bank_fee"), Verdict(True, "owner bank rule")),
        ],
    )
    waiting = {row[1]: row[0] for row in db.pending(conn, "r1")}
    assert set(waiting) == {match.line_id, other}  # the auto verdict needs no human

    db.review(conn, waiting[match.line_id], approve=True, note="checked remittance")
    paid = conn.execute(
        "SELECT count(*) FROM ledger.invoice WHERE status = 'paid' AND id = ANY(%s)",
        [list(match.invoice_ids)],
    ).fetchone()[0]
    assert paid == count(conn, "ledger.allocation") == len(match.invoice_ids)
    with pytest.raises(ValueError, match="already decided"):
        db.review(conn, waiting[match.line_id], approve=True)
    with pytest.raises(ValueError, match="nothing to post"):
        db.review(conn, waiting[other], approve=True)
    db.review(conn, waiting[other], approve=False, note="unknown payer")
    assert db.pending(conn, "r1") == []
    assert count(conn, "audit.approval") == 2


def test_approving_a_part_payment_leaves_the_invoice_open(conn):
    from recon.policy import Verdict
    from recon.score import Decision

    ds = synth.generate(synth.SPECS[0])
    db.reset(conn, ds)
    lines = {line.id: line for line in ds.lines}
    part = next(label for label in ds.labels if label.case == "partial")
    db.queue(conn, "r2", [(Decision(part.line_id, "match", part.invoice_ids), Verdict(False, "x"))])
    (pid,) = [row[0] for row in db.pending(conn, "r2")]
    db.review(conn, pid, approve=True, actor="tester")
    status, allocated = conn.execute(
        "SELECT i.status, a.amount_cents FROM ledger.invoice i"
        " JOIN ledger.allocation a ON a.invoice_id = i.id WHERE i.id = %s",
        [part.invoice_ids[0]],
    ).fetchone()
    assert status == "open" and allocated == lines[part.line_id].amount_cents
    actor = conn.execute("SELECT actor FROM audit.approval").fetchone()[0]
    assert actor == "tester"
