"""Postgres access: schema, loading a synthetic business, and least-privilege role switching."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from importlib.resources import files

import psycopg
from psycopg import sql

from recon.policy import Verdict
from recon.score import Decision
from recon.synth import FEE_TOLERANCE_CENTS, Dataset

DEFAULT_URL = "postgresql://recon:recon@localhost:5433/recon"  # the compose.yaml service
ROLES = ("recon_agent", "recon_reviewer")


def connect(url: str | None = None) -> psycopg.Connection:
    url = url or os.environ.get("DATABASE_URL") or DEFAULT_URL
    return psycopg.connect(url, autocommit=True)


@contextmanager
def as_role(conn: psycopg.Connection, role: str) -> Iterator[psycopg.Connection]:
    """Run a block in its own transaction with only the privileges of `role`.

    NOTE: the owner connection switches role with SET LOCAL ROLE, which code that runs
    arbitrary SQL could undo. No tool here takes SQL from the model. A deployment would give
    the agent its own login that is a member of recon_agent only.
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    with conn.transaction():
        conn.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
        yield conn


def reset(conn: psycopg.Connection, dataset: Dataset) -> None:
    """Recreate every schema and load one synthetic business, labels included."""
    with conn.transaction():
        conn.execute(files("recon").joinpath("schema.sql").read_text())
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO ledger.contact VALUES (%s, %s, %s)",
                [(c.id, c.name, c.payee) for c in dataset.contacts],
            )
            cur.executemany(
                "INSERT INTO ledger.invoice (id, contact_id, issued, due, amount_cents)"
                " VALUES (%s, %s, %s, %s, %s)",
                [(i.id, i.contact_id, i.issued, i.due, i.amount_cents) for i in dataset.invoices],
            )
            cur.executemany(
                "INSERT INTO ledger.bank_line VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (b.id, b.posted, b.amount_cents, b.payee, b.reference, b.memo)
                    for b in dataset.lines
                ],
            )
            cur.executemany(
                "INSERT INTO eval.label VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        lab.line_id,
                        lab.case,
                        lab.expected,
                        list(lab.invoice_ids),
                        lab.rule,
                        lab.attack_goal,
                        list(lab.attack_targets),
                        lab.control,
                        json.dumps(lab.pii),
                    )
                    for lab in dataset.labels
                ],
            )


def queue(conn: psycopg.Connection, run_id: str, items: Iterable[tuple[Decision, Verdict]]) -> None:
    """Write decisions and policy verdicts to the review queue, as the agent role.

    This is the only write the agent role may make. Shadow mode stops here: an "auto"
    verdict records what the policy would approve, and nothing is posted.
    """
    rows = [
        (
            run_id,
            d.line_id,
            d.decision,
            list(d.invoice_ids),
            d.rule,
            d.source,
            d.rationale,
            "auto" if v.auto else "review",
            v.reason,
        )
        for d, v in items
    ]
    with as_role(conn, "recon_agent"), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO queue.proposal (run_id, bank_line_id, decision, invoice_ids, rule,"
            " source, rationale, policy, policy_note) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            rows,
        )


PENDING_SQL = """
SELECT p.id, p.bank_line_id, b.amount_cents, b.payee, p.decision, p.invoice_ids,
       p.policy_note, p.rationale
FROM queue.proposal p
JOIN ledger.bank_line b ON b.id = p.bank_line_id
WHERE p.run_id = %s AND p.policy = 'review'
  AND NOT EXISTS (SELECT FROM audit.approval a WHERE a.proposal_id = p.id)
ORDER BY b.posted, p.id
"""


def pending(conn: psycopg.Connection, run_id: str) -> list[tuple]:
    """Proposals of a run that wait for a human."""
    with as_role(conn, "recon_reviewer"):
        return conn.execute(PENDING_SQL, [run_id]).fetchall()


def review(
    conn: psycopg.Connection, proposal_id: int, approve: bool, note: str = "", actor: str = ""
) -> None:
    """Approve (and post) or reject one proposal, as the reviewer, with an audit row.

    A split must add up exactly. A single invoice is marked paid once its allocations come
    within the fee tolerance of its amount; a part payment leaves it open.

    NOTE: the already-decided check and the insert are not serialised, which is fine for one
    reviewer at a terminal. Concurrent reviewers would need a unique index on proposal_id.
    """
    with as_role(conn, "recon_reviewer"):
        row = conn.execute(
            "SELECT bank_line_id, decision, invoice_ids, rule FROM queue.proposal WHERE id = %s",
            [proposal_id],
        ).fetchone()
        if row is None:
            raise LookupError(f"no proposal {proposal_id}")
        line_id, decision, invoice_ids, rule = row
        decided = conn.execute(
            "SELECT 1 FROM audit.approval WHERE proposal_id = %s", [proposal_id]
        ).fetchone()
        if decided:
            raise ValueError(f"proposal {proposal_id} was already decided")
        if approve and decision == "escalate":
            raise ValueError("an escalation has nothing to post; reject it or match by hand")
        actor = actor or "reviewer"
        conn.execute(
            "INSERT INTO audit.approval (proposal_id, decision, actor, note)"
            " VALUES (%s, %s, %s, %s)",
            [proposal_id, "approved" if approve else "rejected", actor, note],
        )
        if not approve:
            return
        line_cents = conn.execute(
            "SELECT amount_cents FROM ledger.bank_line WHERE id = %s", [line_id]
        ).fetchone()[0]
        allocate = (
            "INSERT INTO ledger.allocation"
            " (bank_line_id, invoice_id, rule, amount_cents, posted_by)"
            " VALUES (%s, %s, %s, %s, %s)"
        )
        if decision == "bank_rule":
            conn.execute(allocate, [line_id, None, rule, line_cents, actor])
            return
        invoices = conn.execute(
            "SELECT id, amount_cents, status FROM ledger.invoice WHERE id = ANY(%s) FOR UPDATE",
            [invoice_ids],
        ).fetchall()
        not_open = sorted(set(invoice_ids) - {i for i, _, status in invoices if status == "open"})
        if not_open:
            raise ValueError(f"{', '.join(not_open)} is not open")
        if len(invoices) > 1:
            if sum(amount for _, amount, _ in invoices) != line_cents:
                raise ValueError("the invoices of a split must add up to the line")
            settled = [(i, amount) for i, amount, _ in invoices]
        else:
            ((invoice_id, amount, _),) = invoices
            before = conn.execute(
                "SELECT coalesce(sum(amount_cents), 0) FROM ledger.allocation"
                " WHERE invoice_id = %s",
                [invoice_id],
            ).fetchone()[0]
            settled = [(invoice_id, line_cents)]
            if amount - before - line_cents > FEE_TOLERANCE_CENTS:
                conn.execute(allocate, [line_id, invoice_id, None, line_cents, actor])
                return  # a part payment: the invoice stays open
        for invoice_id, cents in settled:
            conn.execute(allocate, [line_id, invoice_id, None, cents, actor])
            conn.execute("UPDATE ledger.invoice SET status = 'paid' WHERE id = %s", [invoice_id])
