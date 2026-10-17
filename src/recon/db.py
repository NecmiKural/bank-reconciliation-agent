"""Postgres access: schema, loading a synthetic business, and least-privilege role switching."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files

import psycopg
from psycopg import sql

from recon.synth import Dataset

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
