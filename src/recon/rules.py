"""Deterministic first pass: close only the lines that have exactly one defensible answer.

Everything the rules leave open goes to the agent. The same pass, with nothing after it,
is the baseline every model is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import psycopg

from recon.db import as_role
from recon.score import Decision
from recon.synth import BankLine

# Trigram similarity at which a payee is taken to be the invoice's contact.
PAYEE_THRESHOLD = 0.5

# Bank rules the business owner writes: a payee substring and the account it books to.
BANK_RULES = {"A/C FEE": "bank_fee", "INTEREST": "interest", "MERCHANT": "merchant_fee"}

CANDIDATES_SQL = """
SELECT i.id, i.contact_id, c.payee, i.amount_cents, i.issued,
       similarity(c.payee, %(payee)s) AS payee_similarity
FROM ledger.invoice i
JOIN ledger.contact c ON c.id = i.contact_id
WHERE i.status = 'open' AND i.amount_cents = %(amount)s AND i.issued <= %(posted)s
ORDER BY payee_similarity DESC, i.id
"""


@dataclass(frozen=True)
class Candidate:
    invoice_id: str
    contact_id: str
    payee: str
    amount_cents: int
    issued: date
    payee_similarity: float


def decide(line: BankLine, candidates: list[Candidate], bank_rules: dict[str, str]):
    """A decision for an unambiguous line, or None to leave it for the agent.

    Only bank-written fields count: amount, payee and the reference field. The memo is
    payer-written free text and is never read here.
    """
    for needle, rule in bank_rules.items():
        if needle in line.payee:
            return Decision(line.id, "bank_rule", rule=rule)
    if len(candidates) != 1:
        return None  # no exact amount, or a same-amount twin that needs judgement
    (c,) = candidates
    if line.reference == c.invoice_id or c.payee_similarity >= PAYEE_THRESHOLD:
        return Decision(line.id, "match", (c.invoice_id,))
    return None


def run(conn: psycopg.Connection, bank_rules: dict[str, str] = BANK_RULES) -> dict[str, Decision]:
    """Decide every bank line the rules can close, reading the ledger as the agent role."""
    decisions = {}
    with as_role(conn, "recon_agent"):
        rows = conn.execute(
            "SELECT id, posted, amount_cents, payee, reference, memo FROM ledger.bank_line"
            " ORDER BY posted, id"
        ).fetchall()
        for row in rows:
            line = BankLine(*row)
            params = {"payee": line.payee, "amount": line.amount_cents, "posted": line.posted}
            candidates = [Candidate(*c) for c in conn.execute(CANDIDATES_SQL, params)]
            decision = decide(line, candidates, bank_rules)
            if decision:
                decisions[line.id] = decision
    return decisions
