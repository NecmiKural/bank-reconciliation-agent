"""The agent's four tools, bound to one bank line.

No tool writes to the ledger. Every sum, difference and tolerance is computed here, so the
model only chooses. A proposal may cite only invoices that search_candidates returned in
the same conversation, and it is verified before it can reach the review queue.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import psycopg

from recon.db import as_role
from recon.score import Decision
from recon.synth import FEE_TOLERANCE_CENTS, BankLine

SEARCH_LIMIT = 25
MAX_COMBINATION_IDS = 25  # C(25, 3) = 2300 sums: bounded, and cheap
MAX_INVOICES_PER_MATCH = 3
# Trigram similarity between the bank line's payee and the invoice's payer below which a
# match needs the invoice number in the reference field. Measured on both synthetic
# businesses: every labelled answer clears it (tests/test_db.py).
PAYER_THRESHOLD = 0.3


@dataclass(frozen=True)
class Row:
    """An open invoice as the search returns it.

    payee_similarity compares the invoice's payer with the bank line's own payee, never with
    text the model supplied, so the model cannot steer the payer check.
    """

    invoice: str
    contact_id: str
    payee: str
    amount_cents: int
    issued: date
    payee_similarity: float


Search = Callable[[BankLine, str, str], list[Row]]


def _string(description: str) -> dict:
    return {"type": "string", "description": description}


def _ids(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


def _tool(name: str, description: str, properties: dict[str, dict]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    _tool(
        "search_candidates",
        "Find open invoices for this bank line: same amount, a matching invoice number, or a"
        " payer whose name is similar. Each result says how its amount relates to the line.",
        {
            "payee": _string("Payer name to look for, as you read it on the line."),
            "invoice_number": _string("An invoice number seen on the line, or an empty string."),
        },
    ),
    _tool(
        "find_combinations",
        "Find sets of 2 or 3 invoices of one payer that add up exactly to the line amount.",
        {"invoice_ids": _ids("Invoices returned by search_candidates, at most 25.")},
    ),
    _tool(
        "propose_match",
        "Propose matching the line to 1 to 3 invoices of one payer. Code checks the amounts;"
        " a human or the approval policy decides whether it is posted.",
        {
            "invoice_ids": _ids("Invoices returned by search_candidates."),
            "rationale": _string("One or two sentences for the reviewer. No personal data."),
        },
    ),
    _tool(
        "escalate",
        "Send the line to a human when no match is clearly supported.",
        {"reason": _string("Why, in one sentence. No personal data.")},
    ),
]


def dollars(cents: int, sign: bool = False) -> str:
    text = f"{abs(cents) / 100:,.2f}"
    if sign and cents > 0:
        return "+" + text
    return ("-" if cents < 0 else "") + text


def relation(invoice_cents: int, line_cents: int) -> str:
    difference = invoice_cents - line_cents
    if difference == 0:
        return "exact"
    if 0 < difference <= FEE_TOLERANCE_CENTS:
        return "fee_window"
    return "larger" if difference > 0 else "smaller"


def digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def names(line: BankLine, invoice_id: str) -> bool:
    """True if the reference or memo contains the invoice number as a whole token."""
    pattern = rf"(?<![A-Za-z0-9]){re.escape(invoice_id)}(?![A-Za-z0-9])"
    return re.search(pattern, f"{line.reference} {line.memo}", re.IGNORECASE) is not None


class Toolbox:
    def __init__(self, line: BankLine, search: Search) -> None:
        self.line = line
        self._search = search
        self.seen: dict[str, Row] = {}

    def call(self, name: str, arguments: str) -> tuple[dict, Decision | None]:
        handlers = {
            "search_candidates": self.search_candidates,
            "find_combinations": self.find_combinations,
            "propose_match": self.propose_match,
            "escalate": self.escalate,
        }
        if name not in handlers:
            return {"error": f"unknown tool {name!r}"}, None
        try:
            args = json.loads(arguments)
            return handlers[name](**args)
        except (json.JSONDecodeError, TypeError) as exc:
            return {"error": f"bad arguments for {name}: {exc}"}, None
        except psycopg.DataError as exc:
            return {"error": f"the database rejected that input: {type(exc).__name__}"}, None

    def search_candidates(self, payee: str, invoice_number: str) -> tuple[dict, None]:
        rows = self._search(self.line, payee, invoice_number)
        self.seen.update({r.invoice: r for r in rows})
        candidates = [
            {
                "invoice": r.invoice,
                "payee": r.payee,
                "amount": dollars(r.amount_cents),
                "issued": r.issued.isoformat(),
                "difference": dollars(r.amount_cents - self.line.amount_cents, sign=True),
                "relation": relation(r.amount_cents, self.line.amount_cents),
                "payee_similarity": round(r.payee_similarity, 2),
            }
            for r in rows
        ]
        return {"line_amount": dollars(self.line.amount_cents), "candidates": candidates}, None

    def _unseen(self, ids: list[str]) -> dict | None:
        unknown = [i for i in ids if i not in self.seen]
        if unknown:
            return {"error": f"{unknown} were not returned by search_candidates for this line"}
        return None

    def find_combinations(self, invoice_ids: list[str]) -> tuple[dict, None]:
        ids = list(dict.fromkeys(invoice_ids))
        if len(ids) > MAX_COMBINATION_IDS:
            return {"error": f"pass at most {MAX_COMBINATION_IDS} invoices"}, None
        if error := self._unseen(ids):
            return error, None
        rows = [self.seen[i] for i in ids]
        combos = [
            [r.invoice for r in combo]
            for size in (2, 3)
            for combo in itertools.combinations(rows, size)
            if len({r.contact_id for r in combo}) == 1
            and sum(r.amount_cents for r in combo) == self.line.amount_cents
        ]
        return {"exact_combinations": combos[:10]}, None

    def verify(self, ids: list[str]) -> str | None:
        """Why the books would not support this match, or None if they would."""
        if not ids:
            return "name at least one invoice"
        if len(set(ids)) != len(ids):
            return "an invoice is named twice"
        if len(ids) > MAX_INVOICES_PER_MATCH:
            return f"a payment can settle at most {MAX_INVOICES_PER_MATCH} invoices"
        if error := self._unseen(ids):
            return error["error"]
        rows = [self.seen[i] for i in ids]
        if len({r.contact_id for r in rows}) > 1:
            return "all invoices in one match must belong to the same payer"
        in_reference = any(self.line.reference == r.invoice for r in rows)
        if rows[0].payee_similarity < PAYER_THRESHOLD and not in_reference:
            return "the invoice belongs to a different payer than the one on the bank line"
        line = self.line.amount_cents
        if line <= 0:
            return "money out cannot settle customer invoices"
        difference = sum(r.amount_cents for r in rows) - line
        if len(rows) > 1 and difference != 0:
            return f"several invoices must add up exactly; they differ by {dollars(difference)}"
        if difference < 0:
            return "the line is larger than the invoice; overpayments go to a human"
        if difference > FEE_TOLERANCE_CENTS and not names(self.line, rows[0].invoice):
            return "a part payment needs remittance text naming the invoice"
        return None

    def propose_match(self, invoice_ids: list[str], rationale: str) -> tuple[dict, Decision | None]:
        if problem := self.verify(invoice_ids):
            return {"error": problem}, None
        decision = Decision(self.line.id, "match", tuple(invoice_ids), rationale=rationale)
        return {"status": "proposed"}, decision

    def escalate(self, reason: str) -> tuple[dict, Decision]:
        return {"status": "escalated"}, Decision(self.line.id, "escalate", rationale=reason)


SEARCH_SQL = r"""
SELECT i.id, i.contact_id, c.payee, i.amount_cents, i.issued,
       similarity(c.payee, %(line_payee)s) AS line_similarity,
       (%(digits)s <> '' AND regexp_replace(i.id, '\D', '', 'g') = %(digits)s) AS named,
       greatest(similarity(c.payee, %(payee)s), similarity(c.payee, %(line_payee)s)) AS score
FROM ledger.invoice i
JOIN ledger.contact c ON c.id = i.contact_id
WHERE i.status = 'open' AND i.issued <= %(posted)s
  AND (i.amount_cents = %(amount)s
       OR (%(digits)s <> '' AND regexp_replace(i.id, '\D', '', 'g') = %(digits)s)
       OR similarity(c.payee, %(payee)s) >= 0.3
       OR similarity(c.payee, %(line_payee)s) >= 0.3)
ORDER BY named DESC, (i.amount_cents = %(amount)s) DESC, score DESC, i.issued DESC, i.id
LIMIT %(limit)s
"""


def db_search(conn: psycopg.Connection) -> Search:
    """search_candidates backed by Postgres, reading as the agent role."""

    def search(line: BankLine, payee: str, invoice_number: str) -> list[Row]:
        params: dict[str, Any] = {
            "payee": payee,
            "line_payee": line.payee,
            "digits": digits(invoice_number),
            "amount": line.amount_cents,
            "posted": line.posted,
            "limit": SEARCH_LIMIT,
        }
        with as_role(conn, "recon_agent"):
            return [Row(*r[:6]) for r in conn.execute(SEARCH_SQL, params)]

    return search
