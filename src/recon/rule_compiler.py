"""Turn an owner's plain-English bank rule into a validated rule, or refuse.

The model drafts JSON; code decides. A draft becomes a rule only if it parses, names a known
account with the direction that account allows, stays inside the policy's envelope for that
account, is specific enough, and matches no customer the ledger knows. The model writes the
amount as the owner gave it; code turns it into cents.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from recon import llm, synth
from recon.policy import BANK_RULE_ENVELOPES
from recon.score import wilson

PROMPT = files("recon").joinpath("prompts", "rule_compiler.md").read_text()
EVAL_SET = Path(__file__).resolve().parents[2] / "evals" / "rule_sentences.jsonl"
ACCOUNTS = {"bank_fee": "money_out", "merchant_fee": "money_out", "interest": "money_in"}
# Words that say nothing about which line is meant. A one-word pattern among them is refused.
GENERIC = {
    "BANK", "CARD", "CHARGE", "CHARGES", "CREDIT", "DEBIT", "DEPOSIT", "FEE", "FEES",
    "INTEREST", "INVOICE", "ONLINE", "PAYMENT", "PAYMENTS", "TRANSFER",
}  # fmt: skip
MIN_PATTERN = 4


class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    payee_contains: str
    account: str
    direction: str
    max_amount: str


class Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule: Draft | None
    reason: str


@dataclass(frozen=True)
class BankRule:
    payee_contains: str
    account: str
    direction: str
    max_amount_cents: int


@dataclass(frozen=True)
class Outcome:
    status: str  # accepted, refused_by_model, refused_by_validator or invalid
    rule: BankRule | None
    reason: str


def cents(text: str) -> int | None:
    """A plain decimal such as "12" or "12.50" in cents; anything else is not an amount."""
    if not re.fullmatch(r"\d{1,6}(\.\d{2})?", text.strip()):
        return None
    return int(Decimal(text.strip()) * 100)


def validate(draft: Draft, customer_payees: Iterable[str]) -> tuple[BankRule | None, str]:
    pattern = " ".join(draft.payee_contains.upper().split())
    if len(pattern) < MIN_PATTERN or pattern in GENERIC:
        return None, f"payee pattern {pattern!r} is too short or too generic"
    if draft.account not in ACCOUNTS:
        return None, f"unknown account {draft.account!r}"
    if draft.direction != ACCOUNTS[draft.account]:
        return None, f"{draft.account} lines are {ACCOUNTS[draft.account]}, not {draft.direction}"
    amount = cents(draft.max_amount)
    if amount is None or amount <= 0:
        return None, f"max_amount {draft.max_amount!r} is not a positive plain decimal"
    low, high = BANK_RULE_ENVELOPES[draft.account]
    if amount > max(abs(low), abs(high)):
        return None, f"a {draft.account} rule may cover at most {max(abs(low), abs(high))} cents"
    hits = sorted(p for p in customer_payees if pattern in p)
    if hits:
        return None, f"pattern would also match customer payments from {', '.join(hits[:3])}"
    return BankRule(pattern, draft.account, draft.direction, amount), ""


def compile_rule(sentence: str, model: llm.LLM, customer_payees: Iterable[str]) -> Outcome:
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": sentence}]
    reply = model.chat(messages, tools=[], response_format={"type": "json_object"})
    content = reply.response.choices[0].message.content or ""
    try:
        parsed = Reply.model_validate_json(content)
    except ValidationError as exc:
        return Outcome(
            "invalid", None, f"reply did not match the schema: {exc.error_count()} errors"
        )
    if parsed.rule is None:
        return Outcome("refused_by_model", None, parsed.reason)
    rule, problem = validate(parsed.rule, customer_payees)
    if rule is None:
        return Outcome("refused_by_validator", None, problem)
    return Outcome("accepted", rule, parsed.reason)


def load_eval_set(path: Path = EVAL_SET) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expected_rule(expect: dict | None) -> BankRule | None:
    if expect is None:
        return None
    rule, problem = validate(Draft(**expect), [])
    if rule is None:
        raise ValueError(f"eval set expects an invalid rule: {problem}")
    return rule


def customer_payees() -> set[str]:
    return {c.payee for spec in synth.SPECS for c in synth.generate(spec).contacts}


def score_run(items: list[dict], outcomes: list[Outcome]) -> dict[str, Any]:
    should_accept = [(i, o) for i, o in zip(items, outcomes, strict=True) if i["expect"]]
    should_refuse = [(i, o) for i, o in zip(items, outcomes, strict=True) if not i["expect"]]
    exact = sum(o.rule == expected_rule(i["expect"]) for i, o in should_accept)
    wrong = [i["id"] for i, o in should_accept if o.rule and o.rule != expected_rule(i["expect"])]
    unsafe = [i["id"] for i, o in should_refuse if o.status == "accepted"]
    statuses = [o.status for o in outcomes]
    return {
        "compiled_exactly": exact,
        "should_accept": len(should_accept),
        "falsely_refused": [i["id"] for i, o in should_accept if o.rule is None],
        "wrong_rules_accepted": wrong,
        "unsafe_rules_accepted": unsafe,
        "should_refuse": len(should_refuse),
        "refused_by_validator": statuses.count("refused_by_validator"),
        "refused_by_model": statuses.count("refused_by_model"),
        "invalid_replies": statuses.count("invalid"),
    }


def evaluate(model_name: str, repeats: int, mode: str, cassette: Path | None) -> dict[str, Any]:
    from recon.evaluate import git_commit

    items = load_eval_set()
    payees = customer_payees()
    runs = []
    for k in range(repeats):
        model = llm.LLM(
            llm.MODELS[model_name], mode, cassette, salt="" if k == 0 else f"repeat-{k}"
        )
        outcomes = [compile_rule(item["sentence"], model, payees) for item in items]
        runs.append(
            {
                "summary": score_run(items, outcomes),
                "outcomes": {
                    item["id"]: asdict(o) for item, o in zip(items, outcomes, strict=True)
                },
            }
        )
    bad = {i for run in runs for key in ("wrong_rules_accepted", "unsafe_rules_accepted")
           for i in run["summary"][key]}  # fmt: skip
    return {
        "rules": True,
        "model": model_name,
        "mode": mode,
        "repeats": repeats,
        "sentences": len(items),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "prompt": llm.request_key({"prompt": PROMPT})[:12],
        "bad_rules": {
            "distinct_sentences": sorted(bad),
            "upper95": round(wilson(len(bad), len(items))[1], 4),
        },
        "runs": runs,
    }
