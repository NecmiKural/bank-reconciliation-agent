"""Plain-English bank rules: the model drafts, code decides."""

import json

import pytest
from conftest import FakeClient, completion

from recon import llm
from recon.rule_compiler import (
    Draft,
    compile_rule,
    customer_payees,
    expected_rule,
    load_eval_set,
    validate,
)

PAYEES = {"TAYLOR LTD", "ADAM THORNTON"}


def draft(**kw):
    base = {"payee_contains": "MONTHLY A/C FEE", "account": "bank_fee"}
    base |= {"direction": "money_out", "max_amount": "10.00"}
    return Draft(**(base | kw))


def test_a_sound_draft_becomes_a_rule_with_cents_computed_in_code():
    rule, problem = validate(draft(payee_contains="monthly  a/c fee", max_amount="12.50"), PAYEES)
    assert problem == "" and rule.payee_contains == "MONTHLY A/C FEE"
    assert rule.max_amount_cents == 1250


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"payee_contains": "FEE"}, "too short or too generic"),
        ({"payee_contains": "payments"}, "too short or too generic"),
        ({"payee_contains": ""}, "too short or too generic"),
        ({"account": "income"}, "unknown account"),
        ({"direction": "money_in"}, "not money_in"),
        ({"account": "interest", "direction": "money_in", "max_amount": "50.00"}, "at most"),
        ({"max_amount": "5000.00"}, "at most"),
        ({"max_amount": "$25"}, "not a positive plain decimal"),
        ({"max_amount": "0"}, "not a positive plain decimal"),
        ({"payee_contains": "TAYLOR"}, "customer payments"),
    ],
)
def test_unsafe_drafts_are_refused_by_code(change, why):
    rule, problem = validate(draft(**change), PAYEES)
    assert rule is None and why in problem


def reply(content):
    fake = FakeClient(
        [
            completion(
                content=json.dumps(content) if isinstance(content, dict) else content, finish="stop"
            )
        ]
    )
    return llm.LLM(llm.MODELS["deepseek-flash"], client=fake), fake


def test_the_model_is_asked_for_json_and_its_draft_is_validated():
    model, fake = reply({"rule": draft().model_dump(), "reason": "monthly fee"})
    outcome = compile_rule("The monthly fee is MONTHLY A/C FEE, max $10.", model, PAYEES)
    assert outcome.status == "accepted" and outcome.rule.max_amount_cents == 1000
    assert fake.requests[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("content", "status"),
    [
        ({"rule": None, "reason": "not a bank rule"}, "refused_by_model"),
        (
            {"rule": draft(payee_contains="TAYLOR LTD").model_dump(), "reason": "x"},
            "refused_by_validator",
        ),
        ("not json", "invalid"),
        ({"rule": None, "reason": "x", "extra": 1}, "invalid"),
    ],
)
def test_refusals_and_bad_replies_are_never_rules(content, status):
    model, _ = reply(content)
    outcome = compile_rule("sentence", model, PAYEES)
    assert outcome.status == status and outcome.rule is None


def test_the_eval_set_is_achievable_and_balanced():
    items = load_eval_set()
    assert len(items) == 40 and sum(not i["expect"] for i in items) == 10
    payees = customer_payees()
    for item in items:
        rule = expected_rule(item["expect"])
        if rule is not None:
            assert validate(Draft(**item["expect"]), payees)[0] == rule, item["id"]
    assert "TAYLOR LTD" in payees  # the customer-capture sentence names a real contact
