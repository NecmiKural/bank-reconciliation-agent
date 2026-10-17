"""The rules pass closes only lines with exactly one defensible answer."""

from datetime import date

from recon.rules import BANK_RULES, Candidate, decide
from recon.synth import BankLine


def line(payee="ACME LTD", reference="", amount=10_000):
    return BankLine("L1", date(2026, 3, 2), amount, payee, reference, "Please pay INV-9999")


def candidate(invoice_id="INV-1001", similarity=1.0):
    return Candidate(invoice_id, "C1", "ACME LTD", 10_000, date(2026, 2, 1), similarity)


def test_single_candidate_with_matching_payee_is_closed():
    decision = decide(line(), [candidate()], BANK_RULES)
    assert decision.decision == "match" and decision.invoice_ids == ("INV-1001",)


def test_exact_reference_closes_a_misspelt_payee():
    decision = decide(line(reference="INV-1001"), [candidate(similarity=0.1)], BANK_RULES)
    assert decision.invoice_ids == ("INV-1001",)


def test_weak_payee_without_reference_is_left_for_the_agent():
    assert decide(line(), [candidate(similarity=0.2)], BANK_RULES) is None


def test_same_amount_twins_are_left_for_the_agent():
    twins = [candidate("INV-1001"), candidate("INV-1002", similarity=0.1)]
    assert decide(line(), twins, BANK_RULES) is None


def test_no_candidate_is_left_for_the_agent():
    assert decide(line(), [], BANK_RULES) is None


def test_memo_is_never_used():
    # The memo names INV-9999, which is not a candidate; rules must not act on payer text.
    assert decide(line(), [], BANK_RULES) is None


def test_owner_bank_rules_match_by_payee():
    decision = decide(line(payee="MONTHLY A/C FEE", amount=-500), [], BANK_RULES)
    assert decision.decision == "bank_rule" and decision.rule == "bank_fee"
