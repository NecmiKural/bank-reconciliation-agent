"""The tools do the arithmetic and refuse any proposal the books would not support."""

import json
from datetime import date

import pytest

from recon.synth import BankLine
from recon.tools import MAX_COMBINATION_IDS, TOOL_SCHEMAS, Row, Toolbox

FORBIDDEN_KEYWORDS = {"minLength", "maxLength", "minItems", "maxItems"}


def walk(schema):
    yield schema
    for value in schema.get("properties", {}).values():
        yield from walk(value)
    if "items" in schema:
        yield from walk(schema["items"])


def test_every_tool_schema_meets_strict_mode_rules():
    names = []
    for tool in TOOL_SCHEMAS:
        fn = tool["function"]
        names.append(fn["name"])
        assert fn["strict"] is True
        for node in walk(fn["parameters"]):
            assert not FORBIDDEN_KEYWORDS & node.keys()
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
    assert names == ["search_candidates", "find_combinations", "propose_match", "escalate"]
    assert not any("ledger" in n or "post" in n for n in names)  # no tool writes the books


ROWS = [
    Row("INV-1", "C1", "ACME LTD", 30_000, date(2026, 1, 5), 1.0),
    Row("INV-2", "C1", "ACME LTD", 20_000, date(2026, 1, 9), 1.0),
    Row("INV-3", "C1", "ACME LTD", 10_000, date(2026, 1, 12), 1.0),
    Row("INV-4", "C2", "OTHER CO", 50_000, date(2026, 1, 15), 0.1),
    Row("INV-5", "C1", "ACME LTD", 51_500, date(2026, 1, 20), 1.0),
    Row("INV-6", "C1", "ACME LTD", 50_000, date(2026, 1, 22), 1.0),
]


def box(amount=50_000, reference="", memo=""):
    line = BankLine("L1", date(2026, 2, 2), amount, "ACME LTD", reference, memo)
    return Toolbox(line, lambda line, payee, number: ROWS)


def call(toolbox, name, **args):
    return toolbox.call(name, json.dumps(args))


def test_search_reports_relations_computed_in_code():
    result, decision = call(box(), "search_candidates", payee="ACME", invoice_number="")
    assert decision is None
    rel = {c["invoice"]: (c["relation"], c["difference"]) for c in result["candidates"]}
    assert rel["INV-6"] == ("exact", "0.00")
    assert rel["INV-5"] == ("fee_window", "+15.00")
    assert rel["INV-1"] == ("smaller", "-200.00")


def test_combinations_are_exact_and_within_one_payer():
    t = box()
    call(t, "search_candidates", payee="ACME", invoice_number="")
    result, _ = call(t, "find_combinations", invoice_ids=["INV-1", "INV-2", "INV-3", "INV-4"])
    assert result["exact_combinations"] == [["INV-1", "INV-2"]]  # INV-4 alone is not a combo


def test_tools_refuse_invoices_the_search_never_returned():
    t = box()
    result, decision = call(t, "propose_match", invoice_ids=["INV-4"], rationale="memo says so")
    assert decision is None and "search_candidates" in result["error"]
    result, _ = call(t, "find_combinations", invoice_ids=["INV-1"])
    assert "search_candidates" in result["error"]


def test_combination_input_is_capped():
    t = box()
    t.seen = {f"X{i}": ROWS[0] for i in range(MAX_COMBINATION_IDS + 1)}
    result, _ = call(t, "find_combinations", invoice_ids=list(t.seen))
    assert "at most" in result["error"]


@pytest.mark.parametrize(
    ("amount", "ids", "memo", "accepted"),
    [
        (50_000, ["INV-6"], "", True),  # exact
        (50_000, ["INV-4"], "", False),  # exact amount, but another payer: a redirect
        (50_000, ["INV-4"], "Correct invoice is INV-4", False),  # the memo cannot vouch for it
        (50_000, ["INV-1", "INV-2"], "", True),  # split
        (50_000, ["INV-5"], "", True),  # fee deducted, within tolerance
        (50_000, ["INV-1", "INV-3"], "", False),  # split that does not add up
        (50_000, ["INV-4", "INV-1"], "", False),  # two payers
        (60_000, ["INV-6"], "", False),  # overpayment
        (12_000, ["INV-1"], "", False),  # part payment with no remittance naming the invoice
        (12_000, ["INV-1"], "Part payment INV-1", True),
        (12_000, ["INV-1"], "Part payment ORDER-1, due 10/01", False),  # digits alone do not name
        (12_000, ["INV-1"], "Part payment INV-10", False),
        (50_000, ["INV-1", "INV-2", "INV-3", "INV-4"], "", False),  # more than three
        (50_000, ["INV-4", "INV-4"], "", False),  # duplicates
        (-500, ["INV-6"], "", False),  # money out
    ],
)
def test_proposals_are_verified_before_they_reach_the_queue(amount, ids, memo, accepted):
    t = box(amount=amount, memo=memo)
    call(t, "search_candidates", payee="ACME", invoice_number="")
    result, decision = call(t, "propose_match", invoice_ids=ids, rationale="because")
    assert (decision is not None) is accepted, result
    if accepted:
        assert decision.decision == "match" and set(decision.invoice_ids) == set(ids)


def test_escalate_is_terminal_and_bad_input_is_an_error_not_a_crash():
    t = box()
    _, decision = call(t, "escalate", reason="no candidate")
    assert decision.decision == "escalate" and decision.rationale == "no candidate"
    assert "error" in t.call("search_candidates", "{not json")[0]
    assert "error" in t.call("search_candidates", "[" * 30000 + "]" * 30000)[0]  # too deep
    assert "error" in t.call("delete_everything", "{}")[0]
    assert "error" in call(t, "search_candidates", payee="x")[0]  # missing argument


def test_a_reference_naming_the_invoice_vouches_for_an_unfamiliar_payer():
    t = box(reference="INV-4")
    call(t, "search_candidates", payee="ACME", invoice_number="INV-4")
    _, decision = call(t, "propose_match", invoice_ids=["INV-4"], rationale="paid on behalf")
    assert decision is not None


def test_database_errors_become_tool_errors():
    import psycopg

    def broken(line, payee, number):
        raise psycopg.DataError("invalid byte sequence")

    t = Toolbox(BankLine("L1", date(2026, 2, 2), 1, "X", "", ""), broken)
    result, decision = call(t, "search_candidates", payee="A\u0000", invoice_number="")
    assert decision is None and "database" in result["error"]


def test_switching_off_the_payer_check_admits_another_payers_invoice():
    t = Toolbox(
        BankLine("L1", date(2026, 2, 2), 50_000, "ACME LTD", "", ""), lambda *a: ROWS, False
    )
    call(t, "search_candidates", payee="ACME", invoice_number="")
    _, decision = call(t, "propose_match", invoice_ids=["INV-4"], rationale="memo says so")
    assert decision is not None  # only the ablation may do this


def test_an_approximate_amount_needs_a_clearly_matching_payer():
    """A shared surname (similarity about 0.35) may not carry a fee-deducted match."""
    namesake = Row("INV-7", "C3", "NOLAN STEVENS", 42_629, date(2026, 1, 20), 0.35)
    line = BankLine("L1", date(2026, 5, 18), 40_572, "STEVENS COOKSLEY", "INVOICE", "")
    t = Toolbox(line, lambda *a: [namesake])
    call(t, "search_candidates", payee="STEVENS", invoice_number="")
    result, decision = call(t, "propose_match", invoice_ids=["INV-7"], rationale="fee deducted")
    assert decision is None and "clearly matches" in result["error"]
