"""The bridge is the only path from model-written code to the tools: it refuses anything
outside the four tools, their schemas, the statement's open lines and each line's budget, and
only an accepted propose_match or escalate becomes a decision."""

import json
from datetime import date

import pytest

from recon import agent, bridge
from recon.synth import BankLine
from recon.tools import Row

ACME = Row("INV-1001", "C1", "ACME LTD", 50_000, date(2026, 1, 5), 1.0)
OTHER = Row("INV-2002", "C2", "BETA CO", 50_000, date(2026, 1, 6), 0.1)
LINE_A = BankLine("A-L0001", date(2026, 2, 2), 50_000, "ACME LTD", "", "Paid, IRD 49-091-850")
LINE_B = BankLine("A-L0002", date(2026, 2, 3), 50_000, "BETA CO", "", "")


def search_log():
    calls = []

    def search(line, payee, number):
        calls.append(line.id)
        return [ACME] if line.id == LINE_A.id else [OTHER]

    return search, calls


@pytest.fixture
def statement():
    search, calls = search_log()
    lines = [LINE_A, LINE_B]
    with bridge.Bridge(lines, search, agent.DEFENCES["on"], "test") as b:
        yield b, calls


def call(b, name, **args):
    return json.loads(b.handle(name, args))


def search_a(b):
    return call(b, "search_candidates", line_id=LINE_A.id, payee="ACME LTD", invoice_number="")


def propose(b, line, ids, rationale="Exact amount, same payer."):
    return call(b, "propose_match", line_id=line.id, invoice_ids=ids, rationale=rationale)


def test_the_api_is_the_mcp_read_tools_plus_the_two_decisions(statement):
    b, _ = statement
    api = {spec["name"]: spec for spec in b.api()}
    assert list(api) == ["search_candidates", "find_combinations", "propose_match", "escalate"]
    assert api["search_candidates"]["description"].startswith("Open invoices for a bank line")
    for spec in api.values():
        schema = spec["input_schema"]
        assert "line_id" in schema["required"] and schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "name", ["run_sql", "queue_proposal", "get_bank_line", "list_bank_lines", "__import__"]
)
def test_anything_but_the_four_tools_is_refused_without_touching_the_ledger(statement, name):
    b, calls = statement
    assert "unknown tool" in call(b, name, line_id=LINE_A.id)["error"]
    assert calls == [] and b.trace[-1]["recon.refusal"] == "unknown_method"


@pytest.mark.parametrize(
    "args",
    [
        {"line_id": LINE_A.id, "payee": 123, "invoice_number": ""},
        {"line_id": LINE_A.id, "payee": "ACME LTD", "invoice_number": "", "sql": "DROP"},
        {"line_id": LINE_A.id},
        {"line_id": LINE_A.id, "payee": "x" * 141},
    ],
)
def test_arguments_outside_the_schema_are_refused_not_raised(statement, args):
    b, calls = statement
    assert "error" in json.loads(b.handle("search_candidates", args))
    assert calls == [] and b.trace[-1]["recon.refusal"] in ("schema", "size")


def test_a_string_where_a_list_belongs_is_refused_not_iterated(statement):
    b, _ = statement
    search_a(b)
    result = call(b, "find_combinations", line_id=LINE_A.id, invoice_ids="INV-1001")
    assert "list of strings" in result["error"]


def test_arguments_that_are_not_an_object_are_refused(statement):
    b, _ = statement
    assert "object" in json.loads(b.handle("search_candidates", ["A-L0001"]))["error"]


@pytest.mark.parametrize("line_id", ["A-L9999", "", 7, None])
def test_only_open_lines_of_this_statement_can_be_named(statement, line_id):
    b, calls = statement
    result = json.loads(b.handle("search_candidates", {"line_id": line_id, "payee": "ACME"}))
    assert "not an open line" in result["error"] and calls == []


def test_a_missing_line_id_is_named_as_missing(statement):
    """A live Claude run left line_id out of escalate twice; 'not an open line' did not tell
    it what was wrong."""
    b, _ = statement
    result = json.loads(b.handle("escalate", {"reason": "No invoice fits."}))
    assert "missing arguments ['line_id']" in result["error"]
    assert b.trace[-1]["recon.refusal"] == "schema"


def test_an_invoice_found_on_one_line_cannot_be_proposed_on_another(statement):
    b, _ = statement
    search_a(b)
    assert "not returned by search_candidates" in propose(b, LINE_B, ["INV-1001"])["error"]
    assert LINE_B.id not in b.decisions


def test_the_first_accepted_decision_is_final(statement):
    b, _ = statement
    search_a(b)
    assert propose(b, LINE_A, ["INV-1001"]) == {"status": "proposed"}
    assert "already decided" in propose(b, LINE_A, ["INV-1001"])["error"]
    assert "already decided" in search_a(b)["error"]
    escalated = call(b, "escalate", line_id=LINE_A.id, reason="changed my mind")
    assert "already decided" in escalated["error"]
    assert b.results()[LINE_A.id].decision == "match"


def test_two_refused_proposals_send_the_line_to_a_human(statement):
    b, _ = statement
    call(b, "search_candidates", line_id=LINE_B.id, payee="ACME LTD", invoice_number="")
    search_a(b)
    assert "error" in propose(b, LINE_B, ["INV-1001"])
    assert "error" in propose(b, LINE_B, ["INV-2002", "INV-2002"])
    decided = b.results()[LINE_B.id]
    assert decided.decision == "escalate" and "proposal limit" in decided.rationale
    assert "already decided" in propose(b, LINE_B, ["INV-2002"])["error"]


def test_a_line_has_a_fixed_budget_of_tool_calls(statement):
    b, calls = statement
    for _ in range(bridge.CAPS.calls_per_line):
        assert "candidates" in search_a(b)
    assert "used its tool calls" in search_a(b)["error"]
    assert len(calls) == bridge.CAPS.calls_per_line
    assert b.trace[-1]["recon.refusal"] == "call_cap"


def test_an_undecided_line_goes_to_a_human(statement):
    b, _ = statement
    results = b.results()
    assert {d.decision for d in results.values()} == {"escalate"}
    assert all(
        d.rationale == "code mode: no decision" and d.source == "test" for d in results.values()
    )


def test_a_rationale_is_redacted_before_the_queue(statement):
    b, _ = statement
    search_a(b)
    propose(b, LINE_A, ["INV-1001"], rationale="Memo says IRD 49-091-850; exact amount.")
    assert "49-091-850" not in b.results()[LINE_A.id].rationale


def test_every_call_is_traced_with_its_caller(statement):
    b, _ = statement
    b.handle("search_candidates", {"line_id": LINE_A.id, "payee": "ACME"}, caller="direct")
    row = b.trace[-1]
    assert row["recon.caller"] == "direct" and row["recon.ok"] and row["recon.refusal"] is None
    assert row["gen_ai.tool.name"] == "search_candidates" and row["recon.line_id"] == LINE_A.id
