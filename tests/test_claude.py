"""Claude's two arms share one Messages loop: tool calls go through the bridge whoever made
them, a programmatic continuation carries only tool results and the container, a paused turn
is sent back as it is, and record and replay hold no key. The client here is scripted with
response shapes from Anthropic's programmatic tool calling documentation."""

import json
from datetime import date

import httpx2
import pytest

pytest.importorskip("anthropic")  # an optional dependency: `uv sync --extra claude`
import anthropic  # noqa: E402

from recon import claude, evaluate, llm  # noqa: E402
from recon.synth import BankLine  # noqa: E402
from recon.tools import Row  # noqa: E402

SPEC = llm.MODELS["claude-haiku-5-5"]
ACME = Row("INV-1001", "C1", "ACME LTD", 50_000, date(2026, 1, 5), 1.0)
LINE = BankLine("A-L0001", date(2026, 2, 2), 50_000, "ACME LTD", "", "IRD 49-091-850")
USAGE = {"input_tokens": 200, "cache_read_input_tokens": 800, "output_tokens": 100}
FROM_CODE = {"type": "code_execution_20260120", "tool_id": "srvtoolu_1"}


def search(line, payee, number):
    return [ACME]


def tool_use(name, args, id, caller=None):
    block = {"type": "tool_use", "id": id, "name": name, "input": {"line_id": LINE.id, **args}}
    block["caller"] = caller or {"type": "direct"}
    return block


def response(content, stop="tool_use", container=None, usage=USAGE):
    body = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content,
        "stop_reason": stop,
        "usage": usage,
    }
    if container:
        body["container"] = {"id": container, "expires_at": "2026-10-25T12:00:00Z"}
    return body


SEARCH = ("search_candidates", {"payee": "ACME LTD"})
PROPOSE = ("propose_match", {"invoice_ids": ["INV-1001"], "rationale": "Exact amount."})
SERVER = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "code_execution", "input": {}}


class FakeAnthropic:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs)))
        if not self.script:
            raise AssertionError("called more times than scripted")
        return self.script.pop(0)


def run(script, programmatic=False):
    fake = FakeAnthropic(script)
    model = claude.ClaudeLLM(SPEC, client=fake)
    results, stats = claude.reconcile_statement([LINE], model, search, programmatic=programmatic)
    return results, stats, fake


def test_the_direct_arm_calls_the_tools_through_the_bridge():
    script = [
        response([tool_use(*SEARCH, "toolu_1")]),
        response([tool_use(*PROPOSE, "toolu_2")]),
    ]
    results, stats, fake = run(script)
    assert results[LINE.id].decision.invoice_ids == ("INV-1001",)
    assert results[LINE.id].decision.source == "claude-haiku-5-5"
    first = fake.requests[0]
    assert "code_execution" not in json.dumps(first["tools"])
    assert all("allowed_callers" not in t and "strict" not in t for t in first["tools"])
    assert first["output_config"] == {"effort": "low"} and "temperature" not in first
    assert "49-091-850" not in json.dumps(first["messages"])
    assert stats.programmatic_calls == 0 and stats.ended == "every line decided"


def test_programmatic_calls_continue_with_only_tool_results_and_the_container():
    script = [
        response([SERVER, tool_use(*SEARCH, "toolu_1", FROM_CODE)], container="container_1"),
        response([tool_use(*PROPOSE, "toolu_2", FROM_CODE)], container="container_1"),
    ]
    results, stats, fake = run(script, programmatic=True)
    first, second = fake.requests
    assert {"type": "code_execution_20260120", "name": "code_execution"} in first["tools"]
    custom = [t for t in first["tools"] if "input_schema" in t]
    assert all(t["allowed_callers"] == ["code_execution_20260120"] for t in custom)
    assert "container" not in first and second["container"] == "container_1"
    last = second["messages"][-1]
    assert last["role"] == "user" and {b["type"] for b in last["content"]} == {"tool_result"}
    assert results[LINE.id].decision.decision == "match"
    assert stats.programmatic_calls == 2 and stats.containers == 1


def test_a_paused_turn_is_sent_back_without_a_new_user_message():
    script = [
        response([SERVER], stop="pause_turn", container="container_1"),
        response(
            [tool_use(*SEARCH, "toolu_1", FROM_CODE), tool_use(*PROPOSE, "toolu_2", FROM_CODE)],
            container="container_1",
        ),
    ]
    _, _, fake = run(script, programmatic=True)
    first, second = fake.requests
    assert len(second["messages"]) == len(first["messages"]) + 1
    assert second["messages"][-1]["role"] == "assistant"


def test_a_direct_call_in_the_programmatic_arm_meets_the_same_bridge():
    """allowed_callers is not a security boundary: a direct proposal of an invoice no search
    returned is refused like any other."""
    script = [
        response([tool_use(*PROPOSE, "toolu_1")]),
        response([{"type": "text", "text": "Done."}], stop="end_turn"),
    ]
    results, stats, fake = run(script, programmatic=True)
    tool = fake.requests[1]["messages"][-1]["content"][0]
    assert "not returned by search_candidates" in tool["content"]
    assert results[LINE.id].decision.decision == "escalate" and stats.programmatic_calls == 0


def test_a_refusal_sends_the_line_to_a_human():
    results, stats, _ = run([response([], stop="refusal")])
    assert results[LINE.id].decision.decision == "escalate"
    assert results[LINE.id].decision.rationale == "claude: model stopped (refusal)"


def test_a_provider_error_keeps_its_prefix_so_the_run_is_invalid():
    request = httpx2.Request("POST", "https://api.anthropic.example/v1/messages")

    class Down(FakeAnthropic):
        def create(self, **kwargs):
            raise anthropic.APIConnectionError(request=request)

    model = claude.ClaudeLLM(SPEC, client=Down([]))
    results, _ = claude.reconcile_statement([LINE], model, search)
    assert results[LINE.id].decision.rationale.startswith("provider error")


def test_the_price_card_follows_the_whole_prompt_length():
    short = claude.Usage(fresh=1_000, cache_read=10_000, cache_write=1_000, output=1_000)
    assert claude.cost_usd(SPEC, short) == pytest.approx(
        (1_000 * 0.10 + 10_000 * 0.01 + 1_000 * 0.125 + 1_000 * 0.50) / 1e6
    )
    long = claude.Usage(fresh=1_000, cache_read=99_500, cache_write=0, output=1_000)
    assert claude.cost_usd(SPEC, long) == pytest.approx(
        5 * (1_000 * 0.10 + 99_500 * 0.01 + 1_000 * 0.50) / 1e6
    )


def test_record_then_replay_gives_the_same_decisions_and_stores_no_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key-0000000000")
    cassette = tmp_path / "claude.jsonl"
    script = [
        response([tool_use(*SEARCH, "toolu_1")]),
        response([tool_use(*PROPOSE, "toolu_2")]),
    ]
    recorder = claude.ClaudeLLM(SPEC, "record", cassette, client=FakeAnthropic(script))
    recorded, _ = claude.reconcile_statement([LINE], recorder, search)
    replayer = claude.ClaudeLLM(SPEC, "replay", cassette)
    replayed, stats = claude.reconcile_statement([LINE], replayer, search)
    assert replayed[LINE.id].decision == recorded[LINE.id].decision
    assert all(row["recon.replayed"] for row in stats.trace if "recon.replayed" in row)
    text = cassette.read_text()
    assert "sk-ant" not in text and "x-api-key" not in text.lower()


def test_arms_match_the_models_that_can_run_them():
    evaluate.check_arm("claude-haiku-5-5", "ptc")
    evaluate.check_arm("claude-haiku-5-5", "direct")
    with pytest.raises(ValueError):
        evaluate.check_arm("deepseek-flash", "ptc")
    with pytest.raises(ValueError):
        evaluate.check_arm("claude-haiku-5-5", "sandbox")
