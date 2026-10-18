import pytest
from conftest import FakeClient, completion

from recon import llm

MESSAGES = [{"role": "user", "content": "hi"}]


def test_request_key_ignores_key_order_and_tracks_content():
    a = llm.request_key({"model": "m", "messages": MESSAGES})
    assert a == llm.request_key({"messages": MESSAGES, "model": "m"})
    assert a != llm.request_key({"model": "m", "messages": [{"role": "user", "content": "yo"}]})


def test_record_then_replay_returns_the_same_response_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-value")
    cassette = tmp_path / "c.jsonl"
    fake = FakeClient([completion(content="hello", finish="stop")])
    recorder = llm.LLM(llm.MODELS["deepseek-flash"], mode="record", cassette=cassette, client=fake)
    live = recorder.chat(MESSAGES, tools=[])

    text = cassette.read_text()
    assert "sk-secret-value" not in text and "uthorization" not in text

    player = llm.LLM(llm.MODELS["deepseek-flash"], mode="replay", cassette=cassette)
    replayed = player.chat(MESSAGES, tools=[])
    assert replayed.replayed and not live.replayed
    assert replayed.response.choices[0].message.content == "hello"
    assert replayed.latency_ms == live.latency_ms


def test_replay_miss_is_an_error(tmp_path):
    cassette = tmp_path / "empty.jsonl"
    cassette.write_text("")
    player = llm.LLM(llm.MODELS["deepseek-flash"], mode="replay", cassette=cassette)
    with pytest.raises(llm.ReplayMiss):
        player.chat(MESSAGES, tools=[])


def test_request_carries_the_model_settings():
    fake = FakeClient([completion(content="ok", finish="stop")])
    llm.LLM(llm.MODELS["deepseek-flash"], client=fake).chat(MESSAGES, tools=[])
    (request,) = fake.requests
    assert request["model"] == "deepseek-flash"
    assert request["extra_body"]["thinking"] == {"type": "enabled"}
    assert "temperature" not in request  # ignored in thinking mode, so never sent


def test_cost_uses_cache_hits_when_the_provider_reports_them():
    usage = llm.usage_of(completion())
    assert usage == llm.Usage(input_tokens=1000, cached_tokens=800, output_tokens=100)
    spec = llm.MODELS["deepseek-flash"]
    expected = (200 * spec.usd_in + 800 * spec.usd_cached + 100 * spec.usd_out) / 1e6
    assert llm.cost_usd(spec, usage) == pytest.approx(expected)


def test_local_model_reads_its_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434/v1")
    assert llm.MODELS["ollama:gpt-oss:20b"].url() == "http://10.0.0.5:11434/v1"
    assert llm.cost_usd(llm.MODELS["ollama:gpt-oss:20b"], llm.Usage(10, 0, 10)) == 0


def test_salt_keeps_repeats_apart_and_on_request_sees_every_call(tmp_path):
    cassette = tmp_path / "c.jsonl"
    seen = []
    script = [completion(content="one", finish="stop"), completion(content="two", finish="stop")]
    fake = FakeClient(script)
    for salt in ("", "repeat-1"):
        llm.LLM(llm.MODELS["deepseek-flash"], "record", cassette, fake, salt, seen.append).chat(
            MESSAGES, tools=[]
        )
    assert len(cassette.read_text().splitlines()) == 2 and len(seen) == 2
    for salt, text in (("", "one"), ("repeat-1", "two")):
        reply = llm.LLM(llm.MODELS["deepseek-flash"], "replay", cassette, salt=salt).chat(
            MESSAGES, tools=[]
        )
        assert reply.response.choices[0].message.content == text
