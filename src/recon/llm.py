"""One OpenAI-compatible client for the chat models, with record and replay.

DeepSeek, Ollama, Gemini and OpenAI serve the OpenAI Chat Completions format, so such a
provider is one registry row: base URL, key variable, model id and prices. Claude rows run
through Anthropic's own Messages API instead (`api="anthropic"`, src/recon/claude.py), the
only way to reach its native tool features such as programmatic tool calling.

Replay keys are the sha256 of the canonical request body. The body never holds headers or
keys, so a cassette cannot leak them.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

# DeepSeek API rules, from api-docs.deepseek.com as read on 2026-10-18:
# - thinking is on by default; with tools, the docs require every later request to carry
#   reasoning_content back (a 400 otherwise). A live check that day accepted a request
#   without it, but assistant messages are appended unchanged either way;
# - temperature has no effect in thinking mode, so it is never sent;
# - strict tool schemas need the /beta base URL.
# Prices: USD per million tokens at peak rates, checked 2026-10-18. Off-peak is half.
PRICES_CHECKED = "2026-10-18"
# Claude Haiku 5.5, from platform.claude.com/docs/en/about-claude/pricing on 2026-10-25:
# $0.10 input, $0.125 5-minute cache write, $0.01 cache read, $0.50 output; a request whose
# prompt (cache reads and writes included) is over 100,000 tokens pays five times each.
CLAUDE_PRICES_CHECKED = "2026-10-25"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    api_model: str
    base_url: str  # "env:NAME" reads the URL from an environment variable
    key_env: str | None
    usd_in: float
    usd_cached: float
    usd_out: float
    extra_body: dict[str, Any] = field(default_factory=dict)
    api: str = "openai"  # or "anthropic": the native Messages API, src/recon/claude.py
    usd_cache_write: float = 0.0
    long_prompt: int = 0  # a prompt over this many tokens pays long_multiplier on every price
    long_multiplier: float = 1.0

    def url(self) -> str:
        if self.base_url.startswith("env:"):
            return os.environ[self.base_url.removeprefix("env:")]
        return self.base_url


_DEEPSEEK_THINKING = {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}

MODELS = {
    m.name: m
    for m in (
        ModelSpec(
            "deepseek-flash",
            "deepseek-flash",
            "https://api.deepseek.com/beta",
            "DEEPSEEK_API_KEY",
            0.30,
            0.006,
            1.20,
            _DEEPSEEK_THINKING,
        ),
        ModelSpec(
            "deepseek-v4-pro",
            "deepseek-v4-pro",
            "https://api.deepseek.com/beta",
            "DEEPSEEK_API_KEY",
            1.32,
            0.044,
            3.96,
            _DEEPSEEK_THINKING,
        ),
        ModelSpec(
            "ollama:gpt-oss:20b",
            "gpt-oss:20b",
            "env:OLLAMA_BASE_URL",
            None,
            0.0,
            0.0,
            0.0,
            {"reasoning_effort": "low"},
        ),
        ModelSpec(
            "claude-haiku-5-5",
            "claude-haiku-5-5",
            "https://api.anthropic.com",
            "ANTHROPIC_API_KEY",
            0.10,
            0.01,
            0.50,
            # adaptive thinking is on by default; effort defaults to medium on this model, so
            # it is set to low explicitly, like DeepSeek's reasoning_effort
            {"output_config": {"effort": "low"}, "cache_control": {"type": "ephemeral"}},
            api="anthropic",
            usd_cache_write=0.125,
            long_prompt=100_000,
            long_multiplier=5.0,
        ),
    )
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cached_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Reply:
    response: ChatCompletion
    latency_ms: float
    replayed: bool


class ReplayMiss(LookupError):
    """A replayed run asked for a request that was never recorded."""


def request_key(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def usage_of(response: ChatCompletion) -> Usage:
    u = response.usage
    if u is None:
        return Usage(0, 0, 0)
    cached = getattr(u, "prompt_cache_hit_tokens", None)  # DeepSeek
    if cached is None and u.prompt_tokens_details is not None:
        cached = u.prompt_tokens_details.cached_tokens  # OpenAI style
    return Usage(u.prompt_tokens, cached or 0, u.completion_tokens)


def cost_usd(spec: ModelSpec, usage: Usage) -> float:
    fresh = usage.input_tokens - usage.cached_tokens
    return (
        fresh * spec.usd_in
        + usage.cached_tokens * spec.usd_cached
        + usage.output_tokens * spec.usd_out
    ) / 1e6


class LLM:
    def __init__(
        self,
        spec: ModelSpec,
        mode: str = "live",
        cassette: Path | None = None,
        client: Any = None,
        salt: str = "",
        on_request: Callable[[list[dict]], None] | None = None,
    ) -> None:
        """salt separates repeated runs of the same requests in one cassette; on_request sees
        every outgoing message list, in replay too, so leak scans cover recorded runs."""
        if mode not in ("live", "record", "replay"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode != "live" and cassette is None:
            raise ValueError(f"{mode} mode needs a cassette path")
        self.spec, self.mode, self.cassette = spec, mode, cassette
        self.recorded: dict[str, dict] = {}
        if cassette is not None and cassette.exists():
            for line in cassette.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.recorded[entry["key"]] = entry
        self._client = client
        self.salt = salt
        self.on_request = on_request
        self._lock = threading.Lock()

    @property
    def client(self) -> Any:
        if self._client is None:
            key = os.environ.get(self.spec.key_env, "") if self.spec.key_env else "unused"
            if not key:
                raise RuntimeError(
                    f"{self.spec.key_env} is not set: add it to .env and run with"
                    " `uv run --env-file .env ...`"
                )
            self._client = OpenAI(base_url=self.spec.url(), api_key=key, timeout=120)
        return self._client

    def chat(
        self, messages: list[dict], tools: list[dict], response_format: dict | None = None
    ) -> Reply:
        body = {
            "model": self.spec.api_model,
            "messages": messages,
            "tools": tools,
            "extra_body": self.spec.extra_body,
        }
        if response_format is not None:  # added only when used, so older recordings still match
            body["response_format"] = response_format
        if self.on_request is not None:
            self.on_request(messages)
        key = request_key({**body, "salt": self.salt} if self.salt else body)
        if self.mode == "replay":
            entry = self.recorded.get(key)
            if entry is None:
                raise ReplayMiss(f"no recorded response for request {key[:12]}")
            response = ChatCompletion.model_validate(entry["response"])
            return Reply(response, entry["latency_ms"], replayed=True)

        start = time.perf_counter()
        kwargs = {k: v for k, v in body.items() if v or k != "tools"}  # omit an empty tool list
        response = self.client.chat.completions.create(**kwargs)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        if self.mode == "record":
            entry = {
                "key": key,
                "latency_ms": latency_ms,
                "response": response.model_dump(mode="json"),
            }
            with self._lock:
                self.cassette.parent.mkdir(parents=True, exist_ok=True)
                with self.cassette.open("a") as f:
                    f.write(json.dumps(entry, sort_keys=True) + "\n")
        return Reply(response, latency_ms, replayed=False)
