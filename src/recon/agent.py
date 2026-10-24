"""A hand-rolled tool-calling loop: one isolated conversation per bank line.

The loop ends only when the model calls a terminal tool (propose_match or escalate) and
the tool accepts. A text answer, a truncated reply, a provider error, the step limit or the
budget limit all end in an escalation, so nothing is matched by accident.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import openai

from recon.llm import LLM, cost_usd, usage_of
from recon.redact import redact, untrusted_block
from recon.score import Decision
from recon.synth import BankLine
from recon.tools import TOOL_SCHEMAS, Search, Toolbox, dollars

PROMPT = files("recon").joinpath("prompts", "agent.md").read_text()
_GUIDANCE_START = "Payee, reference and memo are written"
_PII_RULE = " Never include email addresses, phone numbers, bank account numbers or IRD numbers."


def unguarded(prompt: str) -> str:
    """The prompt without its untrusted-text paragraph and personal-data rule, for the
    defences-off ablation only. Any prompt that carries both can be unguarded."""
    guidance = prompt[prompt.index(_GUIDANCE_START) :].split("\n\n")[0]
    return prompt.replace(guidance + "\n\n", "").replace(_PII_RULE, "")


UNGUARDED_PROMPT = unguarded(PROMPT)
TERMINAL_TOOLS = {"propose_match", "escalate"}
# Worth escalating one line for. Anything else (a bad key, no balance, a malformed request)
# would repeat on every line, so it stops the run instead of filling the queue.
TRANSIENT_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


PROMPT_SHA = sha(PROMPT)


@dataclass(frozen=True)
class Defences:
    """Injection defences, each of which an ablation can switch off."""

    fence: bool = True  # payer text is fenced and the prompt says it is data, never instructions
    redact: bool = True  # personal data is removed before the model and from rationales
    payer_check: bool = True  # the verifier refuses an invoice of another payer
    screen: bool = True  # the policy never auto-approves instruction-like payer text

    def prompt(self, base: str = PROMPT) -> str:
        return base if self.fence else unguarded(base)


DEFENCES = {
    "on": Defences(),
    "prompt_only": Defences(fence=True, redact=False, payer_check=False, screen=False),
    "code_only": Defences(fence=False, redact=True, payer_check=True, screen=True),
    "off": Defences(fence=False, redact=False, payer_check=False, screen=False),
}


@dataclass(frozen=True)
class Limits:
    steps: int = 6
    max_tokens: int = 60_000  # input plus output, summed over the line's calls
    max_usd: float = 0.05


@dataclass
class LineResult:
    decision: Decision
    steps: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)


def user_message(line: BankLine, defences: Defences = DEFENCES["on"]) -> str:
    direction = "money in" if line.amount_cents > 0 else "money out"
    fields = {"payee": line.payee, "reference": line.reference, "memo": line.memo}
    if defences.fence:
        payer_text = untrusted_block(fields, defences.redact)
    else:
        clean = redact if defences.redact else str
        payer_text = "\n".join(f"{name}: {clean(value)}" for name, value in fields.items())
    return (
        f"Bank line {line.id}\n"
        f"Posted: {line.posted.isoformat()}\n"
        f"Amount: NZ${dollars(line.amount_cents)} ({direction})\n\n"
        f"{payer_text}"
    )


def finalize(decision: Decision, source: str, defences: Defences) -> Decision:
    """The decision as it goes to the policy and the queue: labelled with the model that made
    it and, as defence in depth, with no personal data in its rationale."""
    return Decision(
        decision.line_id,
        decision.decision,
        decision.invoice_ids,
        decision.rule,
        source=source,
        rationale=redact(decision.rationale) if defences.redact else decision.rationale,
    )


def assistant_message(message: Any) -> dict[str, Any]:
    """The assistant turn exactly as returned, reasoning_content included."""
    turn = message.model_dump(exclude_none=True)
    turn["content"] = message.content
    return turn


def reconcile(
    line: BankLine,
    model: LLM,
    search: Search,
    limits: Limits,
    defences: Defences = DEFENCES["on"],
) -> LineResult:
    toolbox = Toolbox(line, search, defences.payer_check)
    result = LineResult(decision=Decision(line.id, "escalate"))
    prompt = defences.prompt()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message(line, defences)},
    ]
    provider = model.spec.name.split(":")[0].split("-")[0]

    def finish(decision: Decision) -> LineResult:
        result.decision = finalize(decision, model.spec.name, defences)
        return result

    def escalate(reason: str) -> LineResult:
        return finish(Decision(line.id, "escalate", rationale=reason))

    for step in range(1, limits.steps + 1):
        try:
            reply = model.chat(messages, TOOL_SCHEMAS)
        except TRANSIENT_ERRORS as exc:
            return escalate(f"provider error: {type(exc).__name__}")
        response = reply.response
        if not response.choices:
            return escalate("provider returned no choices")
        usage = usage_of(response)
        cost = cost_usd(model.spec, usage)
        choice = response.choices[0]
        result.steps = step
        result.input_tokens += usage.input_tokens
        result.cached_tokens += usage.cached_tokens
        result.output_tokens += usage.output_tokens
        result.cost_usd += cost
        result.latency_ms += reply.latency_ms
        result.trace.append(
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": provider,
                "gen_ai.request.model": model.spec.api_model,
                "gen_ai.response.model": response.model,
                "gen_ai.response.finish_reasons": [choice.finish_reason],
                "gen_ai.usage.input_tokens": usage.input_tokens,
                "gen_ai.usage.output_tokens": usage.output_tokens,
                "recon.cached_tokens": usage.cached_tokens,
                "recon.cost_usd": round(cost, 6),
                "recon.latency_ms": reply.latency_ms,
                "recon.replayed": reply.replayed,
                "recon.prompt_sha": sha(prompt),
                "recon.line_id": line.id,
                "recon.step": step,
            }
        )
        messages.append(assistant_message(choice.message))
        calls = choice.message.tool_calls or []
        if choice.finish_reason != "tool_calls" or not calls:
            return escalate(f"model stopped ({choice.finish_reason}) without a decision")
        if sum(call.function.name in TERMINAL_TOOLS for call in calls) > 1:
            return escalate("model gave more than one final decision in one turn")

        for call in calls:
            outcome, decision = toolbox.call(call.function.name, call.function.arguments)
            result.trace.append(
                {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": call.function.name,
                    "gen_ai.tool.call.id": call.id,
                    "recon.ok": "error" not in outcome,
                    "recon.line_id": line.id,
                    "recon.step": step,
                }
            )
            content = json.dumps(outcome, sort_keys=True)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            if decision is not None:
                return finish(decision)

        spent_tokens = result.input_tokens + result.output_tokens
        if spent_tokens > limits.max_tokens or result.cost_usd > limits.max_usd:
            return escalate("budget limit reached")
    return escalate("step limit reached")
