# Bank Reconciliation Agent

A rules-first, tool-calling agent that matches New Zealand small-business bank lines to invoices on a synthetic ledger. Deterministic rules close the easy lines. For the rest, the agent searches and proposes matches through four narrow tools, code verifies every proposal, and anything outside a small policy envelope goes to a human. **The model never does arithmetic and never posts to the books.**

> **Status: under active development.** Results appear here once the eval harness lands. Every number in this README will be generated from `results/*.json` by one command; none is typed by hand.

## What it measures

- Match precision, recall and F1 per case type (exact, typo, split, partial, fee deducted, duplicate amount, no match), with Wilson 95% intervals, against a rules-only baseline on a held-out synthetic business.
- Wrong auto-approvals, reported as "0 of N, 95% upper bound X%" rather than as a guarantee.
- Prompt-injection attack success through payer-written bank memos, with defences on and off.
- PII leaks in outbound model requests (gate: zero).
- Cost per 100 lines, cache share, and p50/p95 latency.

## Design choices

- **Rules first.** An LLM only sees what deterministic matching could not close.
- **Least privilege, twice.** There is no posting tool, and the agent's Postgres role can only read the ledger and insert proposals.
- **Provider-agnostic.** One OpenAI-compatible client: hosted DeepSeek models, a local `gpt-oss:20b` through Ollama, and Gemini for embeddings. Adding Anthropic, OpenAI or another Gemini model is one registry row.

> **Synthetic data only.** Every business, contact, invoice and bank line is generated from a seed. No real customer, accounting-platform, tax-authority or employer data is used, and only synthetic data is ever sent to a model provider.

## Run

Needs [uv](https://docs.astral.sh/uv/) and Docker.

```bash
docker compose up -d --wait            # Postgres 17 with pgvector and pg_trgm on localhost:5433
uv sync
uv run recon synth --out data          # write the two synthetic businesses as JSON
uv run recon baseline --business A     # load business A and score the rules pass alone
uv run --env-file .env recon run --business A --limit 12   # rules, then the agent on 12 open lines
uv run pytest                          # includes least-privilege tests against the database
```

Copy `.env.example` to `.env` for the model keys; `uv run --env-file .env ...` reads it.

## How this was built

Pair-programmed with Claude Code. I reviewed every line, and the rules the coding agent works under are committed in [AGENTS.md](AGENTS.md).

## Disclaimer

A portfolio project, not accounting advice.

## Tech

Python 3.13 · uv · PostgreSQL 17 · pg_trgm · pgvector · psycopg 3 · Docker Compose · pytest · ruff · GitHub Actions

---
*Personal portfolio project, MIT licensed.*
