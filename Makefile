# Common tasks. Live targets read model keys from .env.
.PHONY: db test baseline eval record-slice demo

db:
	docker compose up -d --wait

test: db
	uv run pytest

baseline: db
	uv run recon baseline --business A
	uv run recon baseline --business B

# Live, three repeats per business; writes results/<date>-<business>-<model>.json.
eval: db
	uv run --env-file .env recon eval --business A --repeats 3 --mode record
	uv run --env-file .env recon eval --business B --repeats 3 --mode record

# Re-record the 40-line slice that CI replays. Needed after any change to prompts, tools or data.
record-slice: db
	rm -f tests/cassettes/A-slice.jsonl
	uv run --env-file .env recon eval --business A --repeats 1 --limit 40 --mode record \
		--cassette tests/cassettes/A-slice.jsonl --out tests/cassettes --name A-slice

# Offline: replays the committed slice, no key needed.
demo: db
	uv run recon run --business A --limit 40 --mode replay --cassette tests/cassettes/A-slice.jsonl
