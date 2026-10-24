# Common tasks. Live targets read model keys from .env.
.PHONY: db test baseline eval record-slice demo docker-demo sandbox-image

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
	rm -f tests/cassettes/A-slice.jsonl tests/cassettes/A-slice-sandbox.jsonl
	uv run --env-file .env recon eval --business A --repeats 1 --limit 40 --mode record \
		--cassette tests/cassettes/A-slice.jsonl --out tests/cassettes --name A-slice
	uv run --env-file .env recon eval --business A --repeats 1 --limit 40 --mode record \
		--arm sandbox --statement-size all --cassette tests/cassettes/A-slice-sandbox.jsonl \
		--out tests/cassettes --name A-slice-sandbox

# Offline: replays the committed slice, no key needed.
demo: db
	uv run recon run --business A --limit 40 --mode replay --cassette tests/cassettes/A-slice.jsonl

# The same demo inside the application image.
docker-demo: db
	docker compose --profile app build app
	docker compose --profile app run --rm app run --business A --limit 40 --mode replay \
		--cassette tests/cassettes/A-slice.jsonl --workers 4

# The code-mode sandbox: Python and sandbox/runner.py only (see src/recon/sandbox.py).
sandbox-image:
	docker build -t recon-sandbox:1 sandbox
