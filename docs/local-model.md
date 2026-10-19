# Running the local model row

How to measure `ollama:gpt-oss:20b` on a machine with its own GPU. Everything runs on that one machine: the model, Postgres and the evaluation. Nothing is exposed to the network. The output is results files on a branch; the README is regenerated elsewhere.

Written for Windows 11 with an NVIDIA GPU. The published runs used an 8 GB RTX 3070, which holds only part of `gpt-oss:20b`, so Ollama ran the rest on the CPU; a GPU with 16 GB or more holds the whole model. On Linux the commands are the same apart from the shell.

## 1. Tools

Install what is missing: Git, GitHub CLI (`gh auth login` once), [uv](https://docs.astral.sh/uv/), Docker Desktop with the WSL 2 backend, and [Ollama](https://ollama.com/download).

## 2. Ollama

Give the model enough context for a whole conversation, and let it serve two lines at once:

```powershell
setx OLLAMA_CONTEXT_LENGTH 16384
setx OLLAMA_NUM_PARALLEL 2
```

Quit Ollama from the tray and start it again, so it reads the new values. Then:

```powershell
ollama pull gpt-oss:20b
ollama run gpt-oss:20b "Say ready."
ollama ps
```

`ollama ps` shows how the model is split between GPU and CPU. If part of it runs on the CPU, the times are not comparable with a full-GPU run (the decisions should not change, but that is untested): write the split down and report it with the results. If it shows `100% CPU`, the GPU is not in use at all: fix that first.

## 3. Repository

```powershell
gh repo clone NecmiKural/bank-reconciliation-agent
cd bank-reconciliation-agent
git config user.name "Necmi Kural"
git config user.email "68390369+NecmiKural@users.noreply.github.com"
git switch -c results/ollama-gpt-oss-20b
Set-Content -Path .env -Value "OLLAMA_BASE_URL=http://localhost:11434/v1"
docker compose up -d --wait
uv sync
uv run pytest
```

Every test must pass. If `test_ground_truth_fingerprint` fails, stop: the synthetic data differs on this platform, so results from here would not be comparable. Report the two hashes it prints.

## 4. Smoke test

```powershell
uv run --env-file .env recon run --business A --model ollama:gpt-oss:20b --limit 5 --workers 2
```

Each line prints `ok` or `BAD`, and the summary at the end includes `provider_errors` and `latency_ms_per_line`. A provider error here usually means the model server is not reachable or a call timed out. A `BAD` line is a measurement, not a failure of the setup.

Estimate the full run: lines left to the agent (92 for A, 102 for B) times the p50 seconds per line, divided by 2 workers, times the number of repeats. Use 3 repeats if both businesses fit in about 6 hours; otherwise use 1 and say so.

## 5. Full run

Do not edit any tracked file until both runs finish: each results file records the commit, and an edit under `src/`, `pyproject.toml` or `uv.lock` marks it dirty.

```powershell
uv run --env-file .env recon eval --business A --model ollama:gpt-oss:20b --repeats 3 --mode record --workers 2
uv run --env-file .env recon eval --business B --model ollama:gpt-oss:20b --repeats 3 --mode record --workers 2
```

`recon eval` refuses to save a run with provider errors. If that happens, rerun that business with `--workers 1`.

Check both files:

```powershell
uv run python -c "import json,glob; [print(f, (r:=json.load(open(f)))['valid'], r['commit'], r['aggregate']['totals']['provider_errors']) for f in glob.glob('results/*ollama*')]"
```

Each must say `True`, a commit without `-dirty`, and `0`.

## 6. Rule compiler (optional, short)

The plain-English rule compiler has its own 40-sentence eval. With the model still loaded:

```powershell
uv run --env-file .env recon rules-eval --model ollama:gpt-oss:20b --repeats 3 --mode record
```

It writes `results/<date>-rules-ollama_gpt-oss_20b-record.json`. Commit it with the other two.

## 7. Hand back

Commit only the results files and push the branch. Do not push to `main`, and do not run `recon report` or edit `README.md`; the main checkout regenerates the README after merging.

```powershell
git add results/*ollama_gpt-oss_20b*
git commit -m "chore: record live results for the local gpt-oss:20b"
git push -u origin results/ollama-gpt-oss-20b
```

Report: the branch, the two file names, and for each business the agent accuracy, wrong matches, redirect and over-allocation lines that succeeded, model time p50 and p95, and repeats used; also the GPU, its memory, the driver and the Ollama version:

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
ollama --version
```
