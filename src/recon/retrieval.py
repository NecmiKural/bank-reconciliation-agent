"""Candidate-search ablation: does vector or hybrid search find the right invoices better
than trigram matching on the payer name?

No language model is involved, only embeddings. For every bank line that has a matching
invoice, all open invoices issued by the posting date are ranked by each method and the
ranks of the labelled invoices are scored. Two query sets are used: the payee as the bank
wrote it, and a stress set of renamed payees ("&" for "AND", initials, word order, legal
suffixes) that the generator never produces, so the second set is a test of name matching
alone. The reference field is ignored here to isolate the name.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openai
import psycopg
from openai import OpenAI

from recon import db, synth

METHODS = ("trigram", "fts", "vector", "rrf")
RRF_K = 60
KS = (5, 10, 25)

Embedder = Callable[[list[str]], dict[str, list[float]]]


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    base_url: str
    key_env: str
    dims: int
    batch: int  # texts per request; the free tier counts each text as one request
    per_minute: int


GEMINI = EmbeddingSpec(
    "gemini-embedding-2",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
    "GEMINI_API_KEY",
    768,
    90,
    100,
)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS ledger.contact_embedding (
    contact_id text PRIMARY KEY REFERENCES ledger.contact,
    embedding  vector(%(dims)s) NOT NULL
);
"""

SCORES_SQL = """
SELECT i.id, i.amount_cents, i.issued,
       similarity(c.payee, %(q)s) AS trigram,
       ts_rank(to_tsvector('simple', c.payee), plainto_tsquery('simple', %(q)s)) AS fts,
       1 - (e.embedding <=> %(qv)s::vector) AS vector
FROM ledger.invoice i
JOIN ledger.contact c ON c.id = i.contact_id
JOIN ledger.contact_embedding e ON e.contact_id = c.id
WHERE i.issued <= %(posted)s
"""


def gemini_embedder(spec: EmbeddingSpec, cache_path: Path) -> Embedder:
    """Embeddings through the OpenAI-compatible endpoint, cached on disk by text."""
    cache: dict[str, list[float]] = (
        json.loads(cache_path.read_text()) if cache_path.exists() else {}
    )
    client = OpenAI(base_url=spec.base_url, api_key=os.environ.get(spec.key_env, ""), timeout=60)

    def embed(texts: list[str]) -> dict[str, list[float]]:
        todo = sorted({t for t in texts if t not in cache})
        for start in range(0, len(todo), spec.batch):
            batch = todo[start : start + spec.batch]
            while True:
                try:
                    response = client.embeddings.create(
                        model=spec.name, input=batch, dimensions=spec.dims
                    )
                    break
                except openai.RateLimitError:
                    time.sleep(62)  # free tier: one minute window
            cache.update({t: d.embedding for t, d in zip(batch, response.data, strict=True)})
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache))
            if start + spec.batch < len(todo):
                time.sleep(60 * len(batch) / spec.per_minute + 2)
        return {t: cache[t] for t in texts}

    return embed


_SUFFIXES = re.compile(r"\s+\b(LTD|LIMITED|INC|PLC|LLC|GROUP)\b$")


def rename(payee: str, seed: str) -> str:
    """A plausible other way to write the same payer, chosen deterministically per line."""
    rng = random.Random(hashlib.sha256(seed.encode()).hexdigest())
    words = payee.split()
    options = []
    if " AND " in f" {payee} ":
        options.append(re.sub(r"\bAND\b", "&", payee))
    if _SUFFIXES.search(payee):
        options.append(_SUFFIXES.sub("", payee))
    elif len(words) >= 2:
        options.append(f"{payee} LIMITED")
    if len(words) == 2:
        options.append(f"{words[1]} {words[0]}")
        options.append(f"{words[0][0]} {words[1]}")
    options.append(payee.title().replace(" ", ", ", 1) if len(words) >= 2 else payee.lower())
    return rng.choice([o for o in options if o != payee] or [payee.lower()])


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def load_embeddings(conn: psycopg.Connection, dataset: synth.Dataset, embed: Embedder, dims: int):
    vectors = embed([c.payee for c in dataset.contacts])
    with conn.transaction():
        conn.execute(SCHEMA.replace("%(dims)s", str(int(dims))))
        conn.execute("TRUNCATE ledger.contact_embedding")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO ledger.contact_embedding VALUES (%s, %s::vector)",
                [(c.id, vector_literal(vectors[c.payee])) for c in dataset.contacts],
            )


def ranks(rows: list[tuple], amount: int) -> dict[str, list[str]]:
    """Invoice ids per method, best first: exact amount, then the method's score, then the
    most recent invoice, as the agent's search orders them."""
    out = {}
    for column, method in ((3, "trigram"), (4, "fts"), (5, "vector")):
        ordered = sorted(rows, key=lambda r, c=column: (r[1] == amount, r[c], r[2], r[0]))
        out[method] = [r[0] for r in reversed(ordered)]
    position = {m: {inv: i for i, inv in enumerate(ids)} for m, ids in out.items()}
    fused = {
        inv: sum(1 / (RRF_K + position[m][inv] + 1) for m in ("trigram", "vector"))
        for inv in out["trigram"]
    }
    out["rrf"] = sorted(fused, key=lambda inv: (-fused[inv], inv))
    return out


def score_ranks(ranked: list[str], truth: set[str]) -> tuple[int, int]:
    """1-based rank of the last and of the first labelled invoice."""
    places = [ranked.index(t) + 1 for t in truth]
    return max(places), min(places)


def metrics(places: list[tuple[int, int]]) -> dict[str, Any]:
    n = len(places)
    out: dict[str, Any] = {"n": n}
    for k in KS:
        out[f"recall_at_{k}"] = round(sum(last <= k for last, _ in places) / n, 4)
    out["mrr"] = round(sum(1 / first for _, first in places) / n, 4)
    return out


def ablate(
    conn: psycopg.Connection, dataset: synth.Dataset, embed: Embedder, dims: int
) -> dict[str, Any]:
    db.reset(conn, dataset)
    load_embeddings(conn, dataset, embed, dims)
    labels = {label.line_id: label for label in dataset.labels}
    lines = [line for line in dataset.lines if labels[line.id].expected == "match"]
    queries = {
        "as_written": {line.id: line.payee for line in lines},
        "renamed": {line.id: rename(line.payee, line.id) for line in lines},
    }
    vectors = embed(sorted({q for qs in queries.values() for q in qs.values()}))
    report: dict[str, Any] = {}
    for name, by_line in queries.items():
        places: dict[str, list[tuple[int, int]]] = {m: [] for m in METHODS}
        for line in lines:
            q = by_line[line.id]
            params = {"q": q, "qv": vector_literal(vectors[q]), "posted": line.posted}
            rows = conn.execute(SCORES_SQL, params).fetchall()
            for method, ranked in ranks(rows, line.amount_cents).items():
                places[method].append(score_ranks(ranked, set(labels[line.id].invoice_ids)))
        report[name] = {m: metrics(p) for m, p in places.items()}
    report["renamed_examples"] = [
        [queries["as_written"][line.id], queries["renamed"][line.id]] for line in lines[:6]
    ]
    return report


def run(spec: EmbeddingSpec, cache_path: Path, database_url: str | None = None) -> dict:
    from recon.evaluate import fingerprint, git_commit

    embed = gemini_embedder(spec, cache_path)
    out: dict[str, Any] = {
        "retrieval": True,
        "embedding_model": spec.name,
        "dims": spec.dims,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "businesses": {},
    }
    with db.connect(database_url) as conn:
        for dataset_spec in synth.SPECS:
            dataset = synth.generate(dataset_spec)
            out["businesses"][dataset_spec.name] = {
                "dataset_fingerprint": fingerprint(dataset),
                **ablate(conn, dataset, embed, spec.dims),
            }
    return out
