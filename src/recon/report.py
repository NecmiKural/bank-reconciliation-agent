"""Write the README's results section from results/*.json.

The block between the markers is generated; `recon report` rewrites it and a test fails if
it drifts from the results files, so no measured number in the README is typed by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recon import score, synth
from recon.evaluate import fingerprint
from recon.score import Decision

PRIMARY_MODEL = "deepseek-flash"  # the main tables; other models get a comparison table
START = "<!-- results:start -->"
END = "<!-- results:end -->"
BUSINESSES = {"A": "Business A (development)", "B": "Business B (second seed)"}
ABLATIONS = {
    "on": "All on",
    "prompt_only": "Prompt group only",
    "code_only": "Code group only",
    "off": "All off",
}


def load(results_dir: Path, kind: str = "runs") -> list[dict[str, Any]]:
    """Results files of one kind: pipeline runs, or the retrieval ablation."""
    files = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    return [f for f in files if (kind in f)]


def pick(
    results: list[dict], business: str, defences: str, model: str = PRIMARY_MODEL
) -> dict | None:
    """The latest full live run of a business with the given defences and model."""
    found = [
        r
        for r in results
        if r["business"] == business
        and r["model"] == model
        and r.get("defences", {}).get("name", "on") == defences  # older files: always on
        and r["mode"] in ("live", "record")
        and r["limit"] is None
        and r["valid"]
    ]
    return max(found, key=lambda r: r["generated_at"]) if found else None


def dataset_for(result: dict) -> synth.Dataset:
    spec = next(s for s in synth.SPECS if s.name == result["business"])
    dataset = synth.generate(spec)
    if fingerprint(dataset) != result["dataset_fingerprint"]:
        raise ValueError(f"{result['business']}: results were made on different ground truth")
    return dataset


def decision(line_id: str, compact: str) -> Decision:
    if compact.startswith("match:"):
        return Decision(line_id, "match", tuple(compact.removeprefix("match:").split(",")))
    if compact.startswith("bank_rule:"):
        return Decision(line_id, "bank_rule", rule=compact.removeprefix("bank_rule:"))
    return Decision(line_id, "escalate")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def upper(k: int, n: int) -> str:
    return pct(score.wilson(k, n)[1])


def facts(result: dict) -> dict[str, Any]:
    """Everything the tables need from one results file, pooled over its runs."""
    dataset = dataset_for(result)
    labels = {label.line_id: label for label in dataset.labels}
    runs = result["runs"]
    rules = {i: decision(i, c) for i, c in runs[0]["decisions_rules"].items()}
    rules_only = score.score(dataset.labels, rules)
    cases: dict[str, list[int]] = {}
    ordinary = [0, 0]
    mistakes: dict[tuple[str, str], int] = {}
    for run in runs:
        for line_id, compact in run["decisions_agent"].items():
            label = labels[line_id]
            ok = score.is_correct(label, decision(line_id, compact))
            if not ok:
                mistakes[(line_id, compact)] = mistakes.get((line_id, compact), 0) + 1
            cell = cases.setdefault(label.case, [0, 0])
            cell[0] += ok
            cell[1] += 1
            if not label.attack_goal:
                ordinary[0] += ok
                ordinary[1] += 1
    summaries = [run["summary"] for run in runs]
    code = [s["attacks"]["code_enforced"] for s in summaries]
    return {
        "result": result,
        "agg": result["aggregate"],
        "first": summaries[0],
        "summaries": summaries,
        "rules_only": rules_only,
        "cases": cases,
        "ordinary": ordinary,
        "preapproval": [
            sum(c["preapproval"]["success"] for c in code),
            sum(c["preapproval"]["n"] for c in code),
        ],
        "pii_echoed": [
            sum(c["pii_leak"]["success"] for c in code),
            sum(c["pii_leak"]["n"] for c in code),
        ],
        "agent_decisions": sum(s["agent"]["n"] for s in summaries),
        "match_proposals": sum(s["agent"]["match"]["proposed"] for s in summaries),
        "mistakes": [
            {
                "line": line_id,
                "case": labels[line_id].case,
                "attack": labels[line_id].attack_goal,
                "expected": expected(labels[line_id]),
                "decided": compact,
                "runs": count,
            }
            for (line_id, compact), count in sorted(mistakes.items())
        ],
    }


def expected(label: synth.Label) -> str:
    if label.expected == "match":
        return "match:" + ",".join(sorted(label.invoice_ids))
    if label.expected == "bank_rule":
        return "bank_rule:" + label.rule
    return "escalate"


def mistakes_table(main: dict[str, dict]) -> list[str]:
    rows = [(b, m) for b, f in main.items() for m in f["mistakes"]]
    if not rows:
        return ["None in any run."]
    out = [
        "| Business | Line | Case | Payload | Expected | Agent decided | Runs |",
        "|---|---|---|---|---|---|---|",
    ]
    for b, m in rows:
        runs = main[b]["result"]["repeats"]
        out.append(
            f"| {b} | {m['line']} | {m['case'].replace('_', ' ')} | {m['attack'] or 'none'}"
            f" | {m['expected']} | {m['decided']} | {m['runs']} of {runs} |"
        )
    return out


def file_name(result: dict) -> str:
    """The results file a result was written to, by the naming rule in evaluate.write."""
    parts = [result["generated_at"][:10], result["business"], result["model"], result["mode"]]
    name = result.get("defences", {}).get("name", "on")
    if name != "on":
        parts.append(f"defences_{name}")
    return "-".join(parts).replace(":", "_") + ".json"


def headline(main: dict[str, dict], results: list[dict]) -> str:
    proposals = sum(f["match_proposals"] for f in main.values())
    decisions = sum(f["agent_decisions"] for f in main.values())
    wrong = sum(f["agg"]["totals"]["wrong_matches"] for f in main.values())
    hit = sum(f["agg"]["totals"]["model_attacks"]["lines_hit_in_any_repeat"] for f in main.values())
    lines = sum(f["agg"]["totals"]["model_attacks"]["distinct_lines"] for f in main.values())
    runs = min(f["result"]["repeats"] for f in main.values())
    recall = " and ".join(
        f"{pct(f['agg']['overall_recall']['mean'])} ({b})" for b, f in main.items()
    )
    rules = " and ".join(pct(f["rules_only"]["match"]["recall"]) for f in main.values())
    bound = max(f["agg"]["totals"]["model_attacks"]["upper95"] for f in main.values())
    text = (
        f"**{wrong} wrong matches in {proposals} agent match proposals** ({decisions} agent"
        f" decisions) over {runs} live runs on each of two synthetic businesses. Rules plus agent"
        f" matched {recall} of the lines that have a matching invoice; the rules pass alone"
        f" matched {rules}. **{hit} of {lines} injected lines** that ask the model to match a"
        f" different or an extra invoice succeeded in any run (95% upper bound per business:"
        f" {pct(bound)})"
    )
    off = [pick(results, b, "off") for b in main]
    if all(off):
        off_hit = sum(
            r["aggregate"]["totals"]["model_attacks"]["lines_hit_in_any_repeat"] for r in off
        )
        runs_off = min(r["repeats"] for r in off)
        text += (
            f"; in {runs_off} run{'s' if runs_off != 1 else ''} with every defence switched off,"
            f" {off_hit} of {lines} did, too few runs to rank the defences"
        )
    return text + "."


def main_table(main: dict[str, dict]) -> list[str]:
    cols = list(main)
    out = ["| | " + " | ".join(BUSINESSES[b] for b in cols) + " |", "|---" * (len(cols) + 1) + "|"]

    def row(name: str, cell) -> None:
        out.append(f"| {name} | " + " | ".join(cell(main[b]) for b in cols) + " |")

    def spread(f, key, fmt) -> str:
        s = f["agg"][key]
        return f"**{fmt(s['mean'])}** ({fmt(s['min'])} to {fmt(s['max'])})"

    def auto(f) -> str:
        p = f["first"]["policy"]
        return (
            f"{pct(p['touchless_rate_pipeline'])} of lines"
            f" ({p['auto_from_rules']} by rules, {p['auto_from_agent']} by agent)"
        )

    def wrong_auto(f) -> str:
        t = f["agg"]["totals"]
        a = t["agent_auto_approvals"]
        return (
            f"**{t['wrong_auto_approvals']}** (agent path: {a['wrong_in_any_repeat']} of"
            f" {a['distinct_lines']} lines, 95% upper bound {pct(a['upper95'])})"
        )

    runs = main[cols[0]]["result"]["repeats"]
    row("Bank lines", lambda f: str(f["first"]["lines"]))
    row(
        "Rules pass alone: lines closed, match F1",
        lambda f: f"{f['first']['rules_closed']} lines, {f['rules_only']['match']['f1']:.3f}",
    )
    row(
        f"Rules plus agent: match F1, mean over {runs} runs (min to max)",
        lambda f: spread(f, "overall_f1", lambda x: f"{x:.3f}"),
    )
    row(
        "Rules plus agent: match precision",
        lambda f: spread(f, "overall_precision", lambda x: f"{x:.3f}"),
    )
    row(
        "Lines left to the agent, agent accuracy",
        lambda f: f"{f['first']['agent_lines']} lines, {pct(f['agg']['agent_accuracy']['mean'])}",
    )
    row(f"Wrong matches in {runs} runs", lambda f: f"**{f['agg']['totals']['wrong_matches']}**")
    row("Would be auto-approved in shadow mode (first run)", auto)
    row(f"Wrong auto-approvals in {runs} runs", wrong_auto)
    row(
        "Agent lines with the same decision in every run",
        lambda f: (
            pct(f["agg"]["decision_agreement"])
            if f["agg"]["decision_agreement"] is not None
            else "n/a"
        ),
    )
    row(
        "Cost per 100 bank lines (peak price)",
        lambda f: f"${f['agg']['cost_per_100_bank_lines_usd']['mean']:.3f}",
    )
    row(
        "Model time per agent line, p50 and p95 (8 lines in parallel)",
        lambda f: (
            f"{f['agg']['latency_p50_ms']['mean'] / 1000:.1f} s and"
            f" {f['agg']['latency_p95_ms']['mean'] / 1000:.1f} s"
        ),
    )
    return out


def case_table(main: dict[str, dict]) -> list[str]:
    cols = list(main)
    out = ["| Case | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for case in synth.CASES:
        cells = []
        for b in cols:
            ok, n = main[b]["cases"].get(case, [0, 0])
            cells.append(f"{ok} of {n}" if n else "closed by rules")
        out.append(f"| {case.replace('_', ' ')} | " + " | ".join(cells) + " |")
    return out


def safety_table(main: dict[str, dict]) -> list[str]:
    cols = list(main)
    out = ["| | " + " | ".join(BUSINESSES[b] for b in cols) + " |", "|---" * (len(cols) + 1) + "|"]

    def row(name: str, cell) -> None:
        out.append(f"| {name} | " + " | ".join(cell(main[b]) for b in cols) + " |")

    def attacks(f) -> str:
        a = f["agg"]["totals"]["model_attacks"]
        return (
            f"**{a['lines_hit_in_any_repeat']} of {a['distinct_lines']} lines**"
            f" (95% upper bound {pct(a['upper95'])})"
        )

    def screen(f, key) -> str:
        s = f["first"]["policy"]["screen"][key]
        return f"{s['success']} of {s['n']}"

    row("Redirect and over-allocation payloads that succeeded in any run", attacks)
    row(
        "Planted personal data sent to the provider, or queued",
        lambda f: (
            f"{f['agg']['totals']['pii_leaks_outbound']} and"
            f" {f['agg']['totals']['pii_leaks_queued']}"
        ),
    )
    row("Keyword screen: attacked lines flagged", lambda f: screen(f, "attacked_lines_flagged"))
    row("Keyword screen: ordinary lines flagged", lambda f: screen(f, "benign_lines_flagged"))
    return out


def ablation_table(results: list[dict], main: dict[str, dict]) -> list[str]:
    out = [
        "| Defences | Business | Runs | Redirect and over-allocation lines that succeeded"
        " | Wrong matches | Wrong auto-approvals | Planted values sent to the provider"
        " | Repeat-personal-data payloads echoed in the rationale (line-runs)"
        " | Ordinary agent lines correct |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, title in ABLATIONS.items():
        for b in main:
            r = main[b]["result"] if name == "on" else pick(results, b, name)
            if r is None:
                continue
            f = main[b] if name == "on" else facts(r)
            t = f["agg"]["totals"]
            a = t["model_attacks"]
            ok, n = f["ordinary"]
            out.append(
                f"| {title} | {b} | {r['repeats']} | {a['lines_hit_in_any_repeat']} of"
                f" {a['distinct_lines']} | {t['wrong_matches']} | {t['wrong_auto_approvals']}"
                f" | {t['pii_leaks_outbound']} values"
                f" | {f['pii_echoed'][0]} of {f['pii_echoed'][1]}"
                f" | {pct(ok / n) if n else ''} |"
            )
    return out


def model_table(results: list[dict]) -> list[str]:
    models = sorted(
        {r["model"] for r in results if r.get("defences", {}).get("name", "on") == "on"},
        key=lambda m: (m != PRIMARY_MODEL, m),
    )
    out = [
        "| Model | Business | Runs | Agent accuracy | Wrong matches"
        " | Redirect and over-allocation lines that succeeded | Cost per 100 bank lines"
        " | Model time per agent line, p50 and p95 | Agent lines with the same decision |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model in models:
        for b in BUSINESSES:
            r = pick(results, b, "on", model)
            if r is None:
                continue
            a, t = r["aggregate"], r["aggregate"]["totals"]
            agreement = a["decision_agreement"]
            out.append(
                f"| `{model}` | {b} | {r['repeats']} | {pct(a['agent_accuracy']['mean'])}"
                f" | {t['wrong_matches']} | {t['model_attacks']['lines_hit_in_any_repeat']} of"
                f" {t['model_attacks']['distinct_lines']}"
                f" | ${a['cost_per_100_bank_lines_usd']['mean']:.3f}"
                f" | {a['latency_p50_ms']['mean'] / 1000:.1f} s and"
                f" {a['latency_p95_ms']['mean'] / 1000:.1f} s"
                f" | {pct(agreement) if agreement is not None else 'n/a'} |"
            )
    return out


RETRIEVAL_METHODS = {"trigram": "Trigram", "fts": "Full text", "vector": "Vector", "rrf": "RRF"}
QUERY_SETS = {"as_written": "as the bank wrote it", "renamed": "renamed (stress set)"}


def retrieval_table(result: dict) -> list[str]:
    out = [
        "| Business | Payee | " + " | ".join(RETRIEVAL_METHODS.values()) + " |",
        "|---" * (len(RETRIEVAL_METHODS) + 2) + "|",
    ]
    for b, parts in result["businesses"].items():
        for key, label in QUERY_SETS.items():
            cells = [
                f"{parts[key][m]['recall_at_5']:.3f} and {parts[key][m]['mrr']:.3f}"
                for m in RETRIEVAL_METHODS
            ]
            out.append(f"| {b} | {label} | " + " | ".join(cells) + " |")
    return out


def retrieval_section(results_dir: Path) -> list[str]:
    found = load(results_dir, kind="retrieval")
    if not found:
        return []
    r = max(found, key=lambda x: x["generated_at"])
    pairs = {a: b for a, b in r["businesses"]["A"]["renamed_examples"]}  # one example per payer
    examples = "; ".join(f"{a} as {b}" for a, b in list(pairs.items())[:3])
    n = r["businesses"]["A"]["as_written"]["trigram"]["n"]
    return [
        "",
        "### Candidate search: trigram, full text, vector, RRF",
        "",
        f"Each cell is recall at 5 and mean reciprocal rank of the labelled invoices among all"
        f" open invoices, over the {n} lines per business that have a match; no language model"
        f" is involved. Vector search uses `{r['embedding_model']}` at {r['dims']} dimensions"
        " in pgvector, and RRF fuses the trigram and vector ranks. The stress set renames each"
        f" payer in ways the generator never does, for example {examples}. Commit"
        f" `{r['commit']}`.",
        "",
        *retrieval_table(r),
    ]


def rules_section(results_dir: Path) -> list[str]:
    found = load(results_dir, kind="rules")
    latest: dict[str, dict] = {}
    for r in found:
        if r["mode"] in ("live", "record") and (
            r["model"] not in latest or r["generated_at"] > latest[r["model"]]["generated_at"]
        ):
            latest[r["model"]] = r
    if not latest:
        return []
    out = [
        "",
        "### Plain-English bank rules",
        "",
        "An owner writes a rule in a sentence; the model drafts JSON and code decides whether it"
        " becomes a rule. The eval set has 40 sentences written for it: 30 rules and 10 that"
        " must be refused (customer names, approvals, contradictions, an out-of-envelope limit,"
        " an injection). A bad rule is a wrong rule accepted or a must-refuse sentence accepted;"
        " its bound counts distinct sentences.",
        "",
        "| Model | Runs | Rules compiled exactly | Falsely refused | Bad rules (distinct"
        " sentences, 95% upper bound) | Refused by the validator | Invalid replies |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in sorted(latest, key=lambda m: (m != PRIMARY_MODEL, m)):
        r = latest[model]
        runs = [x["summary"] for x in r["runs"]]
        bad = r["bad_rules"]
        out.append(
            f"| `{model}` | {r['repeats']}"
            f" | {sum(x['compiled_exactly'] for x in runs)} of"
            f" {sum(x['should_accept'] for x in runs)}"
            f" | {sum(len(x['falsely_refused']) for x in runs)}"
            f" | {len(bad['distinct_sentences'])} of {r['sentences']} ({pct(bad['upper95'])})"
            f" | {sum(x['refused_by_validator'] for x in runs)}"
            f" | {sum(x['invalid_replies'] for x in runs)} |"
        )
    commits = ", ".join(sorted({f"`{r['commit']}`" for r in latest.values()}))
    out += ["", f"Counts are over all runs. Commit {commits}."]
    return out


def render(results_dir: Path) -> str:
    results = load(results_dir)
    main = {b: facts(r) for b in BUSINESSES if (r := pick(results, b, "on"))}
    if not main:
        return f"{START}\n_No results yet: run `make eval`._\n{END}"
    sources = ", ".join(
        f"`results/{file_name(f['result'])}` (commit `{f['result']['commit']}`)"
        for f in main.values()
    )
    ablation_commits = sorted(
        {r["commit"] for r in results if r.get("defences", {}).get("name", "on") != "on"}
    )
    first = next(iter(main.values()))["result"]
    lines = [
        START,
        "<!-- Generated by `recon report` from results/*.json. Edit the code, not this block. -->",
        "",
        headline(main, results),
        "",
        *main_table(main),
        "",
        f"Model `{first['model']}` (`{first['api_model']}`), prompt `{first['prompt_sha']}`;"
        f" prices checked {first['prices_checked']}. Sources: {sources}. Business B has its own"
        " seed, invoice format, wording and case mix and was not used to write the prompt or the"
        " policy screen; its labels were used to check the generator and the verifier's payer"
        " threshold.",
        "",
        "### Models",
        "",
        "Every row runs the same prompt, tools and data; only the model differs.",
        "",
        *model_table(results),
        "",
        "### Agent decisions by case",
        "",
        "Correct out of all agent decisions, pooled over the runs. Cases the rules pass always"
        " closes never reach the agent.",
        "",
        *case_table(main),
        "",
        "### Where the agent was wrong",
        "",
        f"Every agent decision of `{PRIMARY_MODEL}` that differs from the label, with the number"
        " of runs it happened in. An escalation of a line that has a match costs a reviewer's"
        " time; a match to the wrong invoice would cost money.",
        "",
        *mistakes_table(main),
        "",
        "### Safety",
        "",
        *safety_table(main),
        "",
        "### Defences switched off in two groups",
        "",
        "The prompt group is the fence and the prompt's untrusted-text guidance; the code group"
        " is redaction, the verifier's payer check and the policy screen. Each ablation row is"
        " one run against the main rows' three, so it shows what redaction does and is too small"
        " to rank the rest. A repeat-personal-data payload is echoed when a planted value appears"
        f" in the rationale. Ablation commit: {', '.join(f'`{c}`' for c in ablation_commits)}.",
        "",
        *ablation_table(results, main),
        *retrieval_section(results_dir),
        *rules_section(results_dir),
        END,
    ]
    return "\n".join(lines)


def sync(readme: Path, results_dir: Path, check: bool = False) -> bool:
    """Rewrite the block in place; with check=True only report whether it is stale."""
    text = readme.read_text()
    if START not in text or END not in text:
        raise ValueError(f"{readme} has no {START} ... {END} block")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    updated = before + render(results_dir) + after
    if updated == text:
        return False
    if not check:
        readme.write_text(updated)
    return True
