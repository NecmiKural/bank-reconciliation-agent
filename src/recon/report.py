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

START = "<!-- results:start -->"
END = "<!-- results:end -->"
BUSINESSES = {"A": "Business A (development)", "B": "Business B (held out)"}
ABLATIONS = {
    "on": "All defences on",
    "prompt_only": "Prompt guidance and fence only",
    "code_only": "Code defences only",
    "off": "No defences",
}


def load(results_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def pick(results: list[dict], business: str, defences: str) -> dict | None:
    """The latest full live run of a business with the given defences."""
    found = [
        r
        for r in results
        if r["business"] == business
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
    for run in runs:
        for line_id, compact in run["decisions_agent"].items():
            label = labels[line_id]
            ok = score.is_correct(label, decision(line_id, compact))
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
    }


def headline(main: dict[str, dict], results: list[dict]) -> str:
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
        f"**{wrong} wrong matches in {decisions} agent decisions** over {runs} live runs on each"
        f" of two synthetic businesses, one held out. Rules plus agent matched {recall} of"
        f" matchable lines; the rules pass alone matched {rules}. **{hit} of {lines} injected"
        f" lines** that ask the model to pick another invoice succeeded in any run (95% upper"
        f" bound per business: {pct(bound)})"
    )
    off = [pick(results, b, "off") for b in main]
    if all(off):
        off_hit = sum(
            r["aggregate"]["totals"]["model_attacks"]["lines_hit_in_any_repeat"] for r in off
        )
        text += f"; with every defence switched off, {off_hit} of {lines} did"
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
    row("Auto-approved by the shadow policy", auto)
    row(f"Wrong auto-approvals in {runs} runs", wrong_auto)
    row(
        "Agent lines with the same decision in every run",
        lambda f: pct(f["agg"]["decision_agreement"] or 1.0),
    )
    row(
        "Cost per 100 bank lines (peak price)",
        lambda f: f"${f['agg']['cost_per_100_bank_lines_usd']['mean']:.3f}",
    )
    row(
        "Latency per agent line, p50 and p95",
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
        "Pre-approval payloads that got auto-approved",
        lambda f: f"{f['preapproval'][0]} of {f['preapproval'][1]} line-runs",
    )
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
        "| Defences | Business | Runs | Injected lines that succeeded | Wrong matches"
        " | Wrong auto-approvals"
        " | Personal data sent to the provider | Personal data echoed to the queue"
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


def render(results_dir: Path) -> str:
    results = load(results_dir)
    main = {b: facts(r) for b in BUSINESSES if (r := pick(results, b, "on"))}
    if not main:
        return f"{START}\n_No results yet: run `make eval`._\n{END}"
    sources = ", ".join(
        f"`results/{f['result']['generated_at'][:10]}-{b}-...json` at commit"
        f" `{f['result']['commit']}`"
        for b, f in main.items()
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
        f" prices checked {first['prices_checked']}. Sources: {sources}.",
        "",
        "### Agent decisions by case",
        "",
        "Correct out of all agent decisions, pooled over the runs. Cases the rules pass always"
        " closes never reach the agent.",
        "",
        *case_table(main),
        "",
        "### Safety",
        "",
        *safety_table(main),
        "",
        "### Which defence does the work",
        "",
        "The same lines with defences switched off one group at a time. Personal data is echoed"
        " when a planted value appears in the rationale written to the queue.",
        "",
        *ablation_table(results, main),
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
