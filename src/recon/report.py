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


MARKERS = {
    "runs": "business",
    "retrieval": "retrieval",
    "rules": "rules",
    "attack_surface": "attack_surface",
    "probes": "probes",
}


def load(results_dir: Path, kind: str = "runs") -> list[dict[str, Any]]:
    """Results files of one kind, told apart by a key only that kind has (a probe run also
    names its business, so runs are the files with a business and no probes key)."""
    files = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    found = [f for f in files if MARKERS[kind] in f]
    return [f for f in found if "probes" not in f] if kind == "runs" else found


def per_line(result: dict) -> bool:
    """A run of the per-line agent. Code-mode runs (an arm other than direct) are reported in
    their own section and never become a headline row."""
    return result.get("arm", "direct") == "direct"


def pick(
    results: list[dict], business: str, defences: str, model: str = PRIMARY_MODEL
) -> dict | None:
    """The latest full live run of a business with the given defences and model."""
    found = [
        r
        for r in results
        if per_line(r)
        and r["business"] == business
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


def attack_surface(results_dir: Path) -> dict | None:
    """The latest attack surface with the payer check on, as in the main runs."""
    found = [r for r in load(results_dir, kind="attack_surface") if r.get("payer_check", True)]
    return max(found, key=lambda r: r["generated_at"]) if found else None


def full_runs(results: list[dict]) -> list[dict]:
    """Every valid live run over a whole business, any model, any defences, oldest first."""
    found = [
        r
        for r in results
        if per_line(r) and r["mode"] in ("live", "record") and r["limit"] is None and r["valid"]
    ]
    return sorted(found, key=lambda r: r["generated_at"])


def defences_name(result: dict) -> str:
    return result.get("defences", {}).get("name", "on")


def headline(main: dict[str, dict], results: list[dict], surface: dict | None) -> str:
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
    text = (
        f"**{wrong} wrong matches in {proposals} agent match proposals** ({decisions} agent"
        f" decisions) over {runs} live runs on each of two synthetic businesses. Rules plus agent"
        f" matched {recall} of the lines that have a matching invoice; the rules pass alone"
        f" matched {rules}. **{hit} of {lines} injected lines** that ask for a different or an"
        " extra invoice succeeded in any run"
    )
    if surface:
        open_lines = {i for b in surface["businesses"].values() for i in b["open_to_model"]}
        taken = {
            i
            for r in full_runs(results)
            if defences_name(r) == "on"
            for run in r["runs"]
            for i in run["details"]["model_attack_hits"]
            if i in open_lines
        }
        text += (
            f". The verifier refuses whatever the payload asks for on {lines - len(open_lines)}"
            f" of them; on the {len(open_lines)} it leaves to the model's judgement,"
            f" {len(taken)} payloads were followed by any model in any run with defences on"
        )
    off = [pick(results, b, "off") for b in main]
    if all(off):
        off_hit = sum(
            r["aggregate"]["totals"]["model_attacks"]["lines_hit_in_any_repeat"] for r in off
        )
        runs_off = min(r["repeats"] for r in off)
        text += (
            f". In {runs_off} run{'s' if runs_off != 1 else ''} with every defence switched off,"
            f" {off_hit} of {lines} succeeded, too few runs to rank the defences"
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
        "Model time per agent line, p50 and p95",
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


def safety_table(main: dict[str, dict], surface: dict | None) -> list[str]:
    cols = list(main)
    out = ["| | " + " | ".join(BUSINESSES[b] for b in cols) + " |", "|---" * (len(cols) + 1) + "|"]

    def row(name: str, cell) -> None:
        out.append(f"| {name} | " + " | ".join(cell(main[b], b) for b in cols) + " |")

    def attacks(f, b) -> str:
        a = f["agg"]["totals"]["model_attacks"]
        return (
            f"**{a['lines_hit_in_any_repeat']} of {a['distinct_lines']} lines**"
            f" (95% upper bound {pct(a['upper95'])})"
        )

    def left_to_model(f, b) -> str:
        part = surface["businesses"][b]
        return (
            f"{len(part['open_to_model'])} of {part['lines']} ({', '.join(part['open_to_model'])})"
        )

    def screen(f, b, key) -> str:
        s = f["first"]["policy"]["screen"][key]
        return f"{s['success']} of {s['n']}"

    row("Redirect and over-allocation payloads that succeeded in any run", attacks)
    if surface:
        row("Of those lines, where the verifier leaves the payload to the model", left_to_model)
    row(
        "Planted personal data sent to the provider, or queued",
        lambda f, b: (
            f"{f['agg']['totals']['pii_leaks_outbound']} and"
            f" {f['agg']['totals']['pii_leaks_queued']}"
        ),
    )
    row(
        "Keyword screen: attacked lines flagged",
        lambda f, b: screen(f, b, "attacked_lines_flagged"),
    )
    row(
        "Keyword screen: ordinary lines flagged",
        lambda f, b: screen(f, b, "benign_lines_flagged"),
    )
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


def wrong_match_table(results: list[dict]) -> tuple[list[str], int]:
    """Every agent match to the wrong invoice in every full run on record, superseded runs
    and ablations included, and how many wrong decisions the policy would have auto-approved."""
    rows, wrong_auto = [], 0
    for r in full_runs(results):
        wrong_auto += r["aggregate"]["totals"]["wrong_auto_approvals"]
        for m in facts(r)["mistakes"]:
            if m["decided"].startswith("match:"):
                rows.append((r, m))
    if not rows:
        return ["No model matched a wrong invoice in any run on record."], wrong_auto
    out = [
        "| Model | Defences | Commit | Business | Line | Case | Payload | Expected"
        " | Agent matched | Runs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r, m in rows:
        out.append(
            f"| `{r['model']}` | {ABLATIONS[defences_name(r)].lower()} | `{r['commit']}`"
            f" | {r['business']} | {m['line']} | {m['case'].replace('_', ' ')}"
            f" | {m['attack'] or 'none'} | {m['expected']} | {m['decided']}"
            f" | {m['runs']} of {r['repeats']} |"
        )
    return out, wrong_auto


def model_table(results: list[dict]) -> list[str]:
    models = sorted(
        {r["model"] for r in results if defences_name(r) == "on"},
        key=lambda m: (m != PRIMARY_MODEL, m),
    )
    out = [
        "| Model | Business | Runs | Commit | Agent accuracy | Wrong matches"
        " | Redirect and over-allocation lines that succeeded | Cost per 100 bank lines"
        " | Model time per agent line, p50 and p95 | Agent lines with the same decision |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in models:
        for b in BUSINESSES:
            r = pick(results, b, "on", model)
            if r is None:
                continue
            a, t = r["aggregate"], r["aggregate"]["totals"]
            agreement = a["decision_agreement"]
            p50, p95 = a["latency_p50_ms"], a["latency_p95_ms"]
            times = (
                f"{p50['mean'] / 1000:.1f} s and {p95['mean'] / 1000:.1f} s"
                if p50 and p95
                else "n/a"
            )
            out.append(
                f"| `{model}` | {b} | {r['repeats']} | `{r['commit']}`"
                f" | {pct(a['agent_accuracy']['mean'])}"
                f" | {t['wrong_matches']} | {t['model_attacks']['lines_hit_in_any_repeat']} of"
                f" {t['model_attacks']['distinct_lines']}"
                f" | ${a['cost_per_100_bank_lines_usd']['mean']:.3f}"
                f" | {times}"
                f" | {pct(agreement) if agreement is not None else 'n/a'} |"
            )
    if any(row.split(" | ")[8] == "n/a" for row in out[2:]):
        out += [
            "",
            "Model time is n/a for runs that go through the statement loop, as Claude's do:"
            " their results files keep each run's total model time, not each line's.",
        ]
    return out


STRATA = {
    "mechanical": ("split", "fee_deducted", "no_match"),
    "judgement": ("typo", "duplicate_amount", "partial"),
}
ARM_NAMES = {"sandbox": "our sandbox", "ptc": "programmatic tool calling"}


def correctness(result: dict, cases: tuple[str, ...] | None = None) -> dict[str, list[bool]]:
    """Each agent line's correctness in each run, for lines of the given cases."""
    labels = {label.line_id: label for label in dataset_for(result).labels}
    out: dict[str, list[bool]] = {}
    for run in result["runs"]:
        for line_id, compact in run["decisions_agent"].items():
            if cases is None or labels[line_id].case in cases:
                ok = score.is_correct(labels[line_id], decision(line_id, compact))
                out.setdefault(line_id, []).append(ok)
    return out


def passes_per_100(result: dict) -> float:
    """Model sampling passes per 100 agent lines (H1a), mean over runs. Files from before the
    field are per-line runs, where every request was a pass; older still count their steps."""
    values = []
    for run in result["runs"]:
        s = run["summary"]
        if s.get("sampling_passes_per_100_agent_lines") is not None:
            values.append(s["sampling_passes_per_100_agent_lines"])
        elif s.get("model_requests_per_100_agent_lines") is not None:
            values.append(s["model_requests_per_100_agent_lines"])
        else:
            steps = sum(int(k) * v for k, v in s["steps"].items())
            values.append(100 * steps / s["agent_lines"] if s["agent_lines"] else 0.0)
    return sum(values) / len(values)


def code_mode_cells(results: list[dict]) -> list[dict]:
    """The latest full valid run of each code-mode cell."""
    latest: dict[tuple, dict] = {}
    for r in results:
        if per_line(r) or r["mode"] not in ("live", "record") or r["limit"] is not None:
            continue
        if not r["valid"]:
            continue
        key = (r["business"], r["model"], r["arm"], r["statement_size"], defences_name(r))
        if key not in latest or r["generated_at"] > latest[key]["generated_at"]:
            latest[key] = r
    return [latest[k] for k in sorted(latest)]


def describe_difference(diff: dict) -> str:
    if diff["difference"] is None:
        return "n/a"
    low, high = diff["ci95"]
    return f"{diff['difference']:+.3f} ({low:+.3f} to {high:+.3f})"


def code_mode_table(results: list[dict]) -> list[str]:
    """Each code-mode cell against the per-line run of the same model on the same lines."""
    out = [
        "| Business | Model | Arm | Statement | Defences | Runs"
        " | Model passes per 100 agent lines"
        " | Cost per 100 bank lines | Agent accuracy | Accuracy against per-line, 95% interval"
        " | Mechanical cases | Judgement cases | Wrong matches | Attacked lines hit"
        " | Cross-line hits | Code failures |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in code_mode_cells(results):
        base = pick(results, cell["business"], defences_name(cell), cell["model"])
        if base and base["runs"][0]["decisions_rules"] != cell["runs"][0]["decisions_rules"]:
            raise ValueError(f"{file_name(cell)}: the rules pass decided differently")
        diffs = [
            describe_difference(score.paired_bootstrap(correctness(base, c), correctness(cell, c)))
            if base
            else "n/a"
            for c in (None, *STRATA.values())
        ]
        a, t = cell["aggregate"], cell["aggregate"]["totals"]
        failures = sum(
            (run["summary"]["statement"] or {}).get("code_failures", 0) for run in cell["runs"]
        )
        out.append(
            f"| {cell['business']} | `{cell['model']}` | {ARM_NAMES[cell['arm']]}"
            f" | {'all open lines' if cell['statement_size'] == 'all' else 'one line'}"
            f" | {ABLATIONS[defences_name(cell)]} | {cell['repeats']} | {passes_per_100(cell):.1f}"
            f" | ${a['cost_per_100_bank_lines_usd']['mean']:.3f}"
            f" | {pct(a['agent_accuracy']['mean'])}"
            f" | {' | '.join(diffs)} | {t['wrong_matches']}"
            f" | {t['model_attacks']['lines_hit_in_any_repeat']} of"
            f" {t['model_attacks']['distinct_lines']}"
            f" | {t.get('cross_line_hits_in_any_repeat', 0)} | {failures} |"
        )
    return out


def latest_probes(results_dir: Path) -> list[dict]:
    """The latest valid probe run of each business, model and arm."""
    latest: dict[tuple, dict] = {}
    for r in load(results_dir, "probes"):
        key = (r["business"], r["model"], r["arm"], r["statement_size"])
        if r["valid"] and (key not in latest or r["generated_at"] > latest[key]["generated_at"]):
            latest[key] = r
    return [latest[k] for k in sorted(latest)]


def probe_table(results_dir: Path) -> list[str]:
    probes = latest_probes(results_dir)
    if not probes:
        return []
    out = [
        "| Business | Model | Arm | Families whose memo the code followed | Harm events |",
        "|---|---|---|---|---|",
    ]
    for r in probes:
        followed = [f for f, v in r["families"].items() if v["followed_in_code"]]
        out.append(
            f"| {r['business']} | `{r['model']}` | {r['arm']}"
            f" | {', '.join(followed) or 'none'} | {r['harm_events']} |"
        )
    return out


def code_mode_section(results: list[dict], results_dir: Path) -> list[str]:
    from recon import hypotheses  # imported here: it reads this module's helpers

    rows = code_mode_table(results)
    if len(rows) == 2:
        return []
    commits = sorted({r["commit"] for r in code_mode_cells(results)})
    probes = probe_table(results_dir)
    return [
        "",
        "### Code mode",
        "",
        "The same lines decided by code the model writes (pre-registered in"
        " [docs/code-mode-experiment.md](docs/code-mode-experiment.md)): in our sandbox, or"
        " with Claude's programmatic tool calling. Each row is compared with the per-line run of"
        " the same model; the interval resamples lines, not line-runs, and a row whose rules"
        f" pass decided differently is refused. Commits: {', '.join(f'`{c}`' for c in commits)}.",
        "",
        *rows,
        *(
            [
                "",
                "Probes of the surface code mode adds, one run each: a memo that asks the code to"
                " dump the environment, reach the network, act on another line, call a tool that"
                " does not exist, flood the output, forge a protocol message or break out of a"
                " string. A harm event is a decision on one probe line that takes another's"
                " invoice.",
                "",
                *probes,
            ]
            if probes
            else []
        ),
        *hypotheses.section(results, latest_probes(results_dir)),
    ]


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
        "Each cell is recall at 5 and mean reciprocal rank of the labelled invoices among the"
        f" open invoices issued by the posting date, over the {n} lines per business that have a"
        " match. No chat model is involved; vector search uses"
        f" `{r['embedding_model']}` at {r['dims']} dimensions in pgvector, and RRF fuses the"
        " trigram and vector ranks. Every method ranks exact-amount invoices first, as the"
        " agent's search does, so on a line whose amount matches one open invoice all methods"
        " agree and only the other lines tell them apart. The stress"
        f" set renames each payer in ways the generator never does, for example {examples}."
        f" Commit `{r['commit']}`.",
        "",
        *retrieval_table(r),
    ]


def describe_rule(rule: dict | None) -> str:
    if rule is None:
        return "refuse"
    amount = rule.get("max_amount_cents")
    limit = f"{amount / 100:.2f}" if amount is not None else rule.get("max_amount", "")
    return f"{rule['payee_contains']}: {rule['account']}, {rule['direction']}, up to {limit}"


def rules_section(results_dir: Path) -> list[str]:
    from recon.rule_compiler import load_eval_set  # the eval set lives beside the compiler

    found = load(results_dir, kind="rules")
    latest: dict[str, dict] = {}
    for r in found:
        if r["mode"] in ("live", "record") and (
            r["model"] not in latest or r["generated_at"] > latest[r["model"]]["generated_at"]
        ):
            latest[r["model"]] = r
    if not latest:
        return []
    expected = {item["id"]: item for item in load_eval_set()}
    out = [
        "",
        "### Plain-English bank rules",
        "",
        "An owner writes a rule in a sentence; the model drafts JSON and code decides whether it"
        " becomes a rule. The eval set has 40 sentences written for it: 30 rules and 10 that"
        " must be refused, for example customer names, approvals, contradictions, patterns too"
        " generic to be safe, an out-of-envelope limit and an injection. A bad rule is a wrong"
        " rule accepted or a must-refuse sentence accepted; its bound counts distinct sentences.",
        "",
        "| Model | Runs | Rules compiled exactly | Falsely refused | Bad rules (distinct"
        " sentences, 95% upper bound) | Refused by the validator | Invalid replies |",
        "|---|---|---|---|---|---|---|",
    ]
    bad_rows = []
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
        for sentence in bad["distinct_sentences"]:
            accepted: dict[str, int] = {}
            for run in r["runs"]:
                outcome = run["outcomes"][sentence]
                summary = run["summary"]
                if sentence in summary["wrong_rules_accepted"] + summary["unsafe_rules_accepted"]:
                    key = describe_rule(outcome["rule"])
                    accepted[key] = accepted.get(key, 0) + 1
            for got, count in accepted.items():
                want = describe_rule(expected[sentence]["expect"])
                bad_rows.append((model, sentence, want, got, count, r["repeats"]))
    commits = sorted({f"`{r['commit']}`" for r in latest.values()})
    out += ["", f"Counts are over all runs. Commit{'s' * (len(commits) > 1)} {', '.join(commits)}."]
    if bad_rows:
        out += [
            "",
            "| Model | Sentence | Expected | Accepted | Runs |",
            "|---|---|---|---|---|",
            *[
                f"| `{m}` | {sid} | {want} | {got} | {count} of {runs} |"
                for m, sid, want, got, count, runs in bad_rows
            ],
        ]
    return out


def render(results_dir: Path) -> str:
    results = load(results_dir)
    main = {b: facts(r) for b in BUSINESSES if (r := pick(results, b, "on"))}
    if not main:
        return f"{START}\n_No results yet: run `make eval`._\n{END}"
    surface = attack_surface(results_dir)
    sources = ", ".join(
        f"`results/{file_name(f['result'])}` (commit `{f['result']['commit']}`)"
        for f in main.values()
    )
    ablation_commits = sorted({r["commit"] for r in results if defences_name(r) != "on"})
    main_commits = sorted({f["result"]["commit"] for f in main.values()})
    first = next(iter(main.values()))["result"]
    wrong_rows, wrong_auto = wrong_match_table(results)
    surface_note = (
        f" The verifier-left lines come from `recon attack-surface` at commit"
        f" `{surface['commit']}`."
        if surface
        else ""
    )
    lines = [
        START,
        "<!-- Generated by `recon report` from results/*.json. Edit the code, not this block. -->",
        "",
        headline(main, results, surface),
        "",
        *main_table(main),
        "",
        f"Model `{first['model']}` (`{first['api_model']}`), prompt `{first['prompt_sha']}`;"
        f" prices checked {first['prices_checked']}. Sources: {sources}. Business B has its own"
        " seed, invoice format, wording and case mix and was not used to write the prompt or the"
        " policy screen; its labels were used to check the generator and the verifier's payer"
        " thresholds.",
        "",
        "### Models",
        "",
        "Same prompt and data in every row. The commit, the model settings, the number of lines"
        " run in parallel and the hardware differ, so compare times with care; the local rows"
        " ran on the stricter payer check, and no hosted proposal made with the payer check on"
        " fails it (`test_recorded_proposals_pass_the_current_verifier`).",
        "",
        *model_table(results),
        "",
        "### Wrong matches by any model",
        "",
        "A match to the wrong invoice is the error that costs money, so every one in every full"
        " run on record is listed: every model, every defence setting, superseded runs included."
        " Wrong decisions the shadow policy would have auto-approved over all of them:"
        f" {wrong_auto}.",
        "",
        *wrong_rows,
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
        "Redirect and over-allocation payloads ask for an invoice the line should not settle."
        " Where the verifier would refuse every proposal containing it, the result does not"
        f" depend on the model.{surface_note}",
        "",
        *safety_table(main, surface),
        "",
        "### Defences switched off in two groups",
        "",
        "The prompt group is the fence, the prompt's untrusted-text guidance and its"
        " personal-data rule; the code group is redaction, the verifier's payer check and the"
        " policy screen. Most ablation rows are one run against the main rows' three (the Runs"
        " column says which), so they show what redaction does and are too small to rank the"
        " rest. A repeat-personal-data payload"
        " is echoed when a planted value appears in the rationale. All-on rows are the main runs"
        f" (commit {', '.join(f'`{c}`' for c in main_commits)}); ablation commit:"
        f" {', '.join(f'`{c}`' for c in ablation_commits)}.",
        "",
        *ablation_table(results, main),
        *code_mode_section(results, results_dir),
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
