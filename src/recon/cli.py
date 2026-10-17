"""Command line entry point: `recon <command>`."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from recon import agent, db, llm, rules, score, synth, tools
from recon.trace import Tracer

SPECS = {spec.name: spec for spec in synth.SPECS}


def cmd_synth(args: argparse.Namespace) -> None:
    for spec in synth.SPECS:
        ds = synth.generate(spec)
        path = synth.write(ds, args.out)
        print(f"{path}: {len(ds.lines)} lines, {len(ds.invoices)} invoices")


def cmd_load(args: argparse.Namespace) -> None:
    with db.connect() as conn:
        db.reset(conn, synth.generate(SPECS[args.business]))
    print(f"loaded business {args.business}")


def cmd_baseline(args: argparse.Namespace) -> None:
    ds = synth.generate(SPECS[args.business])
    with db.connect() as conn:
        db.reset(conn, ds)
        decisions = rules.run(conn)
    print(json.dumps(score.score(ds.labels, decisions), indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    spec = llm.MODELS[args.model]
    ds = synth.generate(SPECS[args.business])
    safe_model = args.model.replace(":", "_")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{args.business}-{safe_model}-{stamp}"
    cassette = args.cassette or Path("cassettes") / f"{args.business}-{safe_model}.jsonl"
    model = llm.LLM(spec, args.mode, None if args.mode == "live" else cassette)
    tracer = Tracer(Path("runs") / run_id / "trace.jsonl")
    labels = {label.line_id: label for label in ds.labels}

    with db.connect() as conn:
        db.reset(conn, ds)
        decisions = rules.run(conn)
        open_lines = [line for line in ds.lines if line.id not in decisions][: args.limit]
        search = tools.db_search(conn)
        results = {}
        for line in open_lines:
            result = agent.reconcile(line, model, search, agent.Limits())
            for row in result.trace:
                tracer.emit({"recon.run_id": run_id, **row})
            results[line.id] = result
            decisions[line.id] = result.decision
            label = labels[line.id]
            mark = "ok " if score.is_correct(label, result.decision) else "BAD"
            attack = f" [{label.attack_goal}]" if label.attack_goal else ""
            shown = ",".join(result.decision.invoice_ids) or result.decision.rationale[:60]
            print(
                f"{mark} {line.id} {label.case:16}{attack:16} -> {result.decision.decision:8}"
                f" {shown} ({result.steps} steps, ${result.cost_usd:.4f})"
            )
        db.queue(conn, run_id, decisions.values())

    agent_only = [labels[line_id] for line_id in results]
    provider_errors = sum(
        r.decision.rationale.startswith("provider error") for r in results.values()
    )
    summary = {
        "run_id": run_id,
        "model": args.model,
        "mode": args.mode,
        "rules_closed": len(decisions) - len(results),
        "agent_lines": len(results),
        "limited": args.limit is not None,  # lines left unprocessed count as escalations
        "provider_errors": provider_errors,
        "agent": score.score(agent_only, decisions),
        "rules_plus_agent": score.score(ds.labels, decisions),
        "cost_usd": round(sum(r.cost_usd for r in results.values()), 4),
        "tokens": {
            "input": sum(r.input_tokens for r in results.values()),
            "cached": sum(r.cached_tokens for r in results.values()),
            "output": sum(r.output_tokens for r in results.values()),
        },
    }
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="recon", description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("synth", help="write the synthetic businesses as JSON")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.set_defaults(func=cmd_synth)

    for name, func, text in (
        ("load", cmd_load, "recreate the database and load one business"),
        ("baseline", cmd_baseline, "load one business and score the rules pass alone"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--business", choices=sorted(SPECS), default="A")
        p.set_defaults(func=func)

    p = sub.add_parser("run", help="rules pass, then the agent on the lines it leaves open")
    p.add_argument("--business", choices=sorted(SPECS), default="A")
    p.add_argument("--model", choices=sorted(llm.MODELS), default="deepseek-flash")
    p.add_argument("--mode", choices=("live", "record", "replay"), default="live")
    p.add_argument("--cassette", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N open lines")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
