"""Command line entry point: `recon <command>`."""

from __future__ import annotations

import argparse
import getpass
import json
from datetime import UTC, datetime
from pathlib import Path

from recon import agent, db, evaluate, llm, report, retrieval, rules, score, synth

SPECS = {spec.name: spec for spec in synth.SPECS}


def default_cassette(business: str, model: str) -> Path:
    return Path("cassettes") / f"{business}-{evaluate.safe_name(model)}.jsonl"


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
    """One run with a line-by-line printout; leaves its proposals in the queue for review."""
    ds = synth.generate(SPECS[args.business])
    cassette = args.cassette or default_cassette(args.business, args.model)
    model = llm.LLM(llm.MODELS[args.model], args.mode, None if args.mode == "live" else cassette)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{args.business}-{evaluate.safe_name(args.model)}-{stamp}"
    defences = agent.DEFENCES[args.defences]
    run = evaluate.run_once(ds, model, run_id, args.limit, args.workers, defences=defences)
    labels = {label.line_id: label for label in ds.labels}
    for line in ds.lines:
        result = run.agent_results.get(line.id)
        if result is None:
            continue
        label, decision = labels[line.id], result.decision
        mark = "ok " if score.is_correct(label, decision) else "BAD"
        attack = f" [{label.attack_goal}]" if label.attack_goal else ""
        shown = ",".join(decision.invoice_ids) or printable(decision.rationale)[:60]
        policy = "auto" if run.verdicts[line.id].auto else "review"
        print(
            f"{mark} {line.id} {label.case:16}{attack:16} -> {decision.decision:8} {shown}"
            f" [{policy}] ({result.steps} steps, ${result.cost_usd:.4f})"
        )
    summary, _ = evaluate.summarize(ds, run)
    print(json.dumps({"run_id": run_id, **summary}, indent=2))
    print(f"review with: recon review list --run {run_id}")


def cmd_eval(args: argparse.Namespace) -> None:
    cassette = args.cassette or default_cassette(args.business, args.model)
    result = evaluate.evaluate(
        args.business,
        args.model,
        args.repeats,
        args.mode,
        None if args.mode == "live" else cassette,
        args.limit,
        args.workers,
        defences_name=args.defences,
    )
    print(json.dumps(result["aggregate"], indent=2))
    if not result["valid"]:
        raise SystemExit("provider errors made this run invalid; results were not saved")
    if args.save:
        print(f"wrote {evaluate.write(result, args.out, args.name)}")


def cmd_mcp(args: argparse.Namespace) -> None:
    from recon import mcp_server  # imported here: the MCP SDK is only needed for this command

    mcp_server.build().run("stdio")


def cmd_retrieval(args: argparse.Namespace) -> None:
    spec = retrieval.GEMINI
    cache = Path("cassettes") / f"embeddings-{spec.name}-{spec.dims}.json"
    result = retrieval.run(spec, cache)
    for business, parts in result["businesses"].items():
        for queries in ("as_written", "renamed"):
            row = "  ".join(
                f"{m} R@5 {v['recall_at_5']:.3f} MRR {v['mrr']:.3f}"
                for m, v in parts[queries].items()
            )
            print(f"{business} {queries:10} {row}")
    day = result["generated_at"][:10]
    path = args.out / f"{day}-retrieval-{spec.name}.json"
    path.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path}")


def cmd_report(args: argparse.Namespace) -> None:
    stale = report.sync(Path("README.md"), Path("results"), check=args.check)
    if args.check and stale:
        raise SystemExit("README results are stale: run `uv run recon report`")
    print("README results are up to date" if not stale else "README results rewritten")


def printable(text: str) -> str:
    """Model- and payer-written text for a terminal: control characters dropped, and the
    dashes this project does not print turned into commas (recordings keep the original)."""
    text = text.replace(" \u2014 ", ", ").replace("\u2014", ", ").replace("\u2013", ", ")
    return "".join(ch for ch in text if ch.isprintable())


def cmd_review(args: argparse.Namespace) -> None:
    with db.connect() as conn:
        if args.action == "list":
            for pid, line_id, cents, payee, decision, ids, note, why in db.pending(conn, args.run):
                target = ",".join(ids) or "-"
                print(
                    f"#{pid} {line_id} {cents / 100:>10,.2f} {printable(payee):20}"
                    f" {decision:8} {target}"
                )
                print(f"     policy: {note}; agent: {printable(why)[:100]}")
        else:
            approve = args.action == "approve"
            db.review(conn, args.proposal, approve, args.note, getpass.getuser())
            print(f"proposal {args.proposal} {'approved' if approve else 'rejected'}")


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

    for name, func, text in (
        ("run", cmd_run, "rules, agent and shadow policy once, printed line by line"),
        ("eval", cmd_eval, "repeated runs, measured; writes results/*.json"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--business", choices=sorted(SPECS), default="A")
        p.add_argument("--model", choices=sorted(llm.MODELS), default="deepseek-flash")
        p.add_argument("--mode", choices=("live", "record", "replay"), default="live")
        p.add_argument("--cassette", type=Path, default=None)
        p.add_argument("--limit", type=int, default=None, help="only the first N open lines")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--defences", choices=sorted(agent.DEFENCES), default="on")
        p.set_defaults(func=func)
        if name == "eval":
            p.add_argument("--repeats", type=int, default=3)
            p.add_argument("--out", type=Path, default=Path("results"))
            p.add_argument("--name", default=None, help="file name without .json")
            p.add_argument("--no-save", dest="save", action="store_false")

    p = sub.add_parser("mcp", help="serve the read-only ledger tools over MCP on stdio")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("retrieval", help="compare trigram, full-text, vector and RRF search")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.set_defaults(func=cmd_retrieval)

    p = sub.add_parser("report", help="rewrite the README results block from results/*.json")
    p.add_argument("--check", action="store_true", help="fail if the block is stale")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("review", help="list, approve or reject proposals as the reviewer")
    actions = p.add_subparsers(dest="action", required=True)
    a = actions.add_parser("list", help="proposals of a run that wait for a human")
    a.add_argument("--run", required=True)
    for action in ("approve", "reject"):
        a = actions.add_parser(action, help=f"{action} one proposal, with an audit row")
        a.add_argument("proposal", type=int)
        a.add_argument("--note", default="")
    p.set_defaults(func=cmd_review)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
