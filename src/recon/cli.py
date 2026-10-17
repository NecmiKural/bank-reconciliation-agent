"""Command line entry point: `recon <command>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recon import db, rules, score, synth

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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
