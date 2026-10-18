"""Render the offline demo to docs/demo.svg for the README.

Run with the database up: `uv run python scripts/record_demo.py`. It replays the committed
40-line slice through the rules pass, the agent and the shadow policy, exactly as
`make demo` does, and needs no model key.
"""

import contextlib
import io
import json
from pathlib import Path

from rich.console import Console
from rich.text import Text

from recon import cli

ROOT = Path(__file__).resolve().parents[1]
COMMAND = [
    "run",
    "--business",
    "A",
    "--limit",
    "40",
    "--mode",
    "replay",
    "--cassette",
    str(ROOT / "tests" / "cassettes" / "A-slice.jsonl"),
    "--workers",
    "4",
]


def main() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli.main(COMMAND)
    text = out.getvalue()
    lines = [line for line in text.splitlines() if line.startswith(("ok ", "BAD"))]
    summary = json.loads(text[text.index("\n{") + 1 : text.rindex("}") + 1])

    console = Console(record=True, width=150, file=io.StringIO())
    console.print(Text("$ make demo", style="bold"))
    console.print(
        Text("  replays the committed 40-line slice: rules pass, agent, shadow policy", "dim")
    )
    for line in lines:
        console.print(Text(line, style="green" if line.startswith("ok ") else "bold red"))
    agent = summary["agent"]
    attacks = summary["attacks"]["model_decided"]["all"]
    console.print(
        Text(
            f"agent lines {agent['correct']} of {agent['n']} correct, "
            f"wrong matches {summary['overall']['wrong_matches']}, "
            f"injected lines that succeeded {attacks['success']}, "
            f"proposals waiting for review in the queue",
            style="bold",
        )
    )
    path = ROOT / "docs" / "demo.svg"
    console.save_svg(str(path), title="recon: offline demo")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
