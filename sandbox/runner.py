"""Runs model-written Python for code mode, inside the sandbox image.

Standard library only, and nothing from recon: the image holds this file and Python. The host
talks to it over stdin and stdout, one JSON object per line:

  host -> runner  {"type": "init", "tools": ["search_candidates", ...]}
  runner -> host  {"type": "ready"}
  host -> runner  {"type": "cell", "code": "..."}
  runner -> host  {"type": "call", "id": 1, "method": "search_candidates", "params": {...}}
  host -> runner  {"type": "result", "id": 1, "result": "<json string>"}
                  or {"type": "error", "id": 1, "error": "..."}
  runner -> host  {"type": "done", "stdout": "...", "error": null, "truncated": false}

Each tool is an async function that takes one dict and returns a string, as in Claude's
programmatic tool calling; calls are served one at a time. Globals persist between cells.
The host treats every line it reads as untrusted, so code that forges a message gains nothing.
"""

import ast
import asyncio
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

KEEP = 8 * 1024  # characters of output returned per cell
ABORT = 1024 * 1024  # a cell that writes more is stopped

# The protocol keeps its own copy of the pipes; plain stdout and stderr go nowhere, so print()
# and os.write(1, ...) in a cell cannot reach the host.
_out = os.fdopen(os.dup(1), "w", buffering=1)
_in = os.fdopen(os.dup(0), "r")
_null = os.open(os.devnull, os.O_RDWR)
for fd in (0, 1, 2):
    os.dup2(_null, fd)


class OutputLimit(BaseException):
    """Raised past ABORT characters; a BaseException so `except Exception` cannot keep going."""


class ToolError(Exception):
    pass


class Capped(io.TextIOBase):
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.kept = 0
        self.written = 0
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.written += len(text)
        room = KEEP - self.kept
        if room > 0:
            self.parts.append(text[:room])
            self.kept += min(len(text), room)
        if len(text) > room:
            self.truncated = True
        if self.written > ABORT:
            raise OutputLimit
        return len(text)


def send(message: dict) -> None:
    _out.write(json.dumps(message) + "\n")


def receive() -> dict:
    line = _in.readline()
    if not line:
        sys.exit(0)  # the host closed the pipe
    return json.loads(line)


calls = 0


def stub(name: str):
    async def call(params: dict) -> str:
        global calls
        calls += 1
        number = calls
        send({"type": "call", "id": number, "method": name, "params": params})
        reply = receive()
        if reply.get("id") != number:
            raise RuntimeError("the host answered a different call")
        if reply.get("type") == "error":
            raise ToolError(reply.get("error", ""))
        return reply["result"]

    call.__name__ = name
    return call


def run_cell(code: str, scope: dict) -> None:
    out = Capped()
    error = None
    try:
        compiled = compile(code, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        with redirect_stdout(out), redirect_stderr(out):
            pending = eval(compiled, scope)
            if asyncio.iscoroutine(pending):
                asyncio.run(pending)
    except OutputLimit:
        error = f"output limit: a cell may print at most {ABORT} characters"
    except BaseException:  # the cell's own failure, reported to the model, not fatal here
        error = traceback.format_exc(limit=5)[-2000:]
    send({"type": "done", "stdout": "".join(out.parts), "error": error, "truncated": out.truncated})


def main() -> None:
    init = receive()
    scope: dict = {"__name__": "__cell__"}
    for name in init.get("tools", []):
        scope[name] = stub(name)
    send({"type": "ready"})
    while True:
        message = receive()
        if message.get("type") == "cell":
            run_cell(message.get("code", ""), scope)


if __name__ == "__main__":
    main()
