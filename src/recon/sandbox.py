"""Run model-written Python in a throwaway container with no network and no credentials.

The container holds Python and sandbox/runner.py only. It reaches the tools solely through
calls it writes to stdout, which this module hands to `on_call` (the bridge) one at a time.
Every line read from the container is treated as untrusted: it must be short, valid JSON of
an expected shape, or the cell ends as a protocol violation. A cell that runs past its
deadline is killed.

NOTE: runc containers share the host kernel; seccomp, dropped capabilities, a read-only root,
no network and an unprivileged user narrow that, but do not remove it. gVisor
(`--runtime=runsc`) is the upgrade path where real untrusted data is handled; every dataset
here is synthetic.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

IMAGE = "recon-sandbox:1"  # `make sandbox-image`; bump the tag when runner.py's protocol changes
NAME_PREFIX = "recon-sbx-"
DEADLINE_S = 90.0
START_TIMEOUT_S = 30.0
KEEP = 8 * 1024
MAX_LINE = 64 * 1024
# What the docker client needs to reach the daemon, and nothing else from this environment.
ENV_KEYS = ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT")


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class CellResult:
    stdout: str
    error: str | None
    truncated: bool
    killed: bool
    exec_ms: float


def docker_argv(name: str, image: str = IMAGE) -> list[str]:
    return [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "--pids-limit", "64", "--memory", "256m", "--memory-swap", "256m", "--cpus", "1",
        "--ulimit", "nofile=64:64",
        "--log-driver", "none",
        image,
    ]  # fmt: skip


def docker_env(environ: Mapping[str, str] = os.environ) -> dict[str, str]:
    return {key: environ[key] for key in ENV_KEYS if key in environ}


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], env=docker_env(), capture_output=True, text=True, timeout=30
    )


def cleanup() -> None:
    """Kill sandbox containers left by an earlier process that died mid-cell."""
    listed = _docker("ps", "-q", "--filter", f"name={NAME_PREFIX}")
    for container in listed.stdout.split():
        _docker("kill", container)


def _shape_ok(message: Any) -> bool:
    if not isinstance(message, dict):
        return False
    kind = message.get("type")
    if kind == "call":
        return (
            isinstance(message.get("id"), int)
            and isinstance(message.get("method"), str)
            and len(message["method"]) <= 64
            and isinstance(message.get("params"), dict)
        )
    if kind == "done":
        return (
            isinstance(message.get("stdout"), str)
            and isinstance(message.get("error"), str | None)
            and isinstance(message.get("truncated"), bool)
        )
    return kind == "ready"


class Sandbox:
    """One container for one statement: globals persist between its cells."""

    def __init__(self, image: str = IMAGE, deadline_s: float = DEADLINE_S) -> None:
        self.image, self.deadline_s = image, deadline_s
        self.name = NAME_PREFIX + uuid.uuid4().hex[:12]
        self.start_ms = 0.0
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue[bytes | None] = queue.Queue()

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self, tools: list[str]) -> None:
        began = time.perf_counter()
        self._proc = subprocess.Popen(
            docker_argv(self.name, self.image),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=docker_env(),
        )
        threading.Thread(target=self._read, daemon=True).start()
        self._send({"type": "init", "tools": list(tools)})
        message = self._next(time.monotonic() + START_TIMEOUT_S)
        if not (isinstance(message, dict) and message.get("type") == "ready"):
            self.close()
            raise SandboxError(f"the sandbox did not start; is {self.image} built?")
        self.start_ms = round((time.perf_counter() - began) * 1000, 1)

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for raw in iter(lambda: self._proc.stdout.readline(MAX_LINE + 1), b""):
            self._lines.put(raw)
        self._lines.put(None)

    def _send(self, message: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write((json.dumps(message) + "\n").encode())
        self._proc.stdin.flush()

    def _next(self, deadline: float) -> Any:
        """The next message, None at the deadline or when the container ended, or the string
        "invalid" for a line that is too long or not a known message."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            raw = self._lines.get(timeout=remaining)
        except queue.Empty:
            return None
        if raw is None:
            return None
        if len(raw) > MAX_LINE or not raw.endswith(b"\n"):
            return "invalid"
        try:
            message = json.loads(raw)
        except (ValueError, RecursionError):  # deep nesting raises RecursionError
            return "invalid"
        return message if _shape_ok(message) else "invalid"

    def run(self, code: str, on_call: Callable[[str, dict[str, Any]], str]) -> CellResult:
        """Run one cell; each tool call it makes goes to on_call, whose string is returned
        to the code. A cell that overruns, breaks the protocol or dies kills the sandbox."""
        began = time.perf_counter()
        deadline = time.monotonic() + self.deadline_s

        def failed(error: str) -> CellResult:
            self._kill()
            elapsed = round((time.perf_counter() - began) * 1000, 1)
            return CellResult("", error, False, True, elapsed)

        if self._proc is None or self._proc.poll() is not None:
            return CellResult("", "the sandbox is not running", False, True, 0.0)
        try:
            self._send({"type": "cell", "code": code})
        except OSError:
            return failed("the sandbox stopped")
        while True:
            message = self._next(deadline)
            if message is None:
                ended = self._proc.poll() is not None or time.monotonic() < deadline
                return failed("the sandbox stopped" if ended else "deadline exceeded")
            if message == "invalid" or message["type"] == "ready":
                return failed("protocol violation")
            if message["type"] == "done":
                elapsed = round((time.perf_counter() - began) * 1000, 1)
                stdout = message["stdout"][:KEEP]
                truncated = message["truncated"] or len(message["stdout"]) > KEEP
                return CellResult(stdout, message["error"], truncated, False, elapsed)
            try:
                result = on_call(message["method"], message["params"])
                reply = {"type": "result", "id": message["id"], "result": result}
            except Exception as exc:  # on_call is the bridge, which should not raise
                reply = {"type": "error", "id": message["id"], "error": type(exc).__name__}
            try:
                self._send(reply)
            except OSError:
                return failed("the sandbox stopped")

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            _docker("kill", self.name)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()  # the runner exits at end of input
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self._kill()
        self._proc = None
