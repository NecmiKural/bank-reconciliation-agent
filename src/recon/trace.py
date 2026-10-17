"""One JSONL row per model call and per tool call.

Field names follow the OpenTelemetry GenAI semantic conventions (gen_ai.*) where one exists;
project fields use the recon.* prefix. Rows carry no message text, only metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
