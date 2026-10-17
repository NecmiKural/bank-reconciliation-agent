"""Repository hygiene checks that run locally and in CI."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Built from code points so this file does not trip its own check.
FORBIDDEN = (chr(0x2013), chr(0x2014), "&" + "ndash;", "&" + "mdash;")


def find_dashes(text: str) -> list[int]:
    """Return the 1-based line numbers that contain an em or en dash."""
    return [n for n, line in enumerate(text.splitlines(), 1) if any(f in line for f in FORBIDDEN)]


def repo_text_files() -> list[Path]:
    """Tracked plus untracked, non-ignored files, skipping binaries."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    paths = [ROOT / p for p in out.decode().split("\0") if p]
    return [p for p in paths if p.is_file() and b"\0" not in p.read_bytes()[:8192]]


def test_detector_flags_each_forbidden_form():
    for dash in FORBIDDEN:
        assert find_dashes(f"fine\nbad {dash} here") == [2]
    assert find_dashes("a plain - hyphen is fine") == []


def test_no_em_or_en_dashes_in_repo():
    offenders = {
        str(p.relative_to(ROOT)): lines
        for p in repo_text_files()
        if (lines := find_dashes(p.read_text(encoding="utf-8", errors="ignore")))
    }
    assert not offenders, f"em or en dashes found (file: lines): {offenders}"
