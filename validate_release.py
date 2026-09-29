#!/usr/bin/env python3
"""Static, standard-library validation for the E1--E6 release collection."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = tuple(f"E{i}" for i in range(1, 7))
ENTRYPOINTS = {
    "E1": ("reproduce.py",),
    "E2": ("reproduce.py",),
    "E3": ("reproduce.py",),
    "E4": ("run.py",),
    "E5": ("reproduce.py",),
    "E6": ("plot_figures.py", "run_experiment.py"),
}
FORBIDDEN_DIRS = {".ipynb_checkpoints", ".venv"}
RUNTIME_CACHE_DIRS = {"__pycache__", ".pytest_cache"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".log", ".swp"}
FORBIDDEN_PATHS = {
    Path("E2/results"),
    Path("E3/results"),
    Path("E4/outputs"),
    Path("E5/results"),
}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".sh", ".bat", ".csv",
}
SENSITIVE_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
    "credential assignment": re.compile(
        r"\b(?:password|passwd|pwd|api[_-]?key|token)\b\s*[:=]\s*['\"][^'\"]{4,}['\"]",
        re.IGNORECASE,
    ),
    "server endpoint": re.compile(
        r"\b(?:ssh|server|host|hostname)\b[^\n]{0,40}\b(?:\d{1,3}\.){3}\d{1,3}\b",
        re.IGNORECASE,
    ),
    "personal home path": re.compile(
        r"(?:/home/[A-Za-z0-9._-]+|C:\\Users\\[A-Za-z0-9._-]+)",
        re.IGNORECASE,
    ),
}


def validate(root: Path = ROOT) -> dict[str, Any]:
    errors: list[str] = []
    found = sorted(path.name for path in root.iterdir() if path.is_dir() and re.fullmatch(r"E\d+", path.name))
    if found != list(EXPERIMENTS):
        errors.append(f"expected experiment directories {EXPERIMENTS}, found {found}")

    for name in EXPERIMENTS:
        directory = root / name
        for required in ("README.md", "requirements.txt", "LICENSE", *ENTRYPOINTS[name]):
            if not (directory / required).is_file():
                errors.append(f"missing {name}/{required}")

    for relative in FORBIDDEN_PATHS:
        if (root / relative).exists():
            errors.append(f"generated directory must not ship: {relative.as_posix()}")

    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in RUNTIME_CACHE_DIRS for part in relative.parts):
            continue
        if path.is_dir() and path.name in FORBIDDEN_DIRS:
            errors.append(f"forbidden cache/environment directory: {relative.as_posix()}")
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden generated file: {relative.as_posix()}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append(f"cannot read {relative.as_posix()}: {exc}")
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} found in {relative.as_posix()}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "experiments": list(EXPERIMENTS),
        "errors": errors,
    }


def main() -> None:
    report = validate()
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

