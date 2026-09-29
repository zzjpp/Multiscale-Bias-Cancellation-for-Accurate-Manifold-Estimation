#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import _bootstrap  # noqa: F401

from e3.config import CONFIG, CONFIG_HASH, code_hash, result_root, source_hashes
from e3.geometry import BY_KEY
from e3.packets import read_packet
from e3.runner import parent_id, validate_parent


def audit(root: Path):
    expected = {(key, repeat) for key in BY_KEY for repeat in range(CONFIG["repeats"])}
    valid = []
    corrupt = []
    duplicates = []
    seen = set()
    for path in sorted((root / "packets").glob("*.production.json.gz")) if (root / "packets").exists() else []:
        try:
            packet = validate_parent(read_packet(path, CONFIG_HASH, code_hash()))
            key = (packet["geometry"], packet["repeat"])
            if key in seen:
                duplicates.append(str(path))
            else:
                seen.add(key)
                valid.append(str(path))
        except Exception as exc:
            corrupt.append({"path": str(path), "error": repr(exc)})
    leftovers = [str(path) for pattern in ("*.lock", "*.partial") for path in root.rglob(pattern)] if root.exists() else []
    return {"schema": "E3_RESULT_AUDIT_V1", "config_hash": CONFIG_HASH, "code_hash": code_hash(), "expected_parent_count": CONFIG["production"]["parent_count"], "expected_main_cell_count": CONFIG["production"]["main_cell_count"], "valid_parent_count": len(seen), "missing": [parent_id(*key) for key in sorted(expected - seen)], "duplicates": duplicates, "corrupt": corrupt, "temporary_or_lock_files": leftovers, "source_hashes": source_hashes(), "complete": seen == expected and not duplicates and not corrupt and not leftovers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit(result_root(args.result_root))
    text = json.dumps(report, indent=2, allow_nan=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if report["corrupt"] or report["duplicates"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
