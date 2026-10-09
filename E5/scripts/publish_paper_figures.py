#!/usr/bin/env python3
"""Publish and verify the exact figure sources used by the paper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "paper_figures"
OUTPUT = ROOT / "results" / "figures"
PROVENANCE = json.loads((ROOT / "data" / "PROVENANCE.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema": "E5_PAPER_FIGURE_MANIFEST_V1",
        "status": "PASS",
        "figures": [],
    }
    expected = PROVENANCE["reference_files"]
    observed = {path.name for path in REFERENCE.iterdir() if path.is_file()}
    if set(expected) != observed:
        raise RuntimeError("paper_figures does not match the frozen provenance inventory")
    for name, expected_hash in sorted(expected.items()):
        source = REFERENCE / name
        if sha256(source) != expected_hash:
            raise RuntimeError(f"reference hash mismatch: {name}")
        destination = OUTPUT / name
        shutil.copy2(source, destination)
        copied_hash = sha256(destination)
        if copied_hash != expected_hash:
            raise RuntimeError(f"published hash mismatch: {name}")
        manifest["figures"].append({"file": name, "sha256": copied_hash})
    (OUTPUT / "FIGURE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"published and verified {len(expected)} paper figure files in {OUTPUT}")


if __name__ == "__main__":
    main()