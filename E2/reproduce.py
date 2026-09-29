#!/usr/bin/env python3
"""One-command entry point for the six released E2 figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from e2plots import load_release_rows, load_style, render_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Render exactly six E2 figures.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results",
        help="Output directory (default: E2/results).",
    )
    args = parser.parse_args()

    # The release plots fixed summaries and contains no random computation.
    # These seeds freeze any future incidental use of Python/NumPy RNGs.
    random.seed(0)
    np.random.seed(0)

    style = load_style(ROOT)
    rows = load_release_rows(ROOT, style)
    products = render_all(rows, style, args.output.resolve())
    report = {
        "status": "PASS",
        "logical_figures": 6,
        "exports": len(products),
        "output": str(args.output.resolve()),
        "files": [path.name for path in products],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

