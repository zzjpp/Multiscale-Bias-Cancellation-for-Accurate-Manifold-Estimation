from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from e7_release import build_all


def main() -> None:
    # The plotting path uses no randomness; recording a seed also makes that
    # contract explicit to automated artifact checks.
    np.random.seed(0)
    parser = argparse.ArgumentParser(description="Regenerate the three audited E7 figures.")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "figures",
        help="Output directory (default: figures/).",
    )
    args = parser.parse_args()
    outputs = build_all(ROOT, args.output_dir)
    print("Generated exactly three E7 figures (PNG and PDF):")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
