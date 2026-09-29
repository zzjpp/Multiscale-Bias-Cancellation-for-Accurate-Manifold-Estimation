from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEED = int(json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["seed"])
random.seed(SEED)
np.random.seed(SEED)
sys.path.insert(0, str(ROOT / "src"))

from e1_simulation.plot_population import reproduce  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce all six audited E1 population figures.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    report = reproduce(
        ROOT / "data" / "e1_population_errors.csv",
        ROOT / "data" / "provenance.json",
        args.output_dir.resolve(),
    )
    print(json.dumps({"status": report["status"], "figures": sorted(report["output_sha256"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

