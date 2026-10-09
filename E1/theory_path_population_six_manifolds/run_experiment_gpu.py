from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from e1_theory_path.gpu_experiment import run_gpu_partition  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one deterministic GPU partition of E1 theory-path production.")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--partition", type=int, required=True)
    parser.add_argument("--partitions", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.partition < args.partitions:
        parser.error("partition must satisfy 0 <= partition < partitions")
    run_gpu_partition(ROOT, args.device, args.partition, args.partitions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
