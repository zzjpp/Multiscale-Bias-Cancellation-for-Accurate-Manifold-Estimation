from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from e1_yy_path.gpu_runner import run_partition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--partition", type=int, required=True)
    parser.add_argument("--partitions", type=int, default=2)
    args = parser.parse_args()
    run_partition(ROOT, args.device, args.partition, args.partitions)


if __name__ == "__main__":
    main()
