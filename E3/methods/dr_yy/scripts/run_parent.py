from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dryy.runner import run_parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e2", "e3"), required=True)
    parser.add_argument("--method", choices=("DR", "YY"), required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--e2-n", type=int, nargs="+")
    parser.add_argument("--e3-scales", type=float, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = run_parent(
        args.experiment, args.method, args.geometry, args.repeat, args.output_root,
        query_count=args.query_count, e2_n_values=args.e2_n, e3_scales=args.e3_scales,
        overwrite=args.overwrite,
    )
    print(path)


if __name__ == "__main__":
    main()

