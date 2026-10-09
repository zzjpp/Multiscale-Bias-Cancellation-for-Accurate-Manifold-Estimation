from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from e1_theory_path.experiment import aggregate, run_all  # noqa: E402
from e1_theory_path.plotting import render  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the E1 joint theory-path population order experiment.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        print(json.dumps(render(ROOT / "results"), indent=2))
        return 0
    config, packets = run_all(ROOT, args.workers)
    audit, _, slopes, _ = aggregate(ROOT, config, packets)
    figures = render(ROOT / "results") if audit["status"] == "PASS" else {"status": "SKIPPED_AUDIT_FAIL"}
    print(json.dumps({"audit": audit["status"], "figures": figures["status"], "slope_rows": len(slopes)}, indent=2))
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
