from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEED = int(json.loads((ROOT / "config.json").read_text(encoding="utf-8"))["seed"])
random.seed(SEED)
np.random.seed(SEED)
sys.path.insert(0, str(ROOT / "src"))

from e1_simulation import plot_sigma_path  # noqa: E402
from circle_triptych import render as render_circle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the audited E1 sigma-path STSP, DR, and YY figures."
    )
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--all-diagnostics", action="store_true", help="Also draw the original slope-window/scaled six-panel diagnostics")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    scope = dict(plot_sigma_path.reproduce.__globals__)
    if not args.all_diagnostics:
        # Output selection only. All original input and numerical-evidence
        # validators still run; estimator arithmetic is not changed.
        scope["render_scaled_errors"] = lambda *args: []
        scope["render_slope_stability"] = lambda *args: []
    draw = types.FunctionType(plot_sigma_path.reproduce.__code__, scope)
    report = draw(ROOT / "data", ROOT / "data" / "provenance.json", output)
    for path in render_circle(ROOT, output):
        report["output_sha256"][path.name] = plot_sigma_path.sha256(path)
    report["paper62_figures"] = [2,3,8,9,10]
    (output / "FIGURE_MANIFEST.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"status": report["status"], "figures": sorted(report["output_sha256"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

