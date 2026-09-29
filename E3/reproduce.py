#!/usr/bin/env python3
"""Single entry point for the fixed-n E3 release."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DESIGN = json.loads((ROOT / "config" / "e3_n12000.json").read_text(encoding="utf-8"))
RESULTS = ROOT / "results"


def run(command: list[str], cwd: Path = ROOT, env: dict | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def environment() -> dict:
    value = os.environ.copy()
    value.update({
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    return value


def validate_release() -> None:
    if DESIGN["n"] != 12000:
        raise RuntimeError("the release contract requires n=12000")
    run([sys.executable, "-m", "pytest", "-q", "tests"])
    run([sys.executable, "-m", "pytest", "-q", "tests"], ROOT / "methods" / "stsp")
    run([sys.executable, "-m", "pytest", "-q", "tests"], ROOT / "methods" / "dr_yy")


def smoke(methods: set[str]) -> None:
    output = RESULTS / "smoke"
    env = environment()
    if "stsp" in methods:
        run([
            sys.executable, "scripts/run_parent.py", "--geometry", "circle_r2",
            "--repeat", "0", "--mode", "smoke", "--bandwidths", "0.2", "1.0", "5.0",
            "--query-count", "2", "--truth-count", "64",
            "--result-root", str(output / "stsp"), "--resume",
        ], ROOT / "methods" / "stsp", env)
    for method in sorted(methods & {"dr", "yy"}):
        scales = ["0.2", "1.0", "5.0"] if method == "dr" else [
            str(0.2 / DESIGN["sigma"]), str(1.0 / DESIGN["sigma"]), str(5.0 / DESIGN["sigma"])
        ]
        run([
            sys.executable, "scripts/run_parent.py", "--experiment", "e3",
            "--method", method.upper(), "--geometry", "circle_r2", "--repeat", "0",
            "--output-root", str(output / "dr_yy"), "--query-count", "2",
            "--e3-scales", *scales, "--overwrite",
        ], ROOT / "methods" / "dr_yy", env)


def stsp_parent(geometry: str, repeat: int, workers_root: Path, env: dict) -> tuple[str, int]:
    command = [
        sys.executable, "scripts/run_parent.py", "--geometry", geometry,
        "--repeat", str(repeat), "--mode", "production", "--result-root", str(workers_root),
        "--resume", "--production-confirmed",
    ]
    result = subprocess.run(command, cwd=ROOT / "methods" / "stsp", env=env, check=False)
    return f"{geometry}:r{repeat:02d}", result.returncode


def formal_stsp(workers: int) -> None:
    output = RESULTS / "formal" / "stsp"
    env = environment()
    tasks = [(geometry, repeat) for geometry in DESIGN["geometries"] for repeat in range(DESIGN["repeats"])]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(stsp_parent, geometry, repeat, output, env) for geometry, repeat in tasks]
        for future in as_completed(futures):
            label, code = future.result()
            print(f"STSP {label}: {'PASS' if code == 0 else 'FAIL'}", flush=True)
            if code:
                failures.append(label)
    if failures:
        raise RuntimeError(f"STSP failed parents: {failures}")
    run([
        sys.executable, "scripts/audit_results.py", "--result-root", str(output),
        "--output", str(output / "audit.json"),
    ], ROOT / "methods" / "stsp", env)


def formal_dr_yy(methods: set[str], gpu_workers: int) -> None:
    project = ROOT / "methods" / "dr_yy"
    output_base = RESULTS / "formal" / "dr_yy"
    env = environment()
    run([
        sys.executable, "gpu_backend/equivalence_gate.py", "--output",
        str(output_base / "cpu_gpu_equivalence.json"),
    ], project, env)
    jobs = [f"e3:{name.upper()}" for name in ("dr", "yy") if name in methods]
    run([
        sys.executable, "gpu_backend/run_production_gpu.py", "--jobs", *jobs,
        "--output-base", str(output_base), "--workers", str(gpu_workers),
    ], project, env)
    if methods >= {"dr", "yy"}:
        result_root = output_base / "e3"
        run([
            sys.executable, "scripts/audit_results.py", "--experiment", "e3",
            "--result-root", str(result_root), "--output", str(result_root / "audit.json"),
        ], project, env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce E3 at the single frozen sample size.")
    parser.add_argument("--mode", choices=("smoke", "formal", "plot"), default="smoke")
    parser.add_argument("--methods", nargs="+", choices=("all", "stsp", "dr", "yy"), default=["all"])
    parser.add_argument("--workers", type=int, default=4, help="CPU workers for STSP (1--48)")
    parser.add_argument("--gpu-workers", type=int, default=4, help="GPU parent workers for DR/YY (1--8)")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    methods = {"stsp", "dr", "yy"} if "all" in args.methods else set(args.methods)
    if not 1 <= args.workers <= 48 or not 1 <= args.gpu_workers <= 8:
        raise ValueError("worker counts are outside the documented range")
    if not args.skip_tests and args.mode != "plot":
        validate_release()
    if args.mode == "smoke":
        smoke(methods)
        return
    if args.mode == "formal":
        if "stsp" in methods:
            formal_stsp(args.workers)
        if methods & {"dr", "yy"}:
            formal_dr_yy(methods, args.gpu_workers)
        if methods == {"stsp", "dr", "yy"}:
            run([sys.executable, "scripts/plot_results.py"])
        return
    run([sys.executable, "scripts/plot_results.py"])


if __name__ == "__main__":
    main()
