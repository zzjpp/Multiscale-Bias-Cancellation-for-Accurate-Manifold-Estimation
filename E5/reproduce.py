#!/usr/bin/env python3
"""Single entry point for the two-figure E5 release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(ROOT / "src"))

from e56v2.experiment import audit, finite_cell, finite_seed, population_cell, run_formal


def load(name: str) -> dict:
    return json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))


def source_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += sorted((ROOT / "scripts").glob("*.py"))
    paths.append(Path(__file__))
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_command(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    environment = os.environ.copy()
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def smoke(population: dict, finite: dict) -> None:
    reduced_numerical = {"latent_count": 128, "radial_order": 32, "longitudinal_order": 32}
    population_result = population_cell(
        population, population["sigma_grid"][0], query_count=2,
        numerical=reduced_numerical,
    )
    finite_result = finite_cell(
        finite, finite["sample_sizes"][0], 0,
        finite_seed(finite, finite["sample_sizes"][0], 0), query_count=2,
    )
    report = {
        "status": "PASS" if (
            population_result["retained_count"] > 0
            and finite_result["retained_count"] > 0
            and set(population_result["summaries"]) == set(finite_result["summaries"])
        ) else "FAIL",
        "population": {
            "retained": population_result["retained_count"],
            "failed": population_result["failure_count"],
        },
        "finite_sample": {
            "retained": finite_result["retained_count"],
            "failed": finite_result["failure_count"],
            "evidence_class": "smoke_only",
            "full_paper_protocol_audit_required": bool(finite["formal_evidence"]),
        },
    }
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce exactly the two released E5 figures.")
    parser.add_argument("--mode", choices=("smoke", "formal", "plot"), default="smoke")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 48:
        raise ValueError("workers must be in [1,48]")
    population = load("population.json")
    finite = load("finite_sample.json")
    if not args.skip_tests and args.mode != "plot":
        run_command([sys.executable, "-m", "pytest", "-q", "tests"])
    if args.mode == "smoke":
        smoke(population, finite)
        return
    results = ROOT / "results"
    if args.mode == "formal":
        run_report = run_formal(results, population, finite, source_hash(), args.workers)
        print(json.dumps(run_report, indent=2))
        if run_report["status"] != "PASS":
            raise SystemExit(1)
        audit_report = audit(results, population, finite, source_hash())
        print(json.dumps(audit_report, indent=2))
        if audit_report["status"] != "PASS":
            raise SystemExit(1)
    run_command([sys.executable, "scripts/plot_two_figures.py", "--results", str(results)])


if __name__ == "__main__":
    main()
