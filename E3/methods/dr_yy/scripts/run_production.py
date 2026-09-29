from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dryy.config import load_config


def task_path(root: Path, experiment: str, method: str, geometry: str, repeat: int) -> Path:
    return root / "packets" / f"{experiment}_{method.lower()}_{geometry}_r{repeat:02d}.json.gz"


def run_one(task, output_root: Path, overwrite: bool) -> tuple[str, int, str]:
    experiment, method, geometry, repeat = task
    identifier = f"{experiment}-{method}-{geometry}-r{repeat:02d}"
    destination = task_path(output_root, experiment, method, geometry, repeat)
    if destination.exists() and not overwrite:
        return identifier, 0, "skipped-existing"
    command = [
        sys.executable, str(ROOT / "scripts" / "run_parent.py"),
        "--experiment", experiment, "--method", method, "--geometry", geometry,
        "--repeat", str(repeat), "--output-root", str(output_root),
    ]
    if overwrite:
        command.append("--overwrite")
    environment = os.environ.copy()
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    message = (completed.stdout + completed.stderr).strip()
    return identifier, int(completed.returncode), message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e2", "e3"), required=True)
    parser.add_argument("--methods", choices=("DR", "YY"), nargs="+", default=("DR", "YY"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 48:
        raise ValueError("workers must be in [1,48]")
    config, _ = load_config(args.experiment)
    tasks = [
        (args.experiment, method, row["key"], repeat)
        for method in args.methods
        for row in config["geometries"]
        for repeat in range(config["repeats"])
    ]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, task, args.output_root, args.overwrite): task for task in tasks}
        for future in as_completed(futures):
            identifier, code, message = future.result()
            print(f"{identifier}: {'PASS' if code == 0 else 'FAIL'} {message}", flush=True)
            if code:
                failures.append((identifier, code, message))
    if failures:
        raise SystemExit(f"{len(failures)} parent tasks failed")


if __name__ == "__main__":
    main()

