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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", nargs="+", required=True, help="Entries such as e2:DR or e3:YY")
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8] for a single GPU")

    tasks = []
    for job in args.jobs:
        experiment, method = job.split(":", 1)
        experiment = experiment.lower()
        method = method.upper()
        config, _ = load_config(experiment)
        for geometry in config["geometries"]:
            for repeat in range(int(config["repeats"])):
                tasks.append((experiment, method, geometry["key"], repeat))

    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })

    def run_one(index_task):
        index, (experiment, method, geometry, repeat) = index_task
        output_root = args.output_base / experiment
        destination = output_root / "packets" / f"{experiment}_{method.lower()}_{geometry}_r{repeat:02d}.json.gz"
        identifier = f"{experiment}-{method}-{geometry}-r{repeat:02d}"
        if destination.exists():
            return index, identifier, 0, "existing", "SKIP"
        command = [
            sys.executable, str(ROOT / "gpu_backend" / "run_parent_gpu.py"),
            "--experiment", experiment, "--method", method, "--geometry", geometry,
            "--repeat", str(repeat), "--output-root", str(output_root),
        ]
        completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
        message = (completed.stdout + completed.stderr).strip().replace("\n", " | ")
        status = "PASS" if completed.returncode == 0 else "FAIL"
        return index, identifier, int(completed.returncode), message, status

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, item) for item in enumerate(tasks, start=1)]
        for future in as_completed(futures):
            index, identifier, code, message, status = future.result()
            print(f"[{index}/{len(tasks)}] {status} {identifier} {message}", flush=True)
            if code:
                failures.append((identifier, code, message))
    if failures:
        print(f"FAILED_PARENT_COUNT={len(failures)}", flush=True)
        raise SystemExit(1)
    print(f"ALL_PARENTS_COMPLETE={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
