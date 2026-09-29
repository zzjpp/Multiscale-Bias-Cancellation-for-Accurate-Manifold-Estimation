from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys


GEOMETRIES = ("sphere2_r3", "clifford_torus_r4", "sphere3_r4", "sphere2_r10")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logs = args.result_root / "stage1" / "stsp_launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    tasks = [(geometry, shard) for geometry in GEOMETRIES for shard in range(args.shard_count)]

    def run(task):
        geometry, shard = task
        log = logs / f"stsp_{geometry}_s{shard:02d}.log"
        command = [
            sys.executable, str(Path(__file__).with_name("run_stsp_stage.py")),
            "--geometry", geometry, "--stage", "1", "--output-root", str(args.result_root),
            "--shard-index", str(shard), "--shard-count", str(args.shard_count),
        ]
        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        return task, completed.returncode, log

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            task, returncode, log = future.result()
            print(task, returncode, log, flush=True)
            if returncode:
                failures.append((task, str(log)))
    if failures:
        raise SystemExit(f"failed tasks: {failures}")


if __name__ == "__main__":
    main()
