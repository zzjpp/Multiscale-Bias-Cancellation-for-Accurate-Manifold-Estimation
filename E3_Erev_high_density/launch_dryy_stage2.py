from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--dr-base", type=Path, required=True)
    parser.add_argument("--yy-base", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    required = {
        (row["method"], row["geometry"])
        for row in audit["stage2_decisions"]
        if row["stage2_required"] and row["method"] in {"DR", "YY"}
    }
    logs = args.result_root / "stage2" / "launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    semaphores = {
        0: threading.Semaphore(args.workers_per_gpu),
        1: threading.Semaphore(args.workers_per_gpu),
    }
    tasks = []
    task_number = 0
    for method, geometry in sorted(required):
        for shard in range(args.shard_count):
            tasks.append((task_number % 2, method, geometry, shard))
            task_number += 1

    def run(task):
        gpu, method, geometry, shard = task
        base = args.dr_base if method == "DR" else args.yy_base
        log = logs / f"{method.lower()}_{geometry}_s{shard:02d}.log"
        command = [
            sys.executable, str(Path(__file__).with_name("run_dryy_increment.py")),
            "--method", method, "--geometry", geometry, "--stage", "2",
            "--base-packet-root", str(base), "--output-root", str(args.result_root),
            "--device", str(gpu), "--shard-index", str(shard),
            "--shard-count", str(args.shard_count),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        with semaphores[gpu], log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        return task, completed.returncode, log

    failures = []
    with ThreadPoolExecutor(max_workers=2 * args.workers_per_gpu) as executor:
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
