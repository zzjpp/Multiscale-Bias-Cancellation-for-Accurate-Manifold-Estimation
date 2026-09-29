from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    geometries = sorted({
        row["geometry"]
        for row in audit["stage2_decisions"]
        if row["stage2_required"] and row["method"] == "STSP"
    })
    logs = args.result_root / "stage2" / "stsp_launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    tasks = [(geometry, shard) for geometry in geometries for shard in range(args.shard_count)]

    def run(task):
        geometry, shard = task
        log = logs / f"stsp_{geometry}_s{shard:02d}.log"
        packet = (
            args.result_root / "stage2" / "stsp_shards" /
            f"s{shard:02d}of{args.shard_count:02d}" / "packets" /
            f"e3_stsp_{geometry}_r00.json.gz"
        )
        if packet.exists():
            return task, 0, packet
        command = [
            sys.executable, str(Path(__file__).with_name("run_stsp_stage.py")),
            "--geometry", geometry, "--stage", "2",
            "--output-root", str(args.result_root),
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
