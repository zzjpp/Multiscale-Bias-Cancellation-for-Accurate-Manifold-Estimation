"""Generate inputs, fit, evaluate and plot NEW manuscript experiments.

No saved fitted points, tables or figure exports are read by this entrypoint.
The bundled E3 query/reference coordinates are fixed design inputs, not fits.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from manifold_experiments.protocol import EXPERIMENTS, ROOT, plan
from manifold_experiments.catalog import (
    SELECTIONS,
    describe_experiments,
    resolve_selections,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment",
        nargs="+",
        choices=SELECTIONS,
        default=["all"],
        help="Paper topic or compatibility code E1–E5. Use --list for the paper mapping.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List paper headings and implementation directories; no fitting.",
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--profile", choices=("paper", "smoke"), default="paper")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Separate interpreters; default one. Set CUDA_VISIBLE_DEVICES externally for GPU.",
    )
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete, source-bound packets; failed tasks are NOT retried.",
    )
    p.add_argument("--evaluate-only", action="store_true")
    a = p.parse_args()
    if a.list:
        print(describe_experiments())
        return
    if a.output is None:
        p.error("--output is required unless --list is used")
    if not 1 <= a.workers <= 48:
        p.error("--workers must be 1..48")
    if a.profile == "smoke" and a.backend != "cpu":
        p.error("Smoke uses the reference CPU implementation explicitly")
    experiments = resolve_selections(a.experiment)
    target = a.output.resolve()
    if target == ROOT or ROOT.is_relative_to(target):
        p.error("Output cannot be the repository or its parent")
    from release_contract import run_source_digest

    source = run_source_digest(verify=True)
    design = plan(experiments, a.profile, a.backend)
    design["source_manifest_sha256"] = source
    if target.exists():
        if not (a.resume or a.evaluate_only):
            raise FileExistsError("Choose a fresh output or explicit --resume")
        if json.loads((target / "PLAN.json").read_bytes()) != design:
            raise ValueError("Resume plan/source mismatch")
    else:
        target.mkdir(parents=True)
        save(target / "PLAN.json", design)
    if a.plan_only:
        print(
            json.dumps(
                dict(
                    plan=str(target / "PLAN.json"),
                    tasks=len(design["jobs"]),
                    new_fits=False,
                )
            )
        )
        return
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
        MPLBACKEND="Agg",
    )

    def execute(job):
        result = target / "packets" / (job["id"] + ".json.gz")
        failed = target / "errors" / (job["id"] + ".json")
        if failed.exists():
            raise RuntimeError(
                "Recorded failed task requires investigation; not silently retried: "
                + str(failed)
            )
        if result.exists():
            with gzip.open(result, "rt", encoding="utf8") as f:
                record = json.load(f)
            if record["job"] != job or record["source_manifest_sha256"] != source:
                raise ValueError("Invalid existing packet")
            return
        log = target / "logs" / (job["id"] + ".log")
        log.parent.mkdir(exist_ok=True)
        with log.open("x", encoding="utf8") as handle:
            done = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-B",
                    str(ROOT / "manifold_experiments/worker.py"),
                    "--run",
                    str(target),
                    "--job",
                    job["id"],
                ],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if done.returncode:
            raise RuntimeError("Task failed; inspect " + str(log))
        print(
            json.dumps(dict(completed=job["id"], experiment=job["experiment"])),
            flush=True,
        )

    def finish():
        if not a.evaluate_only:
            failures = []
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                for task in as_completed(
                    [pool.submit(execute, j) for j in design["jobs"]]
                ):
                    try:
                        task.result()
                    except Exception as error:
                        failures.append(str(error))
                        print(str(error), file=sys.stderr, flush=True)
            if failures:
                save(
                    target / "RUN_FAILED.json",
                    dict(failures=failures, outputs_preserved=True),
                )
                raise SystemExit(1)
        subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "-B",
                str(ROOT / "manifold_experiments/evaluate.py"),
                "--run",
                str(target),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )

    lock = target / "RUNNING.json"
    with lock.open("x", encoding="utf8") as f:
        json.dump(dict(pid=os.getpid(), source=source), f)
    try:
        finish()
    finally:
        if json.loads(lock.read_bytes())["pid"] != os.getpid():
            raise RuntimeError("Run ownership changed; lock retained")
        lock.unlink()


if __name__ == "__main__":
    main()
