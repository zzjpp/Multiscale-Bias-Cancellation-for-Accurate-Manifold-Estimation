"""E6 release entrypoint: figures, raw-result audit, or a NEW frozen fit run.

Figure replay never refits. Full fitting requires the original Linux numerical
environment and writes only a fresh named .reruns directory, never the bundled
formal results. No installation, network, deletion or automatic publication.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCIENCE = ROOT / "recompute_package"


def source_gate(root):
    science = root / "recompute_package"
    spec = importlib.util.spec_from_file_location("e6_release_source_gate", science / "audit_e6_revision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.sha(science / "run_e6_revision.py") != module.LAUNCHER_SHA:
        raise ValueError("Frozen E6 scientific launcher changed")
    import hashlib
    if hashlib.sha256(module.expected_source_manifest(science / "project/E6")).hexdigest() != module.SOURCE_SHA:
        raise ValueError("Frozen E6 scientific source changed")
    for name, expected in (("e6_previous_results.json", module.PREVIOUS_SHA),
                           ("e6_previous_audit.json", module.PREVIOUS_AUDIT_SHA)):
        if module.sha(science / "inputs" / name) != expected:
            raise ValueError("Original unchanged YY input changed")
    return module


def fit_environment():
    if not sys.platform.startswith("linux") or sys.version.split()[0] != "3.12.7":
        raise ValueError("Full fits require original Linux Python 3.12.7; use figure/audit mode on Windows")
    import numpy as np
    import scipy
    import matplotlib
    actual = {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    if actual != {"numpy": "2.1.3", "scipy": "1.18.1", "matplotlib": "3.11.2"}:
        raise ValueError("Use locked fitting dependencies, separate from figure-only requirements: " + repr(actual))
    return dict(actual, python="3.12.7", backend="cpu deterministic quadrature")


def recompute(root, run_name):
    environment = fit_environment()  # Refuse incompatible runtime before creating output.
    source_gate(root)
    if not run_name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_name):
        raise ValueError("Full recompute requires a fresh explicit --run-name")
    base = root / ".reruns"
    destination = base / run_name
    for item in (destination, *destination.parents):
        if item.is_symlink() or item.is_junction():
            raise ValueError("Linked recomputation destination refused")
    if destination.exists():
        raise ValueError("Existing recomputation must be inspected, not overwritten")
    source = root / "recompute_package"
    destination.mkdir(parents=True)
    shutil.copytree(source / "project/E6", destination / "project/E6",
                    ignore=shutil.ignore_patterns("results", "figures", "__pycache__", ".pytest_cache"))
    shutil.copytree(source / "inputs", destination / "inputs")
    for name in ("run_e6_revision.py", "audit_e6_revision.py"):
        shutil.copy2(source / name, destination / name)
    # No RNG is used in E6. Hash seed is recorded solely for repeatable process
    # bookkeeping; numerical quadrature, query grids and scientific source stay.
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    (destination / "INVOCATION.json").write_text(json.dumps({"environment": environment,
        "query_generation": "256 fixed deterministic queries; no random sampling",
        "threads": {k: env[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "PYTHONHASHSEED": "0", "bundled_formal_results_overwritten": False,
        "unchanged_YY_refitted": False}, indent=2) + "\n", encoding="utf-8")
    with (destination / "production.log").open("x", encoding="utf-8") as handle:
        process = subprocess.run([sys.executable, "-B", "-u", "run_e6_revision.py"],
                                 cwd=destination, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RuntimeError("Recomputation failed; all scratch output/logs retained: " + str(destination))
    return subprocess.run([sys.executable, "-B", str(destination / "audit_e6_revision.py"),
                           "--root", str(destination)], env=env).returncode


def main(argv=None, root=ROOT):
    root = Path(root).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("figures", "audit", "recompute"), default="figures")
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    if args.mode == "figures":
        return subprocess.run([sys.executable, "-B", str(root / "reproduce_figures.py")], env=env).returncode
    if args.mode == "audit":
        source_gate(root)
        return subprocess.run([sys.executable, "-B", str(root / "recompute_package/audit_e6_revision.py"),
                               "--root", str(root / "recompute_package")], env=env).returncode
    return recompute(root, args.run_name)


if __name__ == "__main__":
    raise SystemExit(main())
