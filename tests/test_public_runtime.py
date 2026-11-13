"""Protocol and failure/custody regressions; real end-to-end fitting is in CI."""

import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manifold_experiments.protocol import EXPERIMENTS, GEOMETRIES, jobs, plan


def command(*args):
    return subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-B",
            str(ROOT / "run_paper.py"),
            *map(str, args),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf8",
    )


def test_final_protocol_and_full_grid():
    from verify_paper import audit

    report = audit()
    assert report["status"] == "MATCH_PROTOCOL" and report["planned_tasks"] == 1356
    full = jobs(EXPERIMENTS)
    assert len(full) == len({r["id"] for r in full}) == 1356
    assert len(jobs(EXPERIMENTS, "smoke")) == 24


def test_reduced_jobs_cannot_be_mistaken_for_paper_jobs():
    a = plan(EXPERIMENTS, "smoke", "cpu")
    b = plan(EXPERIMENTS, "paper", "cpu")
    assert not a["paper_evidence"] and b["paper_evidence"]
    assert {x["id"] for x in a["jobs"]}.isdisjoint(x["id"] for x in b["jobs"])
    assert {x["method"] for x in jobs(("nonuniform_E2",))} == {"STSP"}


def test_runtime_contract_contains_no_historical_outputs():
    from release_contract import run_source_digest

    assert len(run_source_digest(verify=True)) == 64
    files = json.loads((ROOT / "RUN_SOURCE_MANIFEST.json").read_bytes())["files"]
    for name in files:
        assert not set(Path(name).parts).intersection(
            {
                "data",
                "analysis",
                "results",
                "outputs",
                "imports",
                "reference_values",
                "finite_sample",
                "paper_figures",
            }
        )
    assert (
        "experiments/bandwidth_robustness/assets/corrugated_torus_r3_reverse_reference.npy"
        in files
    )


def test_plan_only_does_no_fitting_and_refuses_changed_resume(tmp_path):
    out = tmp_path / "plan"
    done = command("--experiment", "E2", "--output", out, "--plan-only")
    assert done.returncode == 0, done.stderr
    assert set(p.name for p in out.iterdir()) == {"PLAN.json"}
    changed = command("--experiment", "E3", "--output", out, "--resume", "--plan-only")
    assert changed.returncode != 0
    assert set(p.name for p in out.iterdir()) == {"PLAN.json"}


def test_missing_packets_cannot_be_released_as_figures(tmp_path):
    out = tmp_path / "missing"
    assert command("--experiment", "E2", "--output", out, "--plan-only").returncode == 0
    failed = command("--experiment", "E2", "--output", out, "--evaluate-only")
    assert failed.returncode != 0
    assert not (out / "figures").exists() and not (out / "EVALUATION.json").exists()


def test_constructed_weight_constraints_from_real_coefficients():
    from manifold_experiments.worker import load

    module = load(
        Path("experiments/population_verification/weight_control/analyze.py"),
        "test_independent_stsp_coefficient",
    )
    c = np.array([module.independent_coefficient(1, 2, h, 0.01) for h in (0.03, 0.06)])
    w = np.array([c[1], -c[0]]) / (c[1] - c[0])
    assert abs(w.sum() - 1) < 1e-14 and abs(w @ c) < 1e-18
    assert abs(w[0] - 1.8845) < 1e-4
    assert abs(abs(np.array([4 / 3, -1 / 3]) @ c) / 0.01**2 - 0.4764) < 1e-4


def test_nonuniform_surface_density_and_seed_lineage():
    from manifold_experiments.worker import load

    s = load(
        Path("experiments/nonuniform/fixed_scales/sampling.py"),
        "test_original_nonuniform_sampler",
    )
    assert np.array_equal(
        s.factor("circle_r2", np.array([[1.0, 0.0], [-1.0, 0.0]])), [1.5, 0.5]
    )

    class Circle:
        key = "circle_r2"
        id = 1
        D = 2

        def sample_clean(self, rng, n):
            t = rng.uniform(0, 2 * np.pi, n)
            return np.column_stack((np.cos(t), np.sin(t)))

    c, n, y, lineage = s.generate(Circle(), "STSP", 0, 10000)
    assert np.allclose(y, c + 0.005 * n)
    assert 0.21 < c[:, 0].mean() < 0.29
    assert lineage["acceptance_seed"][-1] == 2 and lineage["noise_seed"][-1] == 1


def test_e5_actual_weights_and_pilot_caveat():
    sys.path.insert(0, str(ROOT / "experiments/yy_directional_ablation/src"))
    from e56v2.experiment import weights_for, finite_seed

    cfg = json.loads(
        (
            ROOT / "experiments/yy_directional_ablation/config/finite_sample.json"
        ).read_bytes()
    )
    w = weights_for(cfg)
    assert abs(np.abs(w["a"]).sum() - 1.771373) < 1e-5
    assert abs(np.abs(w["b"]).sum() - 13.320742) < 1e-5
    assert w["max_abs_residual"] < 1e-10
    assert (
        cfg["legacy_failed_gate"]["pass"] is False and cfg["formal_evidence"] is False
    )
    assert (
        len({finite_seed(cfg, n, r) for n in cfg["sample_sizes"] for r in range(20)})
        == 60
    )


def test_python_sources_parse():
    for p in ROOT.rglob("*.py"):
        if not set(p.relative_to(ROOT).parts).intersection({"tmp", ".venv", "venv"}):
            ast.parse(p.read_text("utf-8-sig"), filename=str(p))
