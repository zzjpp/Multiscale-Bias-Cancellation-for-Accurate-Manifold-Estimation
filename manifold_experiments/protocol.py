"""Explicit paper jobs, separate from reduced integration/interface tests."""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
GEOMETRIES = (
    "circle_r2",
    "sphere2_r3",
    "clifford_torus_r4",
    "sphere3_r4",
    "sphere2_r10",
    "corrugated_torus_r3",
)
SIGMAS = (0.008, 0.01, 0.0125, 0.016, 0.02, 0.025, 0.03)
SIZES = (2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000)
EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "nonuniform_E1", "nonuniform_E2")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def jobs(experiments, profile="paper"):
    result = []

    def add(exp, **kw):
        item = dict(experiment=exp, profile=profile, **kw)
        item["id"] = digest(item)[:24]
        result.append(item)

    smoke = profile == "smoke"
    geometries = GEOMETRIES[:1] if smoke else GEOMETRIES
    sigmas = (0.03,) if smoke else SIGMAS
    for exp in experiments:
        if exp in ("E1", "nonuniform_E1"):
            methods = ("STSP", "DR", "YY", "MS-STSP") if exp == "E1" else ("STSP",)
            for m in methods:
                for g in geometries:
                    for s in sigmas:
                        for resolution in ("production", "reference"):
                            add(
                                exp,
                                method=m,
                                geometry=g,
                                sigma=s,
                                resolution=resolution,
                            )
        elif exp in ("E2", "nonuniform_E2"):
            for m in (("STSP", "DR", "YY") if exp == "E2" else ("STSP",)):
                for g in geometries:
                    for r in (range(1) if smoke else range(20)):
                        add(exp, method=m, geometry=g, repeat=r)
        elif exp == "E3":
            for m in ("STSP", "DR", "YY"):
                for g in geometries:
                    for phase in ("main", "rev"):
                        for r in (range(1) if smoke or phase == "rev" else range(20)):
                            add(exp, method=m, geometry=g, phase=phase, repeat=r)
        elif exp == "E4":
            for r in (range(10, 11) if smoke else range(10, 20)):
                add(exp, repeat=r)
            add(exp, kind="counterexample")
        elif exp == "E5":
            for s in sigmas:
                add(exp, phase="population", sigma=s)
            for n in ((240000,) if smoke else (240000, 480000, 960000)):
                for r in (range(1) if smoke else range(20)):
                    add(exp, phase="finite", n=n, repeat=r)
        else:
            raise ValueError(exp)
    return result


def plan(experiments, profile, backend):
    return dict(
        schema="PAPER_FRESH_PLAN_V1",
        experiments=list(experiments),
        profile=profile,
        backend=backend,
        paper_evidence=profile == "paper",
        jobs=jobs(experiments, profile),
        source_only=True,
        history_replayed=False,
        note="CPU is an explicit full-size reference implementation, not a GPU fallback. Smoke changes budgets and is never paper evidence.",
    )
