"""Isolated population adaptation of arXiv:2610.09423 Algorithms 1--2."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "frozen"))
from e1_theory_path.geometry import GEOMETRIES
from e1_theory_path.population import PopulationSTSP, radial_mass

CONFIG = json.loads((ROOT / "config.json").read_bytes())
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items() if not k.startswith("_private")}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def write_fresh(path, payload):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Output outside isolated study")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plain(payload), indent=2, allow_nan=False) + "\n").encode()
    if path.name.endswith(".gz"):
        with path.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
                out.write(encoded)
    else:
        with path.open("xb") as out:
            out.write(encoded)
            out.flush()
            os.fsync(out.fileno())


def read_packet(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def original_config(key):
    name = "original_corrugated_config.json" if key == "corrugated_torus_r3" else "original_five_config.json"
    return json.loads((ROOT / "frozen" / name).read_bytes())


def baseline_budget():
    audit = json.loads((ROOT / "frozen/baseline/e1_theory_path_audit.json").read_bytes())
    minimum = audit["numerical_maxima"]["minimum_production_multi_error_including_average"]
    return float(minimum) * CONFIG["global_reference_difference_fraction"]


def weights(radii):
    radii = np.asarray(radii, dtype=np.float64)
    if radii.ndim != 1 or len(radii) < 2 or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("At least two finite positive radii required")
    # Scale-free version of the paper's OLS-intercept identity (Lemma S9).
    s = (radii / np.max(radii)) ** 2
    mean = float(s.mean())
    centered = s - mean
    denominator = float(centered @ centered)
    if denominator <= 0:
        raise ValueError("Distinct squared scales required")
    w = np.full(len(s), 1.0 / len(s)) - mean * centered / denominator
    if max(abs(w.sum() - 1), abs(w @ s)) > CONFIG["weight_constraint_max"]:
        raise ArithmeticError("OLS weight constraints failed")
    return w


class NoTrueNormals:
    def __init__(self, geometry):
        self._geometry = geometry

    def __getattr__(self, name):
        if name == "normal_basis_at":
            raise AssertionError("True normal forbidden in fitting")
        return getattr(self._geometry, name)


def operator(key, sigma, relative_radius, resolution, backend, device=0, reduced=False):
    cfg = original_config(key)
    design = cfg["geometries"][key]
    latent = int(design["integration_resolution"][resolution])
    radial = int(cfg["resolutions"][resolution]["stsp_radial_order"])
    if reduced:
        latent = 512 if key == "circle_r2" else 96 if key == "clifford_torus_r4" else 384
        radial = 64
    geometry = NoTrueNormals(GEOMETRIES[key])
    arguments = (geometry, relative_radius * sigma, sigma, latent, radial)
    kwargs = dict(power=0, noncentral_radius_padding=cfg["population_integration"]["stsp_noncentral_radius_padding"], kernel_kind="polynomial")
    if backend == "cpu":
        return PopulationSTSP(*arguments, **kwargs)
    if backend == "gpu_original":
        from e1_theory_path.gpu_population import GpuPopulationSTSP
        return GpuPopulationSTSP(*arguments, device=device, **kwargs)
    if backend == "gpu":
        from fast_gpu import FastGpuPopulationSTSP
        return FastGpuPopulationSTSP(*arguments, device=device, **kwargs)
    raise ValueError(backend)


class OrbitFailure(ArithmeticError):
    def __init__(self, reason, traces):
        super().__init__(reason)
        self.traces = traces


def inspect(op, point, start):
    if not np.isfinite(point).all():
        raise ArithmeticError("nonfinite_orbit")
    if np.linalg.norm(point - start) > CONFIG["search_radius_factor"] * op.h ** 2:
        raise ArithmeticError("original_E1_search_tube_exceeded")
    state = op.evaluate(point)
    if not np.isfinite(state.residual).all() or not np.isfinite(state.normal_basis).all():
        raise ArithmeticError("nonfinite_population_state")
    if not np.isfinite(state.normalized_mass) or state.normalized_mass <= 0:
        raise ArithmeticError("nonpositive_population_mass")
    if not np.isfinite(state.normalized_eigengap) or state.normalized_eigengap < CONFIG["normalized_eigengap_min"]:
        raise ArithmeticError("eigengap_failure")
    full = float(np.linalg.norm(state.residual))
    projected = float(np.linalg.norm(state.normal_basis.T @ state.residual))
    return state, dict(point=point.copy(), residual=state.residual.copy(), projected_residual=projected,
                       full_residual=full, normalized_mass=float(state.normalized_mass),
                       normalized_eigengap=float(state.normalized_eigengap))


def passed(row):
    return row["projected_residual"] <= CONFIG["projected_residual_tolerance"] and row["full_residual"] <= CONFIG["full_residual_tolerance"]


def orbit_triplet(ops, start):
    start = np.asarray(start, dtype=np.float64)
    points = np.repeat(start[None, :], len(ops), axis=0)
    w = weights([op.h for op in ops])
    traces = [[] for _ in ops]
    first_accept = None
    candidate = None
    try:
        for iteration in range(CONFIG["maximum_orbit_iterations"] + 1):
            states, rows = [], []
            for i, op in enumerate(ops):
                state, row = inspect(op, points[i], start)
                row["iteration"] = iteration
                traces[i].append(row)
                states.append(state)
                rows.append(row)
            if first_accept is None and all(passed(r) for r in rows):
                first_accept = iteration
                candidate = w @ points
            if first_accept is not None and iteration >= first_accept + CONFIG["validation_extra_iterations"]:
                multi = w @ points
                extension = float(np.linalg.norm(multi - candidate))
                if not all(passed(r) for r in rows):
                    raise ArithmeticError("residual_lost_after_extension")
                if extension >= baseline_budget():
                    raise ArithmeticError("orbit_extension_exceeds_original_global_budget")
                return dict(status="success", points=points.copy(), multi=multi, horizon=iteration,
                            first_accept=first_accept, acceptance_multi=candidate.copy(), extension_difference=extension, traces=traces)
            if iteration == CONFIG["maximum_orbit_iterations"]:
                raise ArithmeticError("maximum_orbit_iterations")
            # Algorithm1 update, without normal-fiber Newton or moving reference data.
            points = np.asarray([y - state.residual for y, state in zip(points, states)])
    except Exception as error:
        raise OrbitFailure(type(error).__name__ + ": " + str(error), traces) from error


def align_horizon(result, ops, start, horizon):
    points = np.asarray(result["points"]).copy()
    w = weights([op.h for op in ops])
    for iteration in range(result["horizon"], horizon):
        updated = []
        for i, op in enumerate(ops):
            state, row = inspect(op, points[i], start)
            if iteration > result["horizon"]:
                row["iteration"] = iteration
                result["traces"][i].append(row)
            updated.append(points[i] - state.residual)
        points = np.asarray(updated)
    if horizon > result["horizon"]:
        for i, op in enumerate(ops):
            _, row = inspect(op, points[i], start)
            row["iteration"] = horizon
            result["traces"][i].append(row)
            if not passed(row):
                raise OrbitFailure("common_horizon_residual_failure", result["traces"])
    multi = w @ points
    extension = float(np.linalg.norm(multi - result["acceptance_multi"]))
    if extension >= baseline_budget():
        raise OrbitFailure("common_horizon_extension_exceeds_original_budget", result["traces"])
    result.update(points=points, multi=multi, horizon=horizon, extension_difference=extension)
    return result


def run_cell(key, sigma, resolution, backend="cpu", device=0, query_ids=None, reduced=False):
    begun = time.monotonic()
    cfg = original_config(key)
    design = cfg["geometries"][key]
    geometry = GEOMETRIES[key]
    queries, _ = geometry.query_grid(design["query_count"], design.get("query_phase", 0.0))
    ids = list(range(len(queries))) if query_ids is None else list(query_ids)
    radii = CONFIG["ms_radii_over_sigma"]
    ops = [operator(key, sigma, c, resolution, backend, device, reduced) for c in radii]
    w = weights(radii)
    latent_resolution = int(design["integration_resolution"][resolution])
    if reduced:
        latent_resolution = 512 if key == "circle_r2" else 96 if key == "clifford_torus_r4" else 384
    rows = []
    for index in ids:
        try:
            row = orbit_triplet(ops, queries[index])
        except OrbitFailure as error:
            row = dict(status="failure", reason=str(error), traces=error.traces)
        row.update(query_id=index, truth=queries[index].copy())
        rows.append(row)
    common = max((r["horizon"] for r in rows if r["status"] == "success"), default=0)
    for i, row in enumerate(rows):
        if row["status"] != "success":
            continue
        try:
            rows[i] = align_horizon(row, ops, queries[row["query_id"]], common)
            rows[i]["geometric_error"] = float(geometry.distances(rows[i]["multi"][None, :])[0])
        except (OrbitFailure, ArithmeticError) as error:
            row.update(status="failure", reason=str(error))
    success = [r for r in rows if r["status"] == "success"]
    errors = [r["geometric_error"] for r in success]
    return dict(schema="MS_STSP_E1_POPULATION_CELL_V1", geometry=key, sigma=float(sigma), resolution=resolution,
                backend=backend, device=device if backend.startswith("gpu") else None, reduced_quadrature=reduced,
                formal_scope=not reduced and query_ids is None, config_sha256=sha(ROOT / "config.json"),
                source_sha256=sha(Path(__file__)), frozen_manifest_sha256=sha(ROOT / "FROZEN_INPUTS.json"),
                original_config_sha256=sha(ROOT / "frozen" / ("original_corrugated_config.json" if key == "corrugated_torus_r3" else "original_five_config.json")),
                radii_over_sigma=radii, weights=w, weight_l1=float(np.abs(w).sum()), kernel_power=0,
                reference_updated=False, initialization="original noiseless base queries", common_horizon=common,
                latent_resolution=latent_resolution,
                radial_order=ops[0].radial_order, query_count=len(ids), expected_full_query_count=len(queries),
                success_count=len(success), failure_count=len(rows)-len(success), all_queries_successful=len(success)==len(rows),
                E_infinity=max(errors) if errors else None, E_average=float(np.mean(errors)) if errors else None,
                maximum_extension_difference=max((r["extension_difference"] for r in success), default=None),
                seconds=time.monotonic()-begun, query_rows=rows)


def freeze_manifest():
    selected = {}
    for p in sorted((ROOT / "frozen").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            selected[p.relative_to(ROOT).as_posix()] = dict(size=p.stat().st_size, sha256=sha(p))
    source = ROOT.parent / "simulation_experiments/E1"
    protected = {}
    for p in sorted(source.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
            protected[p.relative_to(source).as_posix()] = dict(size=p.stat().st_size, sha256=sha(p))
    # Confirm the reused table has its selected-release bytes.
    provenance = json.loads((ROOT / "frozen/baseline/provenance.json").read_bytes())
    for name in ("e1_theory_path_errors.csv", "e1_theory_path_audit.json", "e1_theory_path_numerical_convergence.csv"):
        if sha(ROOT / "frozen/baseline" / name) != provenance["data_sha256"][name]:
            raise ArithmeticError("Selected baseline evidence changed: " + name)
    write_fresh(ROOT / "FROZEN_INPUTS.json", dict(schema="ORIGINAL_E1_FROZEN_INPUTS_V1", members=selected,
                protected_original_root=str(source), protected_original_members=protected,
                original_table_rows=sum(1 for _ in csv.DictReader((ROOT / "frozen/baseline/e1_theory_path_errors.csv").open())),
                global_numeric_budget=baseline_budget(), radii=CONFIG["ms_radii_over_sigma"], weights=weights(CONFIG["ms_radii_over_sigma"])))
    print(json.dumps(dict(frozen_members=len(selected), protected_original_members=len(protected), budget=baseline_budget())))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    run = sub.add_parser("cell")
    run.add_argument("--geometry", choices=CONFIG["geometry_order"], required=True)
    run.add_argument("--sigma", type=float, required=True)
    run.add_argument("--resolution", choices=("production", "reference"), required=True)
    run.add_argument("--backend", choices=("cpu", "gpu", "gpu_original"), default="cpu")
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--query-ids", type=int, nargs="+")
    run.add_argument("--reduced", action="store_true")
    run.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "freeze":
        freeze_manifest()
        return
    if args.output.exists():
        raise FileExistsError("Never overwrite/refit a saved packet")
    result = run_cell(args.geometry, args.sigma, args.resolution, args.backend, args.device, args.query_ids, args.reduced)
    write_fresh(args.output, result)
    print(json.dumps({k: plain(v) for k, v in result.items() if k != "query_rows"}))
    if not result["all_queries_successful"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
