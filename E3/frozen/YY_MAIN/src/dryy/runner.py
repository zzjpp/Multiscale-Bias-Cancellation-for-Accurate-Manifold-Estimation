from __future__ import annotations

import datetime as dt
import platform
import socket
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree

from .coefficients import dr_weights, yy_weights
from .config import code_hash, load_config
from .dr import fit_dr_query
from .geometry import geometries_from_config, noisy_sample
from .io import write_packet
from .metrics import summarize_points
from .yy import YYFailure
from .yy_common import evaluate_common_start, prepare_common_start


def _geometry(config: dict, key: str):
    matches = [geometry for geometry in geometries_from_config(config) if geometry.key == key]
    if len(matches) != 1:
        raise ValueError("unknown geometry")
    return matches[0]


def _success(point, **diagnostics):
    return {"status": "success", "reason": None, "point": np.asarray(point), "diagnostics": diagnostics}


def _failure(reason: str, **diagnostics):
    return {"status": "failed", "reason": str(reason), "point": None, "diagnostics": diagnostics}


def _dr_cell(data, geometry, queries, sigma: float, h1: float, h2: float, config: dict) -> dict:
    controls = dict(config["dr"])
    controls.setdefault("relative_omitted_weight_bound_max", 0.0)
    tree = cKDTree(data) if controls.get("gaussian_truncation_factor") is not None else None
    weights = dr_weights(h1, h2, sigma)
    single_h1_rows, single_rows, multi_rows = [], [], []
    for truth in queries:
        fitted = fit_dr_query(data, geometry, truth, h1, h2, sigma, controls, config["solver"], tree)
        if fitted["single"]["status"] == "success":
            single_rows.append(_success(fitted["single"]["point"]))
        else:
            single_rows.append(_failure(fitted["single"]["reason"], **fitted["single"].get("diagnostics", {})))
        if fitted["fine"]["status"] == "success":
            single_h1_rows.append(_success(fitted["fine"]["point"]))
        else:
            single_h1_rows.append(_failure(fitted["fine"]["reason"], **fitted["fine"].get("diagnostics", {})))
        if fitted["multi"]["status"] == "fit_success":
            point = weights["weights"][0] * fitted["fine_point"] + weights["weights"][1] * fitted["coarse_point"]
            multi_rows.append(_success(point))
        else:
            multi_rows.append(_failure(fitted["multi"]["reason"], **fitted["multi"].get("diagnostics", {})))
    return {
        "method": "DR", "h1": float(h1), "h2": float(h2), "weights": weights,
        "Single_h1": summarize_points(geometry, single_h1_rows, "point"),
        "Single_h2": summarize_points(geometry, single_rows, "point"),
        "Single": summarize_points(geometry, single_rows, "point"),
        "Multi": summarize_points(geometry, multi_rows, "point"),
        "single_h1_rows": single_h1_rows,
        "single_rows": single_rows, "multi_rows": multi_rows,
    }


def _yy_cell(data, tree, geometry, queries, sigma: float, central_kappa: float, config: dict) -> dict:
    controls = dict(config["yy"])
    controls["weight_residual_tolerance"] = config["validation"]["weight_residual_tolerance"]
    relative = np.asarray(controls["relative_radii"], dtype=float)
    radius_rows = [[] for _ in relative]
    single_rows, multi_rows, starts = [], [], []
    for truth in queries:
        try:
            start = prepare_common_start(data, tree, geometry, truth, sigma, controls)
            fitted = evaluate_common_start(data, tree, geometry, start, sigma, central_kappa, controls)
        except YYFailure as error:
            for rows in radius_rows:
                rows.append(_failure(error.reason, **error.diagnostics))
            single_rows.append(_failure(error.reason, **error.diagnostics))
            multi_rows.append(_failure(error.reason, **error.diagnostics))
            starts.append(None)
            continue
        starts.append(start["digest"])
        for index, (point, stage) in enumerate(zip(fitted["single_radius_points"], fitted["stages"])):
            if point is not None:
                radius_rows[index].append(_success(point, common_start_digest=start["digest"]))
            else:
                radius_rows[index].append(_failure(stage["reason"], common_start_digest=start["digest"], stage=stage))
        if fitted["single_status"] == "success":
            single_rows.append(_success(fitted["single"], common_start_digest=start["digest"]))
        else:
            stage = fitted["stages"][1]
            single_rows.append(_failure(stage["reason"], common_start_digest=start["digest"], stage=stage))
        if fitted["multi_status"] == "success":
            multi_rows.append(_success(fitted["multi"], common_start_digest=start["digest"]))
        else:
            failed = [stage for stage in fitted["stages"] if stage["status"] != "success"]
            multi_rows.append(_failure("second_stage_incomplete", common_start_digest=start["digest"], failed_stages=failed))
    kappas = central_kappa * relative
    radius_summaries = [summarize_points(geometry, rows, "point") for rows in radius_rows]
    return {
        "method": "YY", "central_kappa": float(central_kappa), "r1": float(central_kappa * sigma),
        "kappas": kappas, "radii": sigma * kappas,
        "weights": yy_weights(geometry.d, geometry.q, kappas, controls["kernel_power"]),
        "Single_radii": radius_summaries,
        "Single": summarize_points(geometry, single_rows, "point"),
        "Multi": summarize_points(geometry, multi_rows, "point"),
        "single_radius_rows": radius_rows,
        "single_rows": single_rows, "multi_rows": multi_rows,
        "common_start_digests": starts,
    }


def run_parent(experiment: str, method: str, geometry_key: str, repeat: int, output_root: Path,
               query_count: int | None = None, e2_n_values=None, e3_scales=None, overwrite: bool = False) -> Path:
    config, config_hash = load_config(experiment)
    method = method.upper()
    if method not in {"DR", "YY"}:
        raise ValueError("method must be DR or YY")
    if repeat not in range(int(config["repeats"])):
        raise ValueError("repeat out of range")
    geometry = _geometry(config, geometry_key)
    query_count = int(config["N_q"] if query_count is None else query_count)
    queries = geometry.query_grid(query_count, int(config["query_seed"]))
    name = f"{experiment.lower()}_{method.lower()}_{geometry.key}_r{repeat:02d}.json.gz"
    destination = Path(output_root) / "packets" / name
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    start_time = time.perf_counter()

    if experiment.lower() == "e2":
        observed, lineage = noisy_sample(config, geometry, repeat, int(config["maximum_n"]))
        n_values = list(config["n_values"] if e2_n_values is None else e2_n_values)
        cells = []
        for n in n_values:
            data = observed[: int(n)]
            if method == "DR":
                cell = _dr_cell(data, geometry, queries, float(config["sigma"]), float(config["dr"]["h1"]), float(config["dr"]["h2"]), config)
            else:
                cell = _yy_cell(data, cKDTree(data), geometry, queries, float(config["sigma"]), float(config["yy"]["central_kappa"]), config)
            cell["n"] = int(n)
            cells.append(cell)
        scale_name = "n"
    else:
        observed, lineage = noisy_sample(config, geometry, repeat, int(config["n"]))
        cells = []
        if method == "DR":
            scales = list(config["dr"]["bandwidths_h2"] if e3_scales is None else e3_scales)
            for h2 in scales:
                cells.append(_dr_cell(observed, geometry, queries, float(config["sigma"]), float(h2) * float(config["dr"]["h1_ratio"]), float(h2), config))
            scale_name = "h2"
        else:
            scales = list(config["yy"]["central_kappas"] if e3_scales is None else e3_scales)
            tree = cKDTree(observed)
            controls = dict(config["yy"])
            controls["weight_residual_tolerance"] = config["validation"]["weight_residual_tolerance"]
            relative = np.asarray(controls["relative_radii"], dtype=float)
            per_scale = {
                float(value): {
                    "single": [], "single_radii": [[] for _ in relative], "multi": [], "digests": []
                }
                for value in scales
            }
            for truth in queries:
                try:
                    common = prepare_common_start(observed, tree, geometry, truth, float(config["sigma"]), controls)
                except YYFailure as error:
                    for value in scales:
                        for radius_rows in per_scale[float(value)]["single_radii"]:
                            radius_rows.append(_failure(error.reason, **error.diagnostics))
                        per_scale[float(value)]["single"].append(_failure(error.reason, **error.diagnostics))
                        per_scale[float(value)]["multi"].append(_failure(error.reason, **error.diagnostics))
                        per_scale[float(value)]["digests"].append(None)
                    continue
                for value in scales:
                    fitted = evaluate_common_start(observed, tree, geometry, common, float(config["sigma"]), float(value), controls)
                    per_scale[float(value)]["digests"].append(common["digest"])
                    for index, (point, stage) in enumerate(zip(fitted["single_radius_points"], fitted["stages"])):
                        if point is not None:
                            per_scale[float(value)]["single_radii"][index].append(
                                _success(point, common_start_digest=common["digest"])
                            )
                        else:
                            per_scale[float(value)]["single_radii"][index].append(
                                _failure(stage["reason"], common_start_digest=common["digest"], stage=stage)
                            )
                    if fitted["single_status"] == "success":
                        per_scale[float(value)]["single"].append(_success(fitted["single"], common_start_digest=common["digest"]))
                    else:
                        per_scale[float(value)]["single"].append(_failure("second_stage_mass_failure", common_start_digest=common["digest"]))
                    if fitted["multi_status"] == "success":
                        per_scale[float(value)]["multi"].append(_success(fitted["multi"], common_start_digest=common["digest"]))
                    else:
                        failed = [stage for stage in fitted["stages"] if stage["status"] != "success"]
                        per_scale[float(value)]["multi"].append(_failure("second_stage_incomplete", common_start_digest=common["digest"], failed_stages=failed))
            for value in scales:
                rows = per_scale[float(value)]
                kappas = float(value) * np.asarray(controls["relative_radii"], dtype=float)
                cells.append({
                    "method": "YY", "central_kappa": float(value), "r1": float(value) * float(config["sigma"]),
                    "kappas": kappas, "radii": float(config["sigma"]) * kappas,
                    "weights": yy_weights(geometry.d, geometry.q, kappas, controls["kernel_power"]),
                    "Single_radii": [
                        summarize_points(geometry, radius_rows, "point")
                        for radius_rows in rows["single_radii"]
                    ],
                    "Single": summarize_points(geometry, rows["single"], "point"),
                    "Multi": summarize_points(geometry, rows["multi"], "point"),
                    "single_radius_rows": rows["single_radii"],
                    "single_rows": rows["single"], "multi_rows": rows["multi"],
                    "common_start_digests": rows["digests"],
                })
            scale_name = "central_kappa"

    packet = {
        "schema_version": f"{experiment.upper()}_{method}_PARENT_V2",
        "experiment": experiment.lower(), "method": method, "geometry": geometry.key,
        "d": geometry.d, "D": geometry.D, "q": geometry.q, "repeat": int(repeat),
        "config_hash": config_hash, "code_hash": code_hash(), "sigma": float(config["sigma"]),
        "query_count": query_count, "query_coordinates": queries, "seed": lineage,
        "scale_name": scale_name, "cells": cells, "execution_complete": True,
        "runtime_seconds": float(time.perf_counter() - start_time),
        "environment": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_packet(destination, packet)
    return destination

