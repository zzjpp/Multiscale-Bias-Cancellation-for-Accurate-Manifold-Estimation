from __future__ import annotations

import datetime
import os
import platform
import socket
import time
from pathlib import Path
import numpy as np
import scipy

from .coefficients import two_scale_weights
from .config import CONFIG, CONFIG_HASH, code_hash, result_root
from .geometry import BY_KEY, noisy_sample
from .metrics import finalize_cell
from .packets import atomic_write, read_packet


def parent_id(geometry: str, repeat: int) -> str:
    return f"e3_{geometry}_r{repeat:02d}"


def validate_parent(packet: dict, expected_geometry: str | None = None, expected_repeat: int | None = None) -> dict:
    geometry_key = packet["geometry"]
    if expected_geometry is not None and geometry_key != expected_geometry:
        raise ValueError("geometry mismatch")
    if expected_repeat is not None and packet["repeat"] != expected_repeat:
        raise ValueError("repeat mismatch")
    if packet["config_hash"] != CONFIG_HASH:
        raise ValueError("config hash mismatch")
    if packet["schema_version"] != "E3_PARENT_V1":
        raise ValueError("schema mismatch")
    if packet["task_id"] != parent_id(geometry_key, packet["repeat"]):
        raise ValueError("task identity mismatch")
    expected = CONFIG["bandwidths_h2"] if packet["mode"] in {"production", "full_parent"} else packet["requested_bandwidths"]
    if [row["h2"] for row in packet["bandwidths"]] != expected:
        raise ValueError("bandwidth records incomplete or reordered")
    dimension = BY_KEY[geometry_key].D
    for cell in packet["bandwidths"]:
        for pipeline, method in (("single", "Single_h2"), ("single", "Single_h1"), ("multi", "ENF_Multi")):
            for row in cell[pipeline]["queries"]:
                point = np.asarray(row[method]["point"])
                if point.shape != (dimension,):
                    raise ValueError("invalid point dimension")
                if row[method]["status"] == "success" and not np.isfinite(point).all():
                    raise ValueError("nonfinite successful point")
                if row[method]["status"] != "success" and (not np.isnan(point).all() or not row[method]["failure_reason"]):
                    raise ValueError("failure must carry NaN point and reason")
        if cell["single"]["pipeline_id"] == cell["multi"]["pipeline_id"]:
            raise ValueError("timing paths were not isolated")
    if not packet.get("execution_complete"):
        raise ValueError("truncated parent")
    return packet


def _single_cache(cells: list[dict]) -> list[dict]:
    cache = {}
    for cell in cells:
        for bandwidth_key, method in (("h1", "Single_h1"), ("h2", "Single_h2")):
            bandwidth = cell[bandwidth_key]
            key = f"{bandwidth:.12g}"
            if key not in cache:
                cache[key] = {"bandwidth": bandwidth, "source_h2": cell["h2"], "method": method, "metrics": cell[f"{method}_metrics"], "query_results": [row[method] for row in cell["single"]["queries"]], "reliability_status": "pending_cross_repeat_analysis", "E_best_candidate_status": "pending_cross_repeat_analysis"}
    return [cache[key] for key in sorted(cache, key=lambda item: float(item))]


def run_parent(geometry_key: str, repeat: int, root_value=None, mode="smoke", bandwidths=None, query_count=None, resume=False, production_confirmed=False, truth_count=None):
    if geometry_key not in BY_KEY:
        raise ValueError("unknown geometry")
    if repeat not in range(CONFIG["repeats"]):
        raise ValueError("repeat out of range")
    if mode == "production" and not production_confirmed:
        raise RuntimeError("production requires explicit confirmation and gate audit")
    if mode not in {"smoke", "single_query", "five_geometry", "bandwidth_extremes", "full_parent", "coverage_smoke", "benchmark", "production"}:
        raise ValueError("unknown mode")
    geometry = BY_KEY[geometry_key]
    bandwidths = list(CONFIG["bandwidths_h2"] if bandwidths is None else bandwidths)
    query_count = CONFIG["N_q"] if query_count is None else int(query_count)
    queries = geometry.query_grid(query_count)
    root = result_root(root_value)
    destination = root / ("packets" if mode == "production" else "audits") / f"{parent_id(geometry_key, repeat)}.{mode}.json.gz"
    current_code_hash = code_hash()
    if destination.exists():
        if not resume:
            raise FileExistsError("existing packet; use --resume")
        packet = validate_parent(read_packet(destination, CONFIG_HASH, current_code_hash), geometry_key, repeat)
        return {"status": "skipped_valid", "packet": str(destination), "task_id": packet["task_id"]}
    start = time.perf_counter()
    clean, noise, data, lineage = noisy_sample(geometry, repeat)
    sample_time = time.perf_counter() - start
    cells = []
    from .pipelines import run_cell
    for h2 in bandwidths:
        weights = two_scale_weights(geometry.d, geometry.D, h2 / 2, h2, CONFIG["sigma"])
        if weights["status"] != "converged":
            raise RuntimeError(f"coefficient integration unresolved for {geometry_key}, h2={h2}")
        cell = run_cell(data, geometry, h2, queries, weights)
        cell["coefficients"] = weights["coefficients"]
        cell["weights"] = weights["weights"]
        cell["weight_constraint_residuals"] = weights["weight_constraint_residuals"]
        cell["sum_abs_weights"] = weights["sum_abs_weights"]
        finalize_cell(geometry, cell, truth_count)
        cells.append(cell)
    packet = {"schema_version": "E3_PARENT_V1", "config_hash": CONFIG_HASH, "code_hash": current_code_hash, "task_id": parent_id(geometry_key, repeat), "geometry": geometry_key, "geometry_label": geometry.label, "d": geometry.d, "D": geometry.D, "q": geometry.q, "repeat": repeat, "seed": lineage, "sigma": CONFIG["sigma"], "n": CONFIG["n"], "N_q": query_count, "mode": mode, "requested_bandwidths": bandwidths, "query_coordinates": queries, "query_grid_tolerance": CONFIG["validation"]["query_grid_tolerance"], "data_hashes": {"clean": lineage["clean_sha256"], "noise": lineage["noise_sha256"], "observed": lineage["observed_sha256"]}, "bandwidths": cells, "single_cache": _single_cache(cells), "source_hashes": None, "environment": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}, "timing": {"sample_generation": sample_time, "total_before_serialization": time.perf_counter() - start}, "execution_complete": True, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    from .config import source_hashes
    packet["source_hashes"] = source_hashes()
    validate_parent(packet, geometry_key, repeat)
    atomic_write(destination, packet)
    return {"status": "saved", "packet": str(destination), "task_id": packet["task_id"], "cell_statuses": [cell["status"] for cell in cells]}
