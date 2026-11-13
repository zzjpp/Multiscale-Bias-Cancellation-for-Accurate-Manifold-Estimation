from __future__ import annotations

from collections import Counter
import numpy as np
from scipy.spatial import cKDTree

from .config import CONFIG


def method_metrics(geometry, query_rows: list[dict], method: str, truth_count: int | None = None) -> dict:
    successful = [np.asarray(row[method]["point"], dtype=float) for row in query_rows if row[method]["status"] == "success"]
    failures = [row[method]["failure_reason"] for row in query_rows if row[method]["status"] != "success"]
    count = len(successful)
    total = len(query_rows)
    base = {"query_success_count": count, "query_failure_count": total - count, "query_count": total, "query_failure_rate": (total - count) / total if total else float("nan"), "failure_taxonomy": dict(Counter(failures)), "complete": count == total}
    if not successful:
        return {**base, "E_infty": float("nan"), "E_avg": float("nan"), "E_left": float("nan"), "H_grid": float("nan"), "metric_status": "no_successful_outputs", "distance_name": "grid-based bidirectional distance; not exact Hausdorff distance"}
    points = np.asarray(successful)
    errors = geometry.distance(points)
    if truth_count is None:
        truth_count = 256 * 256 if geometry.torus else 65536
    truth = geometry.query_grid(truth_count)
    reverse = float(cKDTree(points).query(truth, workers=1)[0].max())
    e_infty = float(errors.max())
    return {**base, "E_infty": e_infty, "E_avg": float(errors.mean()), "E_left": reverse, "H_grid": max(e_infty, reverse), "metric_status": "complete" if count == total else "partial_without_interpolation", "truth_grid_count": int(truth_count), "distance_name": "grid-based bidirectional distance; not exact Hausdorff distance"}


def finalize_cell(geometry, cell: dict, truth_count: int | None = None) -> dict:
    single_rows = cell["single"]["queries"]
    multi_rows = cell["multi"]["queries"]
    cell["Single_h2_metrics"] = method_metrics(geometry, single_rows, "Single_h2", truth_count)
    cell["Single_h1_metrics"] = method_metrics(geometry, single_rows, "Single_h1", truth_count)
    cell["ENF_Multi_metrics"] = method_metrics(geometry, multi_rows, "ENF_Multi", truth_count)
    successes = [cell[name]["query_success_count"] for name in ("Single_h2_metrics", "Single_h1_metrics", "ENF_Multi_metrics")]
    cell["status"] = "success" if all(value == len(single_rows) for value in successes) else "failed" if all(value == 0 for value in successes) else "partial"
    cell["query_success_counts"] = {name: cell[name]["query_success_count"] for name in ("Single_h2_metrics", "Single_h1_metrics", "ENF_Multi_metrics")}
    cell["query_failure_counts"] = {name: cell[name]["query_failure_count"] for name in ("Single_h2_metrics", "Single_h1_metrics", "ENF_Multi_metrics")}
    return cell


def reliable(query_failure_rate: float, complete_repeats: int) -> bool:
    return query_failure_rate <= CONFIG["validation"]["query_failure_rate_max"] and complete_repeats >= CONFIG["validation"]["complete_repeats_min"]


def summarize_repeats(values: list[float]) -> dict:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan"), "count": 0}
    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    return {"median": float(median), "q1": float(q1), "q3": float(q3), "count": int(len(finite))}
