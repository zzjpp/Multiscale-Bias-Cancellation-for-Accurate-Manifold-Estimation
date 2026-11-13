from __future__ import annotations

from collections import Counter

import numpy as np


def summarize_points(geometry, rows: list[dict], key: str) -> dict:
    successful = [np.asarray(row[key], dtype=float) for row in rows if row.get("status") == "success"]
    failures = [row.get("reason", "other") for row in rows if row.get("status") != "success"]
    total = len(rows)
    count = len(successful)
    base = {
        "query_count": total,
        "query_success_count": count,
        "query_failure_count": total - count,
        "query_failure_rate": (total - count) / total if total else float("nan"),
        "complete": count == total,
        "failure_taxonomy": dict(Counter(failures)),
    }
    if not count:
        return {**base, "E_avg": float("nan"), "E_infty": float("nan"), "metric_status": "no_successful_outputs"}
    errors = geometry.distance(np.asarray(successful))
    return {
        **base,
        "E_avg": float(np.mean(errors)),
        "E_infty": float(np.max(errors)),
        "metric_status": "complete" if count == total else "partial_without_interpolation",
    }


def repeat_summary(values) -> dict:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"count": 0, "median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    return {"count": int(len(finite)), "median": float(median), "q1": float(q1), "q3": float(q3)}

