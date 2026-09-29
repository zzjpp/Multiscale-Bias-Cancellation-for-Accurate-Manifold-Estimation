from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
E3 = ROOT.parent / "E3"
STSP = E3 / "methods" / "stsp"
DRYY = E3 / "methods" / "dr_yy"
sys.path.insert(0, str(STSP / "src"))
sys.path.insert(0, str(DRYY / "src"))

from e3.packets import read_packet as read_stsp_packet  # noqa: E402
from dryy.config import load_config as load_dryy_config  # noqa: E402
from dryy.geometry import geometries_from_config  # noqa: E402
from dryy.io import read_packet as read_dryy_packet  # noqa: E402


METHODS = ("STSP", "DR", "YY")
ESTIMATORS = ("Single", "Multi")
REFERENCE_SEED_OFFSET = 7919


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def successful_points(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [row["point"] for row in rows if row.get("status") == "success"],
        dtype=float,
    )


def reverse_metrics(rows: list[dict], reference: np.ndarray) -> tuple[float, float, int]:
    points = successful_points(rows)
    if not len(points):
        return float("nan"), float("nan"), 0
    distances = cKDTree(points).query(reference, k=1, workers=-1)[0]
    return float(distances.max()), float(distances.mean()), int(len(points))


def grid_mismatch(actual: np.ndarray, expected: np.ndarray) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    return float(max(
        cKDTree(expected).query(actual, k=1, workers=-1)[0].max(),
        cKDTree(actual).query(expected, k=1, workers=-1)[0].max(),
    ))


def matching_cell(cells: list[dict], scale: float, scale_key: str) -> dict:
    matches = [
        cell for cell in cells
        if np.isclose(float(cell[scale_key]), scale, rtol=0.0, atol=1e-14)
    ]
    if len(matches) != 1:
        raise ValueError(f"scale missing or duplicated: {scale_key}={scale}")
    return matches[0]


def analyze_stsp(
    result_root: Path,
    geometry_key: str,
    geometry,
    reference: np.ndarray,
    expected_queries: np.ndarray,
    failure_rate_max: float,
) -> tuple[list[dict], dict]:
    path = (
        result_root / "stage1" / "merged" / "stsp" / "packets" /
        f"e3_stsp_{geometry_key}_r00.json.gz"
    )
    packet = read_stsp_packet(path)
    queries = np.asarray(packet["query_coordinates"], dtype=float)
    mismatch = grid_mismatch(queries, expected_queries)
    if mismatch > 1e-14:
        raise ValueError(f"STSP query-grid mismatch: {geometry_key}: {mismatch:.3e}")
    delta = float(cKDTree(queries).query(reference, k=1, workers=-1)[0].max())
    rows = []
    for cell in packet["bandwidths"]:
        scale = float(cell["h2"])
        for estimator, pipeline, result_key in (
            ("Single", "single", "Single_h2"),
            ("Multi", "multi", "ENF_Multi"),
        ):
            fitted_rows = [row[result_key] for row in cell[pipeline]["queries"]]
            e_reverse, e_reverse_mean, success_count = reverse_metrics(fitted_rows, reference)
            failure_rate = (len(fitted_rows) - success_count) / len(fitted_rows)
            rows.append({
                "method": "STSP", "geometry": geometry_key, "estimator": estimator,
                "scale": scale, "query_count": len(fitted_rows),
                "query_success_count": success_count,
                "query_failure_rate": failure_rate,
                "reliable": failure_rate <= failure_rate_max,
                "E_reverse": e_reverse, "E_reverse_mean": e_reverse_mean,
                "delta_query": delta, "query_grid_mismatch": mismatch,
                "observed_sha256": packet["data_hashes"]["observed"],
                "scientific_code_hash": packet["code_hash"],
                "packet_path": str(path),
            })
    return rows, {"method": "STSP", "geometry": geometry_key, "packet": str(path),
                  "packet_sha256": sha256(path), "query_grid_mismatch": mismatch,
                  "delta_query": delta}


def analyze_dryy(
    result_root: Path,
    base_root: Path,
    bridge_root: Path,
    method: str,
    geometry_key: str,
    geometry,
    reference: np.ndarray,
    expected_queries: np.ndarray,
    failure_rate_max: float,
) -> tuple[list[dict], dict]:
    base_path = base_root / method.lower() / f"e3_{method.lower()}_{geometry_key}_r00.json.gz"
    bridge_path = bridge_root / f"e3_{method.lower()}_{geometry_key}_r00.json.gz"
    increment_path = (
        result_root / "stage1" / "merged" / "packets" /
        f"e3_{method.lower()}_{geometry_key}_r00.json.gz"
    )
    base = read_dryy_packet(base_path)
    bridge = read_dryy_packet(bridge_path)
    increment = read_dryy_packet(increment_path)
    if base["method"] != method or bridge["method"] != method or increment["method"] != method:
        raise ValueError("method identity mismatch")
    if base["geometry"] != geometry_key or bridge["geometry"] != geometry_key or increment["geometry"] != geometry_key:
        raise ValueError("geometry identity mismatch")
    observed_hashes = {
        base["seed"]["observed_sha256"],
        bridge["seed"]["observed_sha256"],
        increment["seed"]["observed_sha256"],
    }
    if len(observed_hashes) != 1:
        raise ValueError("base/bridge/increment observation hash mismatch")
    queries = np.vstack((
        np.asarray(base["query_coordinates"], dtype=float),
        np.asarray(bridge["query_coordinates"], dtype=float),
        np.asarray(increment["query_coordinates"], dtype=float),
    ))
    mismatch = grid_mismatch(queries, expected_queries)
    if mismatch > 1e-14:
        raise ValueError(f"{method} query-grid mismatch: {geometry_key}: {mismatch:.3e}")
    delta = float(cKDTree(queries).query(reference, k=1, workers=-1)[0].max())
    rows = []
    scale_key = "h2" if method == "DR" else "r1"
    for increment_cell in increment["cells"]:
        scale = float(increment_cell[scale_key])
        base_cell = matching_cell(base["cells"], scale, scale_key)
        bridge_cell = matching_cell(bridge["cells"], scale, scale_key)
        for estimator, row_key in (("Single", "single_rows"), ("Multi", "multi_rows")):
            fitted_rows = (
                list(base_cell[row_key])
                + list(bridge_cell[row_key])
                + list(increment_cell[row_key])
            )
            e_reverse, e_reverse_mean, success_count = reverse_metrics(fitted_rows, reference)
            failure_rate = (len(fitted_rows) - success_count) / len(fitted_rows)
            rows.append({
                "method": method, "geometry": geometry_key, "estimator": estimator,
                "scale": scale, "query_count": len(fitted_rows),
                "query_success_count": success_count,
                "query_failure_rate": failure_rate,
                "reliable": failure_rate <= failure_rate_max,
                "E_reverse": e_reverse, "E_reverse_mean": e_reverse_mean,
                "delta_query": delta, "query_grid_mismatch": mismatch,
                "observed_sha256": increment["seed"]["observed_sha256"],
                "scientific_code_hash": increment["code_hash"],
                "packet_path": str(increment_path),
            })
    return rows, {"method": method, "geometry": geometry_key,
                  "base_packet": str(base_path), "base_packet_sha256": sha256(base_path),
                  "bridge_packet": str(bridge_path), "bridge_packet_sha256": sha256(bridge_path),
                  "increment_packet": str(increment_path),
                  "increment_packet_sha256": sha256(increment_path),
                  "query_grid_mismatch": mismatch, "delta_query": delta}


def stage2_decision(rows: list[dict], minimum_length: int) -> dict:
    by_estimator = {
        estimator: {
            float(row["scale"]): row for row in rows if row["estimator"] == estimator
        }
        for estimator in ESTIMATORS
    }
    grid = sorted(set(by_estimator["Single"]) & set(by_estimator["Multi"]))
    if len(grid) != 28:
        raise ValueError(f"expected 28 common scales, found {len(grid)}")
    delta_values = {float(row["delta_query"]) for row in rows}
    if len(delta_values) != 1:
        raise ValueError("delta_query is not constant within method/geometry")
    delta = next(iter(delta_values))
    candidates = []
    start = 0
    while start < len(grid):
        scale = grid[start]
        single = by_estimator["Single"][scale]
        multi = by_estimator["Multi"][scale]
        reliable = bool(single["reliable"] and multi["reliable"])
        gap = float(multi["E_reverse"] - single["E_reverse"])
        sign = 1 if gap > 0 else -1 if gap < 0 else 0
        if not reliable or sign == 0 or not np.isfinite(gap):
            start += 1
            continue
        end = start
        while end + 1 < len(grid):
            next_scale = grid[end + 1]
            next_single = by_estimator["Single"][next_scale]
            next_multi = by_estimator["Multi"][next_scale]
            next_gap = float(next_multi["E_reverse"] - next_single["E_reverse"])
            next_sign = 1 if next_gap > 0 else -1 if next_gap < 0 else 0
            if not (next_single["reliable"] and next_multi["reliable"] and
                    np.isfinite(next_gap) and next_sign == sign):
                break
            end += 1
        run_scales = grid[start:end + 1]
        if len(run_scales) >= minimum_length:
            for left in range(len(run_scales)):
                for right in range(left + minimum_length, len(run_scales) + 1):
                    interval = run_scales[left:right]
                    gaps = [
                        abs(float(by_estimator["Multi"][value]["E_reverse"] -
                                  by_estimator["Single"][value]["E_reverse"]))
                        for value in interval
                    ]
                    median_gap = float(np.median(gaps))
                    candidates.append({
                        "start": interval[0], "end": interval[-1],
                        "scale_count": len(interval),
                        "ordering": "Multi>Single" if sign > 0 else "Multi<Single",
                        "median_absolute_gap": median_gap,
                        "delta_query": delta,
                        "ratio_delta_to_median_gap": delta / median_gap if median_gap > 0 else float("inf"),
                        "passes_delta_rule": delta <= 0.5 * median_gap,
                    })
        start = end + 1
    passing = [row for row in candidates if row["passes_delta_rule"]]
    best_pool = passing or candidates
    best = None
    if best_pool:
        best = min(
            best_pool,
            key=lambda row: (
                row["ratio_delta_to_median_gap"], -row["scale_count"], row["start"]
            ),
        )
    return {
        "stage1_pass": bool(passing),
        "stage2_required": not bool(passing),
        "minimum_consecutive_scales": minimum_length,
        "delta_query": delta,
        "qualifying_interval": best if passing else None,
        "best_nonqualifying_interval": best if not passing else None,
        "candidate_interval_count": len(candidates),
        "passing_interval_count": len(passing),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--base-packet-root", type=Path, required=True)
    parser.add_argument("--bridge-packet-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    design_path = ROOT / "config.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    expected_config_hash = (ROOT / "FROZEN_CONFIG_SHA256.txt").read_text(encoding="utf-8").split()[0]
    actual_config_hash = sha256(design_path)
    if actual_config_hash != expected_config_hash:
        raise ValueError("frozen config hash mismatch")
    base_config, _ = load_dryy_config("e3")
    geometry_map = {g.key: g for g in geometries_from_config(base_config)}
    reference_count = int(design["reference_count"])
    query_seed = int(base_config["query_seed"])
    failure_rate_max = float(design["reliability"]["maximum_query_failure_rate"])
    minimum_length = int(design["reliability"]["minimum_consecutive_common_reliable_scales"])
    all_rows, provenance = [], []
    for geometry_key, specification in design["geometries"].items():
        geometry = geometry_map[geometry_key]
        target_count = int(specification["query_count_stages"][0])
        expected_queries = geometry.query_grid(target_count, query_seed)
        reference = geometry.query_grid(reference_count, query_seed + REFERENCE_SEED_OFFSET)
        rows, record = analyze_stsp(
            args.result_root, geometry_key, geometry, reference, expected_queries,
            failure_rate_max,
        )
        all_rows.extend(rows)
        provenance.append(record)
        for method in ("DR", "YY"):
            rows, record = analyze_dryy(
                args.result_root, args.base_packet_root, args.bridge_packet_root,
                method, geometry_key,
                geometry, reference, expected_queries, failure_rate_max,
            )
            all_rows.extend(rows)
            provenance.append(record)

    expected_rows = len(METHODS) * len(design["geometries"]) * len(ESTIMATORS) * 28
    failures = []
    if len(all_rows) != expected_rows:
        failures.append(f"expected {expected_rows} rows, found {len(all_rows)}")
    identities = [(row["method"], row["geometry"], row["estimator"], row["scale"]) for row in all_rows]
    if len(identities) != len(set(identities)):
        failures.append("duplicate method/geometry/estimator/scale rows")
    if max(row["query_grid_mismatch"] for row in all_rows) > 1e-14:
        failures.append("query-grid mismatch exceeds 1e-14")
    decisions = []
    for method in METHODS:
        for geometry_key in design["geometries"]:
            subset = [row for row in all_rows if row["method"] == method and row["geometry"] == geometry_key]
            decision = stage2_decision(subset, minimum_length)
            decisions.append({"method": method, "geometry": geometry_key, **decision})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "e3_erev_stage1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "schema_version": "E3_EREV_HIGH_DENSITY_STAGE1_AUDIT_V1",
        "evidence_class": design["evidence_class"],
        "config_sha256": actual_config_hash,
        "reference_count": reference_count,
        "reference_seed_offset": REFERENCE_SEED_OFFSET,
        "row_count": len(all_rows),
        "expected_row_count": expected_rows,
        "packet_provenance": provenance,
        "stage2_decisions": decisions,
        "failures": failures,
    }
    report_path = args.output_dir / "E3_EREV_STAGE1_AUDIT.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_path = args.output_dir / "SHA256SUMS.txt"
    manifest_path.write_text(
        f"{sha256(csv_path)}  {csv_path.name}\n{sha256(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "csv": str(csv_path),
        "audit": str(report_path),
        "stage2_required": [
            f"{row['method']}/{row['geometry']}" for row in decisions if row["stage2_required"]
        ],
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
