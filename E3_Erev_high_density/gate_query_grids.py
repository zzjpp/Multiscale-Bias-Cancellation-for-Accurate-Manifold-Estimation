from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
E3 = ROOT.parent / "E3"
sys.path.insert(0, str(E3 / "methods" / "stsp" / "src"))
sys.path.insert(0, str(E3 / "methods" / "dr_yy" / "src"))

from e3.config import CONFIG as STSP_CONFIG  # noqa: E402
from e3.geometry import BY_KEY as STSP_GEOMETRIES  # noqa: E402
from dryy.geometry import geometries_from_config  # noqa: E402
from dryy.config import load_config as load_dryy_config  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_mismatch(left: np.ndarray, right: np.ndarray) -> float:
    return float(max(
        cKDTree(left).query(right, k=1, workers=-1)[0].max(),
        cKDTree(right).query(left, k=1, workers=-1)[0].max(),
    ))


def main() -> None:
    config_path = ROOT / "config.json"
    frozen_hash = (ROOT / "FROZEN_CONFIG_SHA256.txt").read_text(encoding="utf-8").split()[0]
    actual_hash = sha256(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dryy_config, _ = load_dryy_config("e3")
    dryy_geometries = {g.key: g for g in geometries_from_config(dryy_config)}
    query_seed = int(dryy_config["query_seed"])
    failures: list[str] = []
    if actual_hash != frozen_hash:
        failures.append("frozen config hash mismatch")
    if query_seed != int(STSP_CONFIG["query_seed"]):
        failures.append("STSP and DR/YY query seeds differ")

    records = []
    for geometry_key, specification in config["geometries"].items():
        stsp_geometry = STSP_GEOMETRIES[geometry_key]
        dryy_geometry = dryy_geometries[geometry_key]
        counts = [1024] + [int(v) for v in specification["query_count_stages"]]
        reference = dryy_geometry.query_grid(int(config["reference_count"]), query_seed + 7919)
        previous = None
        for count in counts:
            stsp_grid = stsp_geometry.query_grid(count)
            dryy_grid = dryy_geometry.query_grid(count, query_seed)
            cross_method_mismatch = float(np.max(np.abs(stsp_grid - dryy_grid)))
            duplicate_pairs = len(cKDTree(dryy_grid).query_pairs(1e-12))
            nesting_mismatch = None if previous is None else set_mismatch(
                previous, dryy_grid[cKDTree(dryy_grid).query(previous, k=1, workers=-1)[1]]
            )
            complement_count = None
            if previous is not None:
                distances = cKDTree(previous).query(dryy_grid, k=1, workers=-1)[0]
                complement_count = int(np.count_nonzero(distances > 1e-12))
                expected = count - len(previous)
                if complement_count != expected:
                    failures.append(
                        f"{geometry_key}/{count}: expected complement {expected}, got {complement_count}"
                    )
            if cross_method_mismatch > 1e-14:
                failures.append(
                    f"{geometry_key}/{count}: STSP vs DR/YY mismatch {cross_method_mismatch:.3e}"
                )
            if duplicate_pairs:
                failures.append(f"{geometry_key}/{count}: duplicate query points")
            if nesting_mismatch is not None and nesting_mismatch > 1e-14:
                failures.append(
                    f"{geometry_key}/{count}: non-nested grid mismatch {nesting_mismatch:.3e}"
                )
            covering_radius = float(cKDTree(dryy_grid).query(reference, k=1, workers=-1)[0].max())
            records.append({
                "geometry": geometry_key,
                "intrinsic_dimension": int(specification["intrinsic_dimension"]),
                "query_count": count,
                "cross_method_max_abs_mismatch": cross_method_mismatch,
                "nesting_mismatch": nesting_mismatch,
                "complement_count_from_previous_stage": complement_count,
                "duplicate_pair_count": duplicate_pairs,
                "reference_count": int(config["reference_count"]),
                "query_covering_radius_delta": covering_radius,
            })
            previous = dryy_grid

    report = {
        "status": "PASS" if not failures else "FAIL",
        "config_sha256": actual_hash,
        "failures": failures,
        "records": records,
    }
    output = ROOT / "QUERY_GRID_GATE.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
