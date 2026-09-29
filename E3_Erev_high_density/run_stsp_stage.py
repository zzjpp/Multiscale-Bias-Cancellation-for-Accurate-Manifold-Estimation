from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
STSP = ROOT.parent / "E3" / "methods" / "stsp"
sys.path.insert(0, str(STSP / "src"))

from e3.config import CONFIG  # noqa: E402
from e3.geometry import BY_KEY  # noqa: E402
import e3.runner as stsp_runner  # noqa: E402


class IncrementalGeometry:
    def __init__(self, base, added_queries: np.ndarray):
        self._base = base
        self._added_queries = np.asarray(added_queries, dtype=float)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def query_grid(self, count: int = 256) -> np.ndarray:
        if int(count) == len(self._added_queries):
            return self._added_queries.copy()
        return self._base.query_grid(int(count))


def complement(geometry, base_count: int, target_count: int) -> np.ndarray:
    base = geometry.query_grid(base_count)
    target = geometry.query_grid(target_count)
    distances = cKDTree(base).query(target, k=1, workers=-1)[0]
    added = target[distances > 1e-12]
    if len(added) != target_count - base_count:
        raise RuntimeError(
            f"expected {target_count - base_count} complementary points, got {len(added)}"
        )
    merged = np.vstack((base, added))
    mismatch = max(
        cKDTree(target).query(merged, k=1, workers=-1)[0].max(),
        cKDTree(merged).query(target, k=1, workers=-1)[0].max(),
    )
    if mismatch > 1e-14:
        raise RuntimeError(f"nested query-grid mismatch {mismatch:.3e}")
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--stage", choices=(1, 2), type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    design = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    specification = design["geometries"].get(args.geometry)
    if specification is None:
        raise ValueError("geometry is outside the frozen high-density design")
    targets = [int(value) for value in specification["query_count_stages"]]
    target_count = targets[args.stage - 1]
    if args.shard_count < 1 or args.shard_index not in range(args.shard_count):
        raise ValueError("invalid shard identity")
    geometry = BY_KEY[args.geometry]
    complete_queries = geometry.query_grid(target_count)
    mode = "full_parent"
    if args.stage == 2:
        base_count = targets[0]
        complete_queries = complement(geometry, base_count, target_count)
        mode = "benchmark"
    shard_indices = np.arange(len(complete_queries), dtype=int)[args.shard_index::args.shard_count]
    shard_queries = complete_queries[shard_indices]
    query_count = len(shard_queries)
    replacement = IncrementalGeometry(geometry, shard_queries)

    original = stsp_runner.BY_KEY[args.geometry]
    stsp_runner.BY_KEY[args.geometry] = replacement
    try:
        result = stsp_runner.run_parent(
            args.geometry,
            0,
            root_value=(
                args.output_root / f"stage{args.stage}" / "stsp_shards" /
                f"s{args.shard_index:02d}of{args.shard_count:02d}"
            ),
            mode=mode,
            bandwidths=list(CONFIG["bandwidths_h2"]),
            query_count=query_count,
            resume=False,
            production_confirmed=False,
            truth_count=int(design["reference_count"]),
        )
    finally:
        stsp_runner.BY_KEY[args.geometry] = original
    print(json.dumps({
        "stage": args.stage,
        "geometry": args.geometry,
        "target_query_count": target_count,
        "evaluated_query_count": query_count,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "result": result,
    }, indent=2))


if __name__ == "__main__":
    main()
