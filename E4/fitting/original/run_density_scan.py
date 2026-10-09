from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time


for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from e4.coefficients import two_scale_weights  # noqa: E402
from e4.config import CONFIG  # noqa: E402
from e4.experiment import parent_sample  # noqa: E402
from e4.geometry import gamma, normal_basis, normal_projector, project_to_curve  # noqa: E402
from e4.operator import WeightedSTSP  # noqa: E402
from e4.pipeline import (  # noqa: E402
    build_candidate_library_at_positions,
    fitting_metrics,
    match_nn,
    run_main,
)
from e4.solver import solve_fiber  # noqa: E402


SCAN_CONFIG_PATH = ROOT / "density_scan_config.json"
SCAN = json.loads(SCAN_CONFIG_PATH.read_text(encoding="utf-8"))
DATA_HASH_PATH = ROOT / "data_hashes_n200000.json"
EXPECTED_DATA_HASHES = json.loads(DATA_HASH_PATH.read_text(encoding="utf-8"))
os.environ.setdefault("PYTHONHASHSEED", str(SCAN["master_seed"]))
np.random.seed(int(SCAN["master_seed"]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def nested_candidate_positions(count: int) -> np.ndarray:
    counts = [int(value) for value in SCAN["candidate_counts"]]
    count = int(count)
    if count not in counts:
        raise ValueError(f"candidate count {count} is outside the frozen grid")
    base = int(SCAN["base_candidate_count"])
    maximum = int(SCAN["max_candidate_count"])
    if maximum % base or maximum & (maximum - 1) or base & (base - 1):
        raise ValueError("base and maximum candidate counts must be nested powers of two")
    if not base <= count <= maximum:
        raise ValueError("candidate count is outside the nested hierarchy")
    offset = float(SCAN["base_candidate_offset_numerator"]) / base
    # Start with the released 32-point grid. At every dyadic refinement, insert
    # new midpoints in bit-reversal order. Every prefix remains nested and
    # spatially balanced, while completed levels reproduce the original grids.
    selected = list(range(0, maximum, maximum // base))
    level = base * 2
    while len(selected) < count:
        stride = maximum // level
        new_indices = list(range(stride, maximum, 2 * stride))
        bits = int(np.log2(len(new_indices)))
        order = sorted(range(len(new_indices)), key=lambda value: int(f"{value:0{bits}b}"[::-1], 2))
        take = min(count - len(selected), len(new_indices))
        selected.extend(new_indices[index] for index in order[:take])
        level *= 2
    return np.mod(offset + np.asarray(selected, dtype=np.float64) / maximum, 1.0)


def validate_nested_positions() -> dict:
    counts = [int(value) for value in SCAN["candidate_counts"]]
    grids = {count: nested_candidate_positions(count) for count in counts}
    checks = []
    for coarse, fine in zip(counts, counts[1:]):
        coarse_keys = {np.float64(value).tobytes() for value in grids[coarse]}
        fine_keys = {np.float64(value).tobytes() for value in grids[fine]}
        missing = len(coarse_keys - fine_keys)
        checks.append({"coarse": coarse, "fine": fine, "missing_coarse_points": missing})
        if missing:
            raise RuntimeError("candidate grids are not exactly nested")
    old_m32 = np.mod(
        np.arange(32, dtype=float) / 32
        + 0.37 / int(SCAN["query_count"])
        + float(CONFIG["nn_phase"]) / 32,
        1.0,
    )
    baseline_difference = float(np.max(np.abs(grids[32] - old_m32)))
    if baseline_difference > 2e-16:
        raise RuntimeError("the frozen 32-candidate E4 grid was not preserved")
    return {"status": "PASS", "pairs": checks, "baseline_m32_max_abs_difference": baseline_difference}


def backend_class(name: str):
    if name == "cpu":
        return WeightedSTSP, (lambda: None), {"backend": "numpy-scipy", "dtype": "float64"}
    if name == "gpu":
        from e4.gpu_operator import GPUWeightedSTSP, device_metadata, synchronize

        return GPUWeightedSTSP, synchronize, device_metadata()
    raise ValueError(name)


def diagnostic(point, oracle, theta_star) -> tuple[float, float]:
    if not (np.isfinite(point).all() and np.isfinite(oracle).all()):
        return np.nan, np.nan
    difference = np.asarray(point) - np.asarray(oracle)
    tangent_projector = np.eye(3) - normal_projector(theta_star)
    return float(np.linalg.norm(difference)), float(np.linalg.norm(tangent_projector @ difference))


def metric_record(name: str, points, oracle, theta_star, timing: dict, candidate_count=None, library=None, selected_distance=None) -> dict:
    metrics = fitting_metrics(points)
    complete = metrics["query_failure_count"] == 0
    corr, tan = [], []
    for point, target, theta in zip(points, oracle, theta_star):
        dc, dt = diagnostic(point, target, theta)
        corr.append(dc); tan.append(dt)
    record = {
        "method": name,
        "candidate_count": candidate_count,
        "E_avg": metrics["E_avg"],
        "E_infty": metrics["E_infty"],
        "query_failure_count": metrics["query_failure_count"],
        "query_failure_rate": metrics["query_failure_rate"],
        "complete_repeat": bool(complete),
        "status": metrics["status"],
        "D_corr": float(np.nanmean(corr)) if np.isfinite(corr).any() else np.nan,
        "D_tan": float(np.nanmean(tan)) if np.isfinite(tan).any() else np.nan,
        "timing": timing,
    }
    if library is not None:
        record.update({
            "M_requested": int(library["M_match"]),
            "M_valid": int(library["success_count"]),
            "candidate_failure_count": int(library["failure_count"]),
            "local_solve_count": int(library["local_solve_count"]),
            "newton_iteration_count": int(library["newton_iteration_count"]),
            "residual_function_evaluation_count": int(library["residual_function_evaluation_count"]),
            "largest_uncovered_arclength_gap": float(library["largest_uncovered_arclength_gap"]),
            "mean_reference_to_candidate_distance": float(np.nanmean(selected_distance)),
        })
    return record


def _warmup(operator_h1, operator_h2, sync) -> dict:
    start = time.perf_counter()
    for operator in (operator_h1, operator_h2):
        solve_fiber(operator, gamma(0.0), normal_basis(0.0), operator.h)
    sync()
    return {"excluded_gpu_warmup_seconds": time.perf_counter() - start}


def load_frozen_parent(repeat: int, n: int, smoke: bool, data_dir: Path | None):
    if smoke or data_dir is None:
        clean, standard_noise, lineage = parent_sample(int(repeat), n)
    else:
        path = data_dir / f"repeat_{int(repeat):02d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen parent input: {path}")
        with np.load(path, allow_pickle=False) as archive:
            clean = np.asarray(archive["clean"], dtype=np.float64)
            standard_noise = np.asarray(archive["standard_noise"], dtype=np.float64)
        if clean.shape != (n, 3) or standard_noise.shape != (n, 3):
            raise RuntimeError(f"bad frozen parent shape in {path}")
        lineage = {
            "algorithm": "PCG64DXSM",
            "experiment_id": int(SCAN["experiment_id"]),
            "repeat": int(repeat),
            "clean": [int(SCAN["master_seed"]), int(SCAN["experiment_id"]), int(repeat), 0],
            "noise": [int(SCAN["master_seed"]), int(SCAN["experiment_id"]), int(repeat), 1],
            "theta_sha256": EXPECTED_DATA_HASHES[str(int(repeat))]["theta_sha256"],
            "clean_sha256": array_sha256(clean),
            "standard_noise_sha256": array_sha256(standard_noise),
            "loaded_from": path.name,
            "file_sha256": sha256(path),
        }
    if not smoke:
        expected = EXPECTED_DATA_HASHES[str(int(repeat))]
        for key, value in expected.items():
            if lineage[key] != value:
                raise RuntimeError(f"data hash mismatch for repeat {repeat}: {key}")
    return clean, standard_noise, lineage


def run_repeat(repeat: int, backend: str, smoke: bool = False, data_dir: Path | None = None) -> dict:
    if int(repeat) not in [int(value) for value in SCAN["heldout_repeats"]]:
        raise ValueError("repeat outside the frozen held-out set")
    validate_nested_positions()
    operator_type, sync, device = backend_class(backend)
    n = 20000 if smoke else int(SCAN["sample_size"])
    query_count = 4 if smoke else int(SCAN["query_count"])
    counts = [32, 128] if smoke else [int(value) for value in SCAN["candidate_counts"]]

    sync(); start = time.perf_counter()
    clean, standard_noise, lineage = load_frozen_parent(int(repeat), n, smoke, data_dir)
    observations = clean + float(SCAN["sigma"]) * standard_noise
    sync(); data_seconds = time.perf_counter() - start

    sync(); start = time.perf_counter()
    operator_h1 = operator_type(observations, float(SCAN["h1"]), label=f"{backend}-h1")
    operator_h2 = operator_type(observations, float(SCAN["h2"]), label=f"{backend}-h2")
    sync(); index_seconds = time.perf_counter() - start

    start = time.perf_counter()
    coefficient_record = two_scale_weights(float(SCAN["h1"]), float(SCAN["h2"]), float(SCAN["sigma"]))
    coefficient_seconds = time.perf_counter() - start
    warmup = _warmup(operator_h1, operator_h2, sync)

    shared_result = run_main(
        operator_h1,
        operator_h2,
        float(SCAN["h2"]),
        coefficient_record["weights"],
        N_q=query_count,
        M_match=1,
        phi=float(CONFIG["nn_phase"]),
    )
    runtime = shared_result["runtime"]
    shared_seconds = data_seconds + index_seconds + coefficient_seconds + float(runtime["reference"])
    w1, w2 = np.asarray(coefficient_record["weights"], dtype=float)
    z = np.asarray(shared_result["common_reference"]["z"], dtype=float)
    oracle = np.asarray(shared_result["TNF"]["Z_T2"], dtype=float)
    enf = np.asarray(shared_result["ENF"]["Z_E2"], dtype=float)
    theta_star = np.asarray([project_to_curve(point)["theta"] if np.isfinite(point).all() else np.nan for point in z])

    start = time.perf_counter(); oracle_fit = w1 * z + w2 * oracle; oracle_combine = time.perf_counter() - start
    start = time.perf_counter(); enf_fit = w1 * z + w2 * enf; enf_combine = time.perf_counter() - start
    oracle_timing = {
        "shared_seconds": shared_seconds,
        "correspondence_seconds": float(runtime["TNF_target"]),
        "combine_seconds": oracle_combine,
        "full_seconds": shared_seconds + float(runtime["TNF_target"]) + oracle_combine,
    }
    enf_correspondence = float(runtime["ENF_PCA"] + runtime["ENF_target"])
    enf_timing = {
        "shared_seconds": shared_seconds,
        "normal_estimation_seconds": float(runtime["ENF_PCA"]),
        "intersection_seconds": float(runtime["ENF_target"]),
        "correspondence_seconds": enf_correspondence,
        "combine_seconds": enf_combine,
        "full_seconds": shared_seconds + enf_correspondence + enf_combine,
    }
    records = [
        metric_record("Oracle", oracle_fit, oracle, theta_star, oracle_timing),
        metric_record("ENF", enf_fit, oracle, theta_star, enf_timing),
    ]
    per_query = []
    for method, second, fitted in (("Oracle", oracle, oracle_fit), ("ENF", enf, enf_fit)):
        for j in range(query_count):
            dc, dt = diagnostic(second[j], oracle[j], theta_star[j])
            per_query.append({
                "method": method, "candidate_count": None, "query_id": j,
                "reference": z[j], "second_scale": second[j], "extrapolated": fitted[j],
                "oracle_second_scale": oracle[j], "D_corr": dc, "D_tan": dt,
                "success": bool(np.isfinite(fitted[j]).all()),
            })

    # Predeclared cyclic rotation prevents every large-M timing from always
    # occupying the same thermal/cache position.  It never changes a grid.
    repeat_index = [int(value) for value in SCAN["heldout_repeats"]].index(int(repeat))
    timing_order = list(np.roll(np.asarray(counts, dtype=int), -(repeat_index % len(counts))))
    by_count = {}
    candidate_rows = []
    for count in timing_order:
        print(f"repeat={repeat} start_M={int(count)}", flush=True)
        positions = nested_candidate_positions(int(count))
        sync(); start = time.perf_counter()
        library = build_candidate_library_at_positions(operator_h2, positions)
        sync(); candidate_seconds = time.perf_counter() - start
        sync(); start = time.perf_counter()
        valid_reference = np.isfinite(z).all(axis=1)
        raw_match = match_nn(z[valid_reference], library)
        match = dict(raw_match)
        match["points"] = np.full_like(z, np.nan)
        match["points"][valid_reference] = raw_match["points"]
        match["distances"] = np.full(len(z), np.nan)
        match["distances"][valid_reference] = raw_match["distances"]
        match["selected_candidate_id"] = np.full(len(z), -1, dtype=int)
        match["selected_candidate_id"][valid_reference] = raw_match["selected_candidate_id"]
        sync(); index_search_seconds = time.perf_counter() - start
        second = np.asarray(match["points"], dtype=float)
        start = time.perf_counter(); fitted = w1 * z + w2 * second; combine_seconds = time.perf_counter() - start
        timing = {
            "shared_seconds": shared_seconds,
            "candidate_fit_seconds": candidate_seconds,
            "index_search_seconds": index_search_seconds,
            "combine_seconds": combine_seconds,
            "correspondence_seconds": candidate_seconds + index_search_seconds,
            "precomputed_seconds": index_search_seconds + combine_seconds,
            "full_seconds": shared_seconds + candidate_seconds + index_search_seconds + combine_seconds,
        }
        record = metric_record(
            "NN", fitted, oracle, theta_star, timing,
            candidate_count=int(count), library=library,
            selected_distance=np.asarray(match["distances"], dtype=float),
        )
        by_count[int(count)] = record
        print(
            f"repeat={repeat} done_M={int(count)} "
            f"candidate_seconds={candidate_seconds:.6f} "
            f"search_seconds={index_search_seconds:.6f} "
            f"M_valid={library['success_count']}/{int(count)}",
            flush=True,
        )
        for j in range(query_count):
            dc, dt = diagnostic(second[j], oracle[j], theta_star[j])
            per_query.append({
                "method": "NN", "candidate_count": int(count), "query_id": j,
                "reference": z[j], "second_scale": second[j], "extrapolated": fitted[j],
                "oracle_second_scale": oracle[j], "D_corr": dc, "D_tan": dt,
                "selected_candidate_id": int(match["selected_candidate_id"][j]),
                "reference_to_candidate_distance": float(match["distances"][j]),
                "success": bool(np.isfinite(fitted[j]).all()),
            })
        for idx in range(int(count)):
            candidate_rows.append({
                "candidate_count": int(count), "candidate_id": idx,
                "normalized_arclength": float(library["candidate_normalized_arclength"][idx]),
                "theta": float(library["candidate_theta"][idx]),
                "status": library["candidate_status"][idx],
                "fitted_coordinate": library["candidate_fitted_coordinates"][idx],
                "iterations": int(library["candidate_iterations"][idx]),
                "function_evaluations": int(library["candidate_function_evaluations"][idx]),
            })
    records.extend(by_count[count] for count in counts)

    monotonic_checks = []
    previous = None
    tolerance = float(SCAN["reliability"]["monotonicity_absolute_tolerance"])
    for count in counts:
        distances = np.asarray([
            row["reference_to_candidate_distance"] for row in per_query
            if row["method"] == "NN" and row["candidate_count"] == count
        ])
        if previous is not None:
            violations = np.flatnonzero(distances > previous + tolerance)
            monotonic_checks.append({"candidate_count": count, "violation_count": int(len(violations)), "violation_query_ids": violations})
            if len(violations):
                raise RuntimeError("nearest-candidate distance increased after nested refinement")
        previous = distances

    return {
        "schema_version": SCAN["schema_version"],
        "status": "PASS",
        "repeat": int(repeat),
        "smoke": bool(smoke),
        "backend": backend,
        "device": device,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "seed_lineage": lineage,
        "sample_size": n,
        "query_count": query_count,
        "candidate_counts": counts,
        "timing_order": [int(value) for value in timing_order],
        "config_sha256": sha256(SCAN_CONFIG_PATH),
        "base_config_sha256": sha256(ROOT / "config.json"),
        "data_hash_manifest_sha256": sha256(DATA_HASH_PATH),
        "coefficient_record": coefficient_record,
        "warmup": warmup,
        "shared_timing": {"data_seconds": data_seconds, "index_seconds": index_seconds, "coefficient_seconds": coefficient_seconds, "reference_seconds": float(runtime["reference"]), "shared_seconds": shared_seconds},
        "records": records,
        "per_query": per_query,
        "candidate_validity": candidate_rows,
        "nested_distance_checks": monotonic_checks,
    }


def write_packet(packet: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plain(packet), sort_keys=True, separators=(",", ":")).encode("utf-8")
    with gzip.GzipFile(filename=str(path), mode="wb", compresslevel=6, mtime=0) as handle:
        handle.write(payload)


def read_packet(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def quantiles(values) -> dict:
    data = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(data):
        return {"count": 0, "median": np.nan, "q1": np.nan, "q3": np.nan}
    return {"count": int(len(data)), "median": float(np.median(data)), "q1": float(np.quantile(data, 0.25)), "q3": float(np.quantile(data, 0.75))}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def plot_accuracy(summary_rows: list[dict], cost_rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    counts = np.asarray(SCAN["candidate_counts"], dtype=float)
    positions = np.arange(len(counts), dtype=float)
    # Larger typography for the full-width, three-panel manuscript figure.
    # These settings affect presentation only, not the stored numerical tables.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "axes.labelsize": 16, "legend.fontsize": 15,
        "xtick.labelsize": 12, "ytick.labelsize": 13,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.0), constrained_layout=True)
    for ax, metric, label in zip(axes[:2], ("E_avg", "E_infty"), (r"Average error $E_{\mathrm{avg}}$", r"Maximum error $E_\infty$")):
        nn = [next(row for row in summary_rows if row["method"] == "NN" and row["candidate_count"] == int(count) and row["metric"] == metric) for count in counts]
        median = np.asarray([row["median"] for row in nn]); q1 = np.asarray([row["q1"] for row in nn]); q3 = np.asarray([row["q3"] for row in nn])
        ax.plot(positions, median, "o-", color="#D55E00", lw=1.6, ms=4.5, label="NN")
        ax.fill_between(positions, q1, q3, color="#D55E00", alpha=0.16, linewidth=0)
        for method, color, style in (("ENF", "#3B73AF", "-"), ("Oracle", "#666666", "--")):
            row = next(row for row in summary_rows if row["method"] == method and row["metric"] == metric)
            ax.axhline(row["median"], color=color, ls=style, lw=1.5, label=method)
            ax.axhspan(row["q1"], row["q3"], color=color, alpha=0.10, linewidth=0)
        ax.set_xticks(positions, [str(int(value)) for value in counts], rotation=55, ha="right", fontsize=12)
        ax.set_xlabel(r"NN candidate count $M_{\mathrm{match}}$"); ax.set_ylabel(label)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)
    nn_cost = {int(row["candidate_count"]): row for row in cost_rows if row["method"] == "NN"}
    enf_cost = next(row for row in cost_rows if row["method"] == "ENF")
    median = np.asarray([nn_cost[int(count)]["full_seconds_median"] for count in counts])
    q1 = np.asarray([nn_cost[int(count)]["full_seconds_q1"] for count in counts])
    q3 = np.asarray([nn_cost[int(count)]["full_seconds_q3"] for count in counts])
    axes[2].plot(positions, median, "o-", color="#D55E00", lw=1.6, ms=4.5, label="NN")
    axes[2].fill_between(positions, q1, q3, color="#D55E00", alpha=0.16, linewidth=0)
    axes[2].axhline(enf_cost["full_seconds_median"], color="#3B73AF", lw=1.5, label="ENF")
    axes[2].axhspan(enf_cost["full_seconds_q1"], enf_cost["full_seconds_q3"], color="#3B73AF", alpha=0.10, linewidth=0)
    axes[2].set_yscale("log")
    axes[2].set_xticks(positions, [str(int(value)) for value in counts], rotation=55, ha="right", fontsize=12)
    axes[2].set_xlabel(r"NN candidate count $M_{\mathrm{match}}$")
    axes[2].set_ylabel("Complete computation\ntime (s)")
    axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False); axes[2].grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.savefig(output / "e4_nn_density_accuracy.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "e4_nn_density_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cost(cost_rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    counts = np.asarray(SCAN["candidate_counts"], dtype=float)
    positions = np.arange(len(counts), dtype=float)
    nn = {int(row["candidate_count"]): row for row in cost_rows if row["method"] == "NN"}
    enf = next(row for row in cost_rows if row["method"] == "ENF")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 11.5})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), constrained_layout=True)
    for ax, stem, label in zip(
        axes,
        ("full_seconds", "correspondence_seconds"),
        ("Complete computation time (s)", "Correspondence-stage time (s)"),
    ):
        median = np.asarray([nn[int(count)][f"{stem}_median"] for count in counts])
        q1 = np.asarray([nn[int(count)][f"{stem}_q1"] for count in counts])
        q3 = np.asarray([nn[int(count)][f"{stem}_q3"] for count in counts])
        ax.plot(positions, median, "o-", color="#D55E00", lw=1.6, ms=4.5, label="NN")
        ax.fill_between(positions, q1, q3, color="#D55E00", alpha=0.16, linewidth=0)
        ax.axhline(enf[f"{stem}_median"], color="#3B73AF", lw=1.5, label="ENF")
        ax.axhspan(enf[f"{stem}_q1"], enf[f"{stem}_q3"], color="#3B73AF", alpha=0.10, linewidth=0)
        ax.set_yscale("log")
        ax.set_xticks(positions, [str(int(value)) for value in counts], rotation=35, ha="right", fontsize=9)
        ax.set_xlabel(r"NN candidate count $M_{\mathrm{match}}$"); ax.set_ylabel(label)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.savefig(output / "e4_nn_density_runtime.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "e4_nn_density_runtime.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate(packet_dir: Path, output: Path) -> None:
    repeats = [int(value) for value in SCAN["heldout_repeats"]]
    packets = [read_packet(packet_dir / f"repeat_{repeat:02d}.json.gz") for repeat in repeats]
    if any(packet["status"] != "PASS" for packet in packets):
        raise RuntimeError("not all packets passed")
    if len({json.dumps(packet["device"], sort_keys=True) for packet in packets}) != 1:
        raise RuntimeError("timing packets were not produced on one identical device")
    output.mkdir(parents=True, exist_ok=True)
    repeat_rows, query_rows, candidate_rows = [], [], []
    for packet in packets:
        for record in packet["records"]:
            row = {key: record.get(key) for key in ("method", "candidate_count", "E_avg", "E_infty", "D_corr", "D_tan", "query_failure_count", "query_failure_rate", "complete_repeat", "status", "M_requested", "M_valid", "candidate_failure_count", "local_solve_count", "newton_iteration_count", "residual_function_evaluation_count")}
            row["repeat"] = packet["repeat"]
            for key, value in record["timing"].items(): row[f"time_{key}"] = value
            repeat_rows.append(row)
        for row in packet["per_query"]:
            query_rows.append({"repeat": packet["repeat"], **row})
        for row in packet["candidate_validity"]:
            candidate_rows.append({"repeat": packet["repeat"], **row})
    write_csv(output / "per_repeat_metrics.csv", repeat_rows)
    with gzip.GzipFile(filename=str(output / "per_query.csv.gz"), mode="wb", compresslevel=6, mtime=0) as compressed:
        handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        flat = []
        for row in query_rows:
            item = dict(row)
            for key in ("reference", "second_scale", "extrapolated", "oracle_second_scale"):
                values = item.pop(key)
                for axis, value in zip("xyz", values): item[f"{key}_{axis}"] = value
            flat.append(item)
        fields = sorted({key for row in flat for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(flat)
        handle.flush()
    with gzip.GzipFile(filename=str(output / "candidate_validity.csv.gz"), mode="wb", compresslevel=6, mtime=0) as compressed:
        handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        flat = []
        for row in candidate_rows:
            item = dict(row); values = item.pop("fitted_coordinate")
            for axis, value in zip("xyz", values): item[f"fitted_{axis}"] = value
            flat.append(item)
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        handle.flush()

    summary = []
    for method in ("Oracle", "ENF"):
        selected = [row for row in repeat_rows if row["method"] == method]
        for metric in ("E_avg", "E_infty"):
            complete = [row[metric] for row in selected if row["complete_repeat"]]
            summary.append({"method": method, "candidate_count": "", "metric": metric, "complete_repeat_count": len(complete), **quantiles(complete)})
    for count in SCAN["candidate_counts"]:
        selected = [row for row in repeat_rows if row["method"] == "NN" and row["candidate_count"] == count]
        for metric in ("E_avg", "E_infty"):
            complete = [row[metric] for row in selected if row["complete_repeat"]]
            summary.append({"method": "NN", "candidate_count": count, "metric": metric, "complete_repeat_count": len(complete), **quantiles(complete)})
    write_csv(output / "accuracy_summary.csv", summary)

    cost_rows = []
    for method, count in [("ENF", None)] + [("NN", value) for value in SCAN["candidate_counts"]]:
        selected = [row for row in repeat_rows if row["method"] == method and row["candidate_count"] == count]
        row = {"method": method, "candidate_count": "" if count is None else count}
        for source, label in (("D_corr", "D_corr"), ("D_tan", "D_tan"), ("time_correspondence_seconds", "correspondence_seconds"), ("time_full_seconds", "full_seconds"), ("time_precomputed_seconds", "precomputed_seconds")):
            values = [item[source] for item in selected if item.get(source) is not None]
            stats = quantiles(values)
            for stat, value in stats.items(): row[f"{label}_{stat}"] = value
        row["complete_repeat_count"] = sum(bool(item["complete_repeat"]) for item in selected)
        row["total_query_failures"] = sum(int(item["query_failure_count"]) for item in selected)
        row["M_valid_median"] = "" if method == "ENF" else float(np.median([item["M_valid"] for item in selected]))
        row["local_solve_count_median"] = "" if method == "ENF" else float(np.median([item["local_solve_count"] for item in selected]))
        row["residual_function_evaluation_count_median"] = "" if method == "ENF" else float(np.median([item["residual_function_evaluation_count"] for item in selected]))
        cost_rows.append(row)
    write_csv(output / "accuracy_cost_table.csv", cost_rows)
    plot_accuracy(summary, cost_rows, output)

    all_nested = all(check["violation_count"] == 0 for packet in packets for check in packet["nested_distance_checks"])
    audit = {
        "status": "PASS" if all_nested else "FAIL",
        "repeat_count": len(packets),
        "repeats": repeats,
        "sample_size": int(SCAN["sample_size"]),
        "query_count": int(SCAN["query_count"]),
        "candidate_counts": SCAN["candidate_counts"],
        "nested_grid_gate": validate_nested_positions(),
        "nearest_candidate_distance_nonincreasing": all_nested,
        "fitting_error_monotonicity_required": False,
        "all_candidate_failures_retained": True,
        "all_query_failures_retained": True,
        "device": packets[0]["device"],
        "thread_environment": packets[0]["thread_environment"],
        "config_sha256": sha256(SCAN_CONFIG_PATH),
        "base_config_sha256": sha256(ROOT / "config.json"),
        "packet_sha256": {path.name: sha256(path) for path in sorted(packet_dir.glob("*.json.gz"))},
    }
    (output / "AUDIT.json").write_text(json.dumps(plain(audit), indent=2, sort_keys=True), encoding="utf-8")
    if audit["status"] != "PASS": raise RuntimeError("aggregate audit failed")
    names = ["accuracy_summary.csv", "accuracy_cost_table.csv", "per_repeat_metrics.csv", "per_query.csv.gz", "candidate_validity.csv.gz", "e4_nn_density_accuracy.png", "e4_nn_density_accuracy.pdf", "AUDIT.json"]
    (output / "SHA256SUMS.txt").write_text("\n".join(f"{sha256(output / name)}  {name}" for name in names) + "\n", encoding="utf-8")


def equivalence_gate(output: Path) -> None:
    gate = SCAN["gpu_equivalence"]
    repeat = int(gate["repeat"]); n = int(gate["sample_size"])
    clean, standard_noise, lineage = parent_sample(repeat, n)
    observations = clean + float(SCAN["sigma"]) * standard_noise
    weights = two_scale_weights(float(SCAN["h1"]), float(SCAN["h2"]), float(SCAN["sigma"]))["weights"]
    from e4.gpu_operator import GPUWeightedSTSP, device_metadata, synchronize
    results = {}
    for name, cls in (("cpu", WeightedSTSP), ("gpu", GPUWeightedSTSP)):
        h1 = cls(observations, float(SCAN["h1"]), label=f"gate-{name}-h1")
        h2 = cls(observations, float(SCAN["h2"]), label=f"gate-{name}-h2")
        if name == "gpu": _warmup(h1, h2, synchronize)
        results[name] = run_main(h1, h2, float(SCAN["h2"]), weights, N_q=int(gate["query_count"]), M_match=int(gate["candidate_count"]), phi=float(CONFIG["nn_phase"]))
    comparisons = []
    for label, path in (("reference", ("common_reference", "z")), ("oracle", ("TNF", "Z_T2")), ("enf", ("ENF", "Z_E2")), ("nn", ("NN", "Z_NN2"))):
        cpu = np.asarray(results["cpu"][path[0]][path[1]], dtype=float); gpu = np.asarray(results["gpu"][path[0]][path[1]], dtype=float)
        difference = float(np.nanmax(np.abs(cpu - gpu)))
        comparisons.append({"name": label, "max_abs_difference": difference, "pass": difference <= float(gate["point_absolute_tolerance"])})
    statuses_equal = all(results["cpu"][key]["status"] == results["gpu"][key]["status"] for key in ("TNF", "ENF")) and results["cpu"]["NN"]["library"]["candidate_status"] == results["gpu"]["NN"]["library"]["candidate_status"]
    report = {"status": "PASS" if statuses_equal and all(row["pass"] for row in comparisons) else "FAIL", "seed_lineage": lineage, "device": device_metadata(), "comparisons": comparisons, "statuses_equal": statuses_equal, "config_sha256": sha256(SCAN_CONFIG_PATH)}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if report["status"] != "PASS": raise RuntimeError("CPU/GPU equivalence gate failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="E4 NN candidate-density accuracy-cost experiment")
    parser.add_argument("--mode", choices=("grid-gate", "gpu-gate", "smoke", "repeat", "production", "aggregate"), required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--data-dir", type=Path, help="directory containing frozen repeat_XX.npz inputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "density_scan_outputs")
    args = parser.parse_args(); output = args.output_dir.resolve(); packet_dir = output / "packets"
    if args.mode == "grid-gate":
        output.mkdir(parents=True, exist_ok=True); (output / "NESTED_GRID_GATE.json").write_text(json.dumps(validate_nested_positions(), indent=2), encoding="utf-8")
    elif args.mode == "gpu-gate": equivalence_gate(output / "CPU_GPU_EQUIVALENCE.json")
    elif args.mode == "smoke":
        packet = run_repeat(int(SCAN["heldout_repeats"][0]), args.backend, smoke=True, data_dir=args.data_dir); write_packet(packet, output / "smoke.json.gz")
    elif args.mode == "repeat":
        if args.repeat is None: parser.error("--repeat is required for repeat mode")
        write_packet(run_repeat(args.repeat, args.backend, data_dir=args.data_dir), packet_dir / f"repeat_{args.repeat:02d}.json.gz")
    elif args.mode == "production":
        for repeat in SCAN["heldout_repeats"]:
            path = packet_dir / f"repeat_{int(repeat):02d}.json.gz"
            if not path.exists(): write_packet(run_repeat(int(repeat), args.backend, data_dir=args.data_dir), path)
        aggregate(packet_dir, output / "formal")
    elif args.mode == "aggregate": aggregate(packet_dir, output / "formal")
    print(output)


if __name__ == "__main__":
    main()
