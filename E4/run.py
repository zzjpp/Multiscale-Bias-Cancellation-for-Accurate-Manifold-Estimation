from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from e4.config import CONFIG  # noqa: E402
from e4.experiment import run_repeat  # noqa: E402
from e4.geometry import arc_table, gamma  # noqa: E402


# The experiment itself uses an explicit PCG64DXSM generator per repeat.
# These declarations additionally pin any accidental global-RNG/hash use.
os.environ.setdefault("PYTHONHASHSEED", str(CONFIG["master_seed"]))
np.random.seed(int(CONFIG["master_seed"]))


METHOD_MAP = {"Oracle": "TNF_Multi", "ENF": "ENF_Multi", "NN": "NN_Multi"}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(packet: dict) -> dict:
    result = packet["result"]
    methods = {}
    for public_name, internal_name in METHOD_MAP.items():
        metric = result["metrics"][internal_name]
        methods[public_name] = {
            key: metric[key]
            for key in ("E_avg", "E_infty", "query_failure_count", "query_failure_rate", "status")
        }
    return {
        "repeat": packet["repeat"],
        "seed_lineage": packet["seed_lineage"],
        "coefficient_record": packet["coefficient_record"],
        "methods": methods,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for method in METHOD_MAP:
        for metric in ("E_avg", "E_infty"):
            values = np.asarray([record["methods"][method][metric] for record in records])
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "count": len(values),
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict], representative: dict, output: Path) -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 11.5}
    )
    table = arc_table()
    truth = gamma(table.invert(np.linspace(0, 1, 1200, endpoint=False)))
    result = representative["result"]
    fig = plt.figure(figsize=(10.5, 3.65), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2)
    ax0.plot(truth[:, 0], truth[:, 1], truth[:, 2], color="0.35", lw=1.4, label="True manifold")
    ax0.plot(*result["ENF"]["y_E"].T, "o-", color="#3B73AF", ms=3.2, lw=1.25, label="ENF")
    ax0.plot(*result["NN"]["y_NN"].T, "s-", color="#D55E00", ms=3.0, lw=1.15, label="NN")
    ax0.set_xlabel("$x_1$"); ax0.set_ylabel("$x_2$"); ax0.set_zlabel("$x_3$")
    ax0.view_init(elev=23, azim=-58)
    ax0.grid(False)
    ax0.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    positions = np.asarray([0.0, 1.0])
    width = 0.22
    colors = {"Oracle": "#666666", "ENF": "#3B73AF", "NN": "#D55E00"}
    offsets = {"Oracle": -width, "ENF": 0.0, "NN": width}
    for method in METHOD_MAP:
        selected = [next(row for row in summary if row["method"] == method and row["metric"] == metric)
                    for metric in ("E_avg", "E_infty")]
        median = np.asarray([row["median"] for row in selected])
        error = np.asarray([[row["median"] - row["q1"] for row in selected],
                            [row["q3"] - row["median"] for row in selected]])
        ax1.bar(positions + offsets[method], median, width=width, color=colors[method], alpha=0.9,
                yerr=error, capsize=3, error_kw={"lw": 1.0}, label=method)
    ax1.set_xticks(positions, [r"$E_{\mathrm{avg}}$", r"$E_{\infty}$"])
    ax1.set_ylabel("Error")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.grid(False)
    ax1.legend(frameon=False, loc="upper left")
    fig.savefig(output / "e4_enf_vs_nn_equal_density.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "e4_enf_vs_nn_equal_density.svg", bbox_inches="tight")
    plt.close(fig)


def run(output: Path, smoke: bool) -> None:
    repeats = [CONFIG["aggregation"]["representative_repeat"]] if smoke else CONFIG["heldout_repeats"]
    temporary = Path(tempfile.mkdtemp(prefix="e4-build-", dir=output.parent))
    try:
        (temporary / "packets").mkdir()
        records, representative = [], None
        for repeat in repeats:
            full = run_repeat(int(repeat), smoke=smoke)
            record = compact(full)
            records.append(record)
            (temporary / "packets" / f"repeat_{repeat:02d}.json").write_text(
                json.dumps(plain(record), indent=2, sort_keys=True), encoding="utf-8"
            )
            if repeat == CONFIG["aggregation"]["representative_repeat"]:
                representative = full
        if representative is None:
            raise RuntimeError("representative repeat was not evaluated")
        if any(record["methods"][method]["query_failure_count"] != 0
               for record in records for method in METHOD_MAP):
            raise RuntimeError("at least one query failed; all results were retained in the temporary build")
        summary = summarize(records)
        write_csv(temporary / "summary.csv", summary)
        plot(summary, representative, temporary)
        audit = {
            "status": "PASS",
            "smoke": smoke,
            "repeat_count": len(records),
            "query_count": 4 if smoke else CONFIG["query_count"],
            "nn_candidate_count": 4 if smoke else CONFIG["nn_candidate_count"],
            "all_queries_complete": True,
            "config_sha256": sha256(ROOT / "config.json"),
        }
        (temporary / "AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
    except Exception:
        print(f"Incomplete build retained at: {temporary}", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--smoke", action="store_true", help="run one tiny end-to-end cell")
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.smoke)
    print(f"Wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
