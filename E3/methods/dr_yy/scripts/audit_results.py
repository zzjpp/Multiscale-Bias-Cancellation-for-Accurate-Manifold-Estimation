from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dryy.config import code_hash, load_config
from dryy.io import read_packet
from dryy.metrics import repeat_summary


def expected_scales(config: dict, experiment: str, method: str):
    if experiment == "e2":
        return list(config["n_values"])
    return list(config["dr"]["bandwidths_h2"] if method == "DR" else config["yy"]["central_kappas"])


def cell_scale(cell: dict, experiment: str, method: str):
    if experiment == "e2":
        return cell["n"]
    return cell["h2"] if method == "DR" else cell["central_kappa"]


def audit(experiment: str, result_root: Path) -> dict:
    config, config_hash = load_config(experiment)
    current_code_hash = code_hash()
    errors, packets = [], []
    for method in ("DR", "YY"):
        scales = expected_scales(config, experiment, method)
        for geometry in config["geometries"]:
            for repeat in range(config["repeats"]):
                path = result_root / "packets" / f"{experiment}_{method.lower()}_{geometry['key']}_r{repeat:02d}.json.gz"
                if not path.exists():
                    errors.append(f"missing:{path.name}")
                    continue
                try:
                    packet = read_packet(path)
                except Exception as error:
                    errors.append(f"unreadable:{path.name}:{error}")
                    continue
                packets.append(packet)
                if packet.get("config_hash") != config_hash:
                    errors.append(f"config_hash:{path.name}")
                if packet.get("code_hash") != current_code_hash:
                    errors.append(f"code_hash:{path.name}")
                if not packet.get("execution_complete"):
                    errors.append(f"truncated:{path.name}")
                found = [cell_scale(cell, experiment, method) for cell in packet["cells"]]
                if found != scales:
                    errors.append(f"scale_grid:{path.name}")
                for cell in packet["cells"]:
                    if cell["weights"]["max_abs_residual"] > config["validation"]["weight_residual_tolerance"]:
                        errors.append(f"weight_residual:{path.name}:{cell_scale(cell, experiment, method)}")
                    for name in ("Single", "Multi"):
                        metric = cell[name]
                        if metric["query_success_count"] + metric["query_failure_count"] != packet["query_count"]:
                            errors.append(f"query_accounting:{path.name}:{name}")
                if method == "YY" and experiment == "e3":
                    for query in range(packet["query_count"]):
                        digests = {
                            cell["common_start_digests"][query]
                            for cell in packet["cells"]
                            if cell["common_start_digests"][query] is not None
                        }
                        if len(digests) > 1:
                            errors.append(f"common_start:{path.name}:q{query}")

    summaries = []
    for method in ("DR", "YY"):
        for geometry in config["geometries"]:
            scales = expected_scales(config, experiment, method)
            selected = [p for p in packets if p["method"] == method and p["geometry"] == geometry["key"]]
            by_repeat = {p["repeat"]: p for p in selected}
            for scale_index, scale in enumerate(scales):
                for estimator in ("Single", "Multi"):
                    metrics = [by_repeat[r]["cells"][scale_index][estimator] for r in sorted(by_repeat)]
                    complete = sum(row["complete"] for row in metrics)
                    failures = sum(row["query_failure_count"] for row in metrics)
                    queries = sum(row["query_count"] for row in metrics)
                    failure_rate = failures / queries if queries else float("nan")
                    reliable = bool(
                        failure_rate <= config["validation"]["query_failure_rate_max"]
                        and complete >= config["validation"]["complete_repeats_min"]
                    )
                    taxonomy = Counter()
                    for row in metrics:
                        taxonomy.update(row["failure_taxonomy"])
                    summaries.append({
                        "method": method, "geometry": geometry["key"], "scale": scale,
                        "estimator": estimator, "complete_repeats": complete,
                        "query_failure_rate": failure_rate, "reliable": reliable,
                        "E_avg": repeat_summary([row["E_avg"] for row in metrics]),
                        "E_infty": repeat_summary([row["E_infty"] for row in metrics]),
                        "failure_taxonomy": dict(taxonomy),
                    })
    return {
        "schema_version": "DR_YY_FORMAL_AUDIT_V1", "experiment": experiment,
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "packet_count": len(packets), "expected_packet_count": 200,
        "config_hash": config_hash, "code_hash": current_code_hash, "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e2", "e3"), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.experiment, args.result_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True), encoding="utf-8")
    print(json.dumps({"status": report["status"], "packet_count": report["packet_count"], "errors": report["errors"][:20]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

