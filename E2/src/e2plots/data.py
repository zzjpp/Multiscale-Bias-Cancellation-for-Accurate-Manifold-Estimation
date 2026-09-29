from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_style(root: Path) -> dict[str, Any]:
    path = root / "config" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _check_input_hashes(root: Path, style: dict[str, Any]) -> None:
    for name, expected in style["input_sha256"].items():
        path = root / "data" / name
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Input checksum mismatch for {name}: expected {expected}, got {actual}"
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _load_stsp(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _read_csv(root / "data" / "stsp_formal_summary.csv"):
        if item["metric"] not in {"E_avg", "E_infty"}:
            continue
        estimator = "Multi" if item["method"] == "ENF-Multi" else "Single"
        rows.append(
            {
                "geometry": item["geometry"],
                "scale": int(item["n_sample"]),
                "method": "STSP",
                "estimator": estimator,
                "metric": item["metric"],
                "median": float(item["median"]),
                "q1": float(item["q1"]),
                "q3": float(item["q3"]),
                "reliable": item["reliable"].lower() == "true",
            }
        )
    return rows


def _load_dr_yy(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _read_csv(root / "data" / "dr_yy_formal_summary.csv"):
        rows.append(
            {
                "geometry": item["geometry"],
                "scale": int(item["scale"]),
                "method": item["method"],
                "estimator": item["estimator"],
                "metric": item["metric"],
                "median": float(item["median"]),
                "q1": float(item["q1"]),
                "q3": float(item["q3"]),
                "reliable": item["reliable"].lower() == "true",
            }
        )
    return rows


def load_release_rows(root: Path, style: dict[str, Any]) -> list[dict[str, Any]]:
    _check_input_hashes(root, style)
    rows = _load_stsp(root) + _load_dr_yy(root)
    validate_release_rows(rows, style)
    return rows


def validate_release_rows(rows: list[dict[str, Any]], style: dict[str, Any]) -> None:
    geometries = style["main_geometries"] + style["appendix_geometries"]
    expected_scales = style["sample_sizes"]
    for geometry in geometries:
        for method in style["methods"]:
            for estimator in style["estimators"]:
                for metric in style["metrics"]:
                    current = sorted(
                        row["scale"]
                        for row in rows
                        if row["geometry"] == geometry
                        and row["method"] == method
                        and row["estimator"] == estimator
                        and row["metric"] == metric
                    )
                    if current != expected_scales:
                        raise RuntimeError(
                            "Incomplete or duplicated release grid: "
                            f"{geometry}/{method}/{estimator}/{metric}: {current}"
                        )
    expected_rows = (
        len(geometries)
        * len(style["methods"])
        * len(style["estimators"])
        * len(style["metrics"])
        * len(expected_scales)
    )
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(rows)}")
