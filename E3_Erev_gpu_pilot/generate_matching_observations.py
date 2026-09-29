from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
E3 = ROOT.parent / "E3"
DRYY = E3 / "methods" / "dr_yy"
sys.path.insert(0, str(DRYY / "src"))

from dryy.config import load_config  # noqa: E402
from dryy.geometry import geometries_from_config, noisy_sample  # noqa: E402


def read_packet(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def base_packet(geometry: str) -> Path:
    staging = WORKSPACE / "e2_e3_dr_yy_reproduction" / "e3_merge_staging_20260924" / "packets"
    path = staging / f"e3_dr_{geometry}_r00.json.gz"
    if path.exists():
        return path
    downloaded = WORKSPACE / "e2_e3_dr_yy_reproduction" / "server_download_20260924" / "e3_dr_remaining" / "packets"
    path = downloaded / f"e3_dr_{geometry}_r00.json.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    config, config_hash = load_config("e3")
    output = ROOT / "observations_numpy244"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for geometry in geometries_from_config(config):
        observed, lineage = noisy_sample(config, geometry, 0, int(config["n"]))
        packet = read_packet(base_packet(geometry.key))
        expected = packet["seed"]["observed_sha256"]
        if lineage["observed_sha256"] != expected:
            raise RuntimeError(
                f"{geometry.key}: regenerated {lineage['observed_sha256']} != formal {expected}"
            )
        path = output / f"{geometry.key}_r00_observed.npy"
        np.save(path, np.asarray(observed, dtype=np.float64), allow_pickle=False)
        records.append({
            "geometry": geometry.key,
            "repeat": 0,
            "shape": list(observed.shape),
            "observed_sha256": expected,
            "clean_seed": lineage["clean_seed"],
            "noise_seed": lineage["noise_seed"],
            "base_packet": str(base_packet(geometry.key).relative_to(WORKSPACE)),
            "file": path.name,
        })
    manifest = {
        "schema_version": "E3_MATCHED_OBSERVATIONS_V1",
        "numpy": np.__version__,
        "source_config_hash": config_hash,
        "n": int(config["n"]),
        "sigma": float(config["sigma"]),
        "records": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output / "manifest.json")


if __name__ == "__main__":
    main()
