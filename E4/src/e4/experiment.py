from __future__ import annotations

import hashlib
import numpy as np

from .coefficients import two_scale_weights
from .config import CONFIG
from .geometry import sample_arclength
from .operator import WeightedSTSP
from .pipeline import run_main


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def seed_lineage(repeat: int) -> dict:
    if repeat not in CONFIG["heldout_repeats"]:
        raise ValueError("repeat is outside the frozen held-out set")
    common = [CONFIG["master_seed"], CONFIG["experiment_id"], int(repeat)]
    return {
        "algorithm": "PCG64DXSM",
        "experiment_id": CONFIG["experiment_id"],
        "repeat": int(repeat),
        "clean": common + [0],
        "noise": common + [1],
    }


def parent_sample(repeat: int, n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    lineage = seed_lineage(repeat)
    clean_rng = np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence(lineage["clean"]))
    )
    noise_rng = np.random.Generator(
        np.random.PCG64DXSM(np.random.SeedSequence(lineage["noise"]))
    )
    theta, clean = sample_arclength(clean_rng, n)
    standard_noise = noise_rng.standard_normal((n, 3))
    lineage.update(
        {
            "theta_sha256": _array_hash(theta),
            "clean_sha256": _array_hash(clean),
            "standard_noise_sha256": _array_hash(standard_noise),
        }
    )
    return clean, standard_noise, lineage


def run_repeat(repeat: int, *, smoke: bool = False) -> dict:
    n = 100000 if smoke else int(CONFIG["sample_size"])
    query_count = 4 if smoke else int(CONFIG["query_count"])
    candidate_count = 4 if smoke else int(CONFIG["nn_candidate_count"])
    clean, standard_noise, lineage = parent_sample(repeat, n)
    observations = clean + float(CONFIG["sigma"]) * standard_noise
    h1, h2, sigma = (float(CONFIG[key]) for key in ("h1", "h2", "sigma"))
    operator_h1 = WeightedSTSP(observations, h1, label="finite")
    operator_h2 = WeightedSTSP(observations, h2, label="finite")
    coefficient_record = two_scale_weights(h1, h2, sigma)
    result = run_main(
        operator_h1,
        operator_h2,
        h2,
        coefficient_record["weights"],
        N_q=query_count,
        M_match=candidate_count,
        phi=float(CONFIG["nn_phase"]),
    )
    return {
        "repeat": int(repeat),
        "seed_lineage": lineage,
        "coefficient_record": coefficient_record,
        "result": result,
        "smoke": bool(smoke),
    }
