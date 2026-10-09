from __future__ import annotations

import json
from pathlib import Path

from .experiment import config_hash, run_cell, source_hash, write_json
from .geometry import GEOMETRIES


def all_tasks(config):
    return [
        (geometry, float(sigma), resolution)
        for geometry in config["geometries"]
        for sigma in config["sigma_grid"]
        for resolution in config["resolutions"]
    ]


def task_cost(config, task):
    geometry, _, resolution = task
    design = config["geometries"][geometry]
    latent = int(design["integration_resolution"][resolution])
    latent_nodes = GEOMETRIES[geometry].integration_node_count(latent)
    queries = int(design["query_resolution"][resolution])
    return latent_nodes * queries


def assigned_tasks(config, partition, partitions):
    tasks = sorted(all_tasks(config), key=lambda task: (-task_cost(config, task), task))
    bins = [[] for _ in range(int(partitions))]
    loads = [0 for _ in bins]
    for task in tasks:
        index = min(range(len(bins)), key=lambda item: loads[item])
        bins[index].append(task)
        loads[index] += task_cost(config, task)
    return bins[int(partition)], loads


def run_partition(root: Path, device: int, partition: int, partitions: int):
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    frozen_config_hash = config_hash(config)
    frozen_source_hash = source_hash(root)
    tasks, loads = assigned_tasks(config, partition, partitions)
    packet_dir = root / "results" / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    print(f"partition={partition}/{partitions} device={device} tasks={len(tasks)} loads={loads}", flush=True)
    for index, (geometry, sigma, resolution) in enumerate(tasks, start=1):
        path = packet_dir / f"yy_{geometry}_sigma_{sigma:.6f}_{resolution}.json"
        if path.exists():
            print(f"[{index}/{len(tasks)}] exists {path.name}", flush=True)
            continue
        packet = run_cell(
            config,
            geometry,
            sigma,
            resolution,
            backend="gpu_cupy",
            device=int(device),
        )
        packet["config_sha256"] = frozen_config_hash
        packet["source_sha256"] = frozen_source_hash
        write_json(path, packet)
        print(
            f"[{index}/{len(tasks)}] {geometry} sigma={sigma:.6f} {resolution} "
            f"success={packet['success_count']}/{packet['query_count']}",
            flush=True,
        )
