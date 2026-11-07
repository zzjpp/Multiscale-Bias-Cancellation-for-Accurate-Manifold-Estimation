from __future__ import annotations

import json
from pathlib import Path

from .experiment import config_hash, run_cell, source_hash, write_json
from .geometry import GEOMETRIES


def all_tasks(config: dict):
    return [
        (method, geometry, float(sigma), resolution)
        for method in config["methods"]
        for geometry in config["geometries"]
        for sigma in config["sigma_grid"]
        for resolution in config["resolutions"]
    ]


def task_cost(config: dict, task) -> int:
    method, geometry, _, resolution = task
    query_count = int(config["geometries"][geometry]["query_count"])
    settings = config["resolutions"][resolution]
    latent_resolution = int(config["geometries"][geometry]["integration_resolution"][resolution])
    node_count = GEOMETRIES[geometry].integration_node_count(latent_resolution)
    if method == "STSP":
        return query_count * node_count * int(settings["stsp_radial_order"])
    return query_count * node_count


def partition_tasks(config: dict, partition_count: int):
    bins = [[] for _ in range(int(partition_count))]
    loads = [0 for _ in bins]
    ranked = sorted(all_tasks(config), key=lambda task: (-task_cost(config, task), task))
    for task in ranked:
        target = min(range(len(bins)), key=lambda index: (loads[index], index))
        bins[target].append(task)
        loads[target] += task_cost(config, task)
    return [sorted(block) for block in bins], loads


def gpu_metadata(device: int):
    import cupy as cp

    cp.cuda.Device(int(device)).use()
    properties = cp.cuda.runtime.getDeviceProperties(int(device))
    raw_name = properties["name"]
    name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
    return {
        "device_index": int(device),
        "device_name": name,
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "float_precision": "float64",
    }


def run_gpu_partition(root: Path, device: int, partition: int, partition_count: int):
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    blocks, loads = partition_tasks(config, partition_count)
    tasks = blocks[int(partition)]
    metadata = gpu_metadata(device)
    frozen_config_hash = config_hash(config)
    frozen_source_hash = source_hash(root)
    packet_dir = root / "results" / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for task in tasks:
        method, geometry, sigma, resolution = task
        path = packet_dir / f"{method.lower()}_{geometry}_sigma_{sigma:.6f}_{resolution}.json"
        if path.exists():
            completed += 1
            continue
        packet = run_cell(
            config,
            method,
            geometry,
            sigma,
            resolution,
            backend="gpu_cupy",
            device=int(device),
        )
        packet["config_sha256"] = frozen_config_hash
        packet["source_sha256"] = frozen_source_hash
        packet["gpu"] = metadata
        packet["partition"] = {
            "index": int(partition),
            "count": int(partition_count),
            "declared_loads": loads,
        }
        write_json(path, packet)
        completed += 1
        print(f"[{completed}/{len(tasks)}] GPU{device} partition {partition}: {task}", flush=True)
    return tasks
