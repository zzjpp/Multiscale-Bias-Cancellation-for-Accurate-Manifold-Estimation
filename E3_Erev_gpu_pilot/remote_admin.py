from __future__ import annotations

import argparse
import getpass
import gzip
import hashlib
import json
from pathlib import Path
import posixpath
import subprocess
import sys
import time

import paramiko


HOST = "10.16.130.20"
USER = "zhijun"


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        HOST,
        username=USER,
        password=getpass.getpass("Password: "),
        look_for_keys=False,
        allow_agent=False,
        timeout=15,
    )
    return client


def run(client: paramiko.SSHClient, command: str) -> None:
    _, stdout, stderr = client.exec_command(command)
    print(stdout.read().decode("utf-8", errors="replace"), end="")
    error = stderr.read().decode("utf-8", errors="replace")
    if error:
        print(error, end="")
    status = stdout.channel.recv_exit_status()
    if status:
        raise SystemExit(status)


def mkdirs(sftp: paramiko.SFTPClient, remote: str) -> None:
    current = ""
    for part in remote.strip("/").split("/"):
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_tree(client: paramiko.SSHClient, local: Path, remote: str) -> None:
    sftp = client.open_sftp()
    mkdirs(sftp, remote)
    for path in sorted(local.rglob("*")):
        if (not path.is_file() or "__pycache__" in path.parts or
                "downloaded_remote_results" in path.parts or path.suffix == ".pyc"):
            continue
        relative = path.relative_to(local).as_posix()
        destination = posixpath.join(remote, relative)
        mkdirs(sftp, posixpath.dirname(destination))
        sftp.put(str(path), destination)
    sftp.close()


def download_tree(client: paramiko.SSHClient, remote: str, local: Path) -> None:
    sftp = client.open_sftp()
    local.mkdir(parents=True, exist_ok=True)

    def visit(remote_dir: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(remote_dir):
            remote_path = posixpath.join(remote_dir, entry.filename)
            local_path = local_dir / entry.filename
            if entry.st_mode & 0o170000 == 0o040000:
                visit(remote_path, local_path)
            else:
                sftp.get(remote_path, str(local_path))

    try:
        visit(remote, local)
    finally:
        sftp.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inspect", "inspect-e3", "inspect-e3-tree", "inspect-runtime", "e3-stsp-search", "e7-status", "e7-runtime", "e7-upload-aggregator", "e7-finalize", "upload-high-density", "gate-high-density", "start-high-density", "status-high-density", "gate", "start", "start-tail", "status", "packet-meta", "download", "quarantine-relaunch", "upload"))
    parser.add_argument("--remote", default="/home/zhijun/e3_erev_gpu_pilot")
    args = parser.parse_args()
    client = connect()
    try:
        if args.action == "inspect":
            run(
                client,
                "hostname; "
                "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader; "
                "pgrep -af 'python|cuda' | head -n 40; "
                "find /home/zhijun -maxdepth 6 -type d -iname '*e3*' 2>/dev/null | head -n 60",
            )
        elif args.action == "inspect-e3":
            run(
                client,
                "echo RELEASES; "
                "find /home/zhijun /home/zhijun/shared_data/yao_group -maxdepth 7 "
                "-type f -path '*/methods/dr_yy/gpu_backend/run_parent_gpu.py' 2>/dev/null | head -n 30; "
                "echo PACKETS; "
                "find /home/zhijun/shared_data/yao_group -maxdepth 8 -type f "
                "-name 'e3_dr_circle_r2_r00.json.gz' 2>/dev/null | head -n 30; "
                "echo ENVS; "
                "find /home/zhijun/shared_data/yao_group/.venvs -maxdepth 3 -type f "
                "-path '*/bin/python' 2>/dev/null | head -n 30",
            )
        elif args.action == "inspect-e3-tree":
            run(
                client,
                "for d in "
                "/home/zhijun/shared_data/yao_group/e2_e3_dr_yy_reproduction "
                "/home/zhijun/shared_data/yao_group/e2_e3_dr_yy_gpu_20260923; do "
                "echo ROOT=$d; find $d -maxdepth 5 -type f 2>/dev/null | head -n 160; done; "
                "echo PYTHONS; find /home/zhijun/shared_data/yao_group/.venvs "
                "-maxdepth 4 -type f -name python 2>/dev/null | head -n 50",
            )
        elif args.action == "e7-status":
            root = args.remote.rstrip("/")
            run(
                client,
                f"echo GPU; "
                "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
                "--format=csv,noheader; "
                f"echo MASTER_PID; if kill -0 1687439 2>/dev/null; then echo RUNNING; else echo EXITED; fi; "
                "ps -fp 1687439 || true; readlink -f /proc/1687439/cwd 2>/dev/null || true; "
                f"echo PACKET_COUNTS; "
                f"printf 'DR '; find {root} -type f -name 'e7_dr_*.json.gz' 2>/dev/null | wc -l; "
                f"printf 'YY '; find {root} -type f -name 'e7_yy_*.json.gz' 2>/dev/null | wc -l; "
                f"echo TREE; find {root} -maxdepth 3 -type f 2>/dev/null | tail -n 50; "
                f"echo ACTIVE_PARENTS; pgrep -af 'run_parent_gpu.py.*e7_dr_yy_frozen' || true; "
                f"echo MASTER_LOG; tail -n 40 {root}/production_master.log 2>/dev/null || true; "
                f"echo ERROR_SCAN; grep -RHE 'Traceback|CUDA out of memory|FAILED|Error:' {root}/logs 2>/dev/null | tail -n 20 || true",
            )
        elif args.action == "e7-upload-aggregator":
            root = args.remote.rstrip("/")
            local = Path(__file__).resolve().parent.parent / "E7" / "methods" / "dr_yy"
            remote = f"{root}/project/simulation_experiments/E7/methods/dr_yy"
            sftp = client.open_sftp()
            try:
                sftp.put(str(local / "aggregate_and_plot.py"), posixpath.join(remote, "aggregate_and_plot.py"))
            finally:
                sftp.close()
            print(posixpath.join(remote, "aggregate_and_plot.py"))
        elif args.action == "e7-finalize":
            root = args.remote.rstrip("/")
            project = f"{root}/project/simulation_experiments/E7/methods/dr_yy"
            python = "/home/zhijun/shared_data/yao_group/.venvs/e3_yy_gpu_20260927/bin/python"
            run(
                client,
                f"set -e; cd {project}; "
                f"test $(find {root}/results/packets -maxdepth 1 -name 'e7_dr_*.json.gz' | wc -l) -eq 20; "
                f"test $(find {root}/results/packets -maxdepth 1 -name 'e7_yy_*.json.gz' | wc -l) -eq 20; "
                f"{python} aggregate_and_plot.py --output-root {root}/results --skip-plots; "
                f"grep -q '\"status\": \"PASS\"' {root}/results/e7_dr_yy_audit.json; "
                f"cd {root}/results; sha256sum e7_dr_aggregated.csv e7_yy_aggregated.csv "
                "e7_dr_yy_audit.json > REMOTE_AGGREGATE_SHA256.txt; "
                "cat e7_dr_yy_audit.json; cat REMOTE_AGGREGATE_SHA256.txt",
            )
            local = (
                Path(__file__).resolve().parent.parent / "E7" / "methods" /
                "dr_yy" / "formal_results"
            )
            local.mkdir(parents=True, exist_ok=True)
            names = [
                "e7_dr_aggregated.csv", "e7_yy_aggregated.csv", "e7_dr_yy_audit.json",
                "REMOTE_AGGREGATE_SHA256.txt",
            ]
            sftp = client.open_sftp()
            try:
                for name in names:
                    sftp.get(posixpath.join(root, "results", name), str(local / name))
                download_tree(client, posixpath.join(root, "results", "logs"), local / "logs")
            finally:
                sftp.close()
            plotter = Path(__file__).resolve().parent.parent / "E7" / "methods" / "dr_yy" / "plot_aggregated.py"
            subprocess.run(
                [sys.executable, str(plotter), "--output-root", str(local)],
                check=True,
            )
            output_names = [
                "e7_dr_aggregated.csv", "e7_yy_aggregated.csv", "e7_dr_yy_audit.json",
                "e7_dr_maximum_average_errors.pdf", "e7_dr_maximum_average_errors.png",
                "e7_yy_maximum_average_errors.pdf", "e7_yy_maximum_average_errors.png",
            ]
            manifest = []
            for name in output_names:
                digest = hashlib.sha256((local / name).read_bytes()).hexdigest()
                manifest.append(f"{digest}  {name}")
            (local / "OUTPUT_SHA256.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
            print(local)
        elif args.action == "e7-runtime":
            run(
                client,
                "for p in /usr/bin/python3 $(find /home/zhijun/shared_data/yao_group -maxdepth 6 "
                "\\( -type f -o -type l \\) -path '*/bin/python' 2>/dev/null); do "
                "printf '%s ' $p; $p -c 'import numpy,scipy,matplotlib,cupy; print(\"PASS\")' "
                "2>/dev/null || echo FAIL; done",
            )
        elif args.action == "e3-stsp-search":
            run(
                client,
                "find /home/zhijun /home/zhijun/shared_data/yao_group "
                "-maxdepth 10 -type f "
                "\\( -iname 'e3*stsp*.json*' -o -iname 'e3_stsp_*.json.gz' "
                "-o -path '*/e3*/packets/*stsp*' \\) 2>/dev/null | head -n 200",
            )
        elif args.action == "upload-high-density":
            local_pilot = Path(__file__).resolve().parent
            local_hd = local_pilot.parent / "E3_Erev_high_density"
            local_e3 = local_pilot.parent / "E3"
            project = posixpath.join(args.remote, "project", "simulation_experiments")
            upload_tree(client, local_hd, posixpath.join(project, "E3_Erev_high_density"))
            upload_tree(client, local_e3, posixpath.join(project, "E3"))
            upload_tree(
                client,
                local_pilot / "observations_numpy244",
                posixpath.join(project, "E3_Erev_gpu_pilot", "observations_numpy244"),
            )
            upload_tree(
                client,
                local_pilot.parent.parent / "e2_e3_dr_yy_reproduction" / "e3_final_merge_20260924" / "packets",
                posixpath.join(args.remote, "base_packets", "dr"),
            )
            upload_tree(
                client,
                local_pilot / "base_current_yy",
                posixpath.join(args.remote, "base_packets", "yy"),
            )
            print(args.remote)
        elif args.action == "gate-high-density":
            root = args.remote.rstrip("/")
            project = f"{root}/project/simulation_experiments/E3_Erev_high_density"
            python = "/home/zhijun/shared_data/yao_group/.venvs/e3_yy_gpu_20260927/bin/python"
            run(
                client,
                f"set -e; cd {project}; "
                f"{python} gate_query_grids.py > QUERY_GRID_GATE.server.log; "
                f"CUDA_VISIBLE_DEVICES=0 {python} equivalence_gate_dryy_shards_gpu.py "
                f"> DRYY_GPU_SHARD_EQUIVALENCE.server.log; "
                "grep -H '\"status\": \"PASS\"' QUERY_GRID_GATE.json DRYY_GPU_SHARD_EQUIVALENCE.json; "
                "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader",
            )
        elif args.action == "start-high-density":
            root = args.remote.rstrip("/")
            project = f"{root}/project/simulation_experiments/E3_Erev_high_density"
            python = "/home/zhijun/shared_data/yao_group/.venvs/e3_yy_gpu_20260927/bin/python"
            run(
                client,
                f"set -e; cd {project}; "
                "grep -q '\"status\": \"PASS\"' QUERY_GRID_GATE.json; "
                "grep -q '\"status\": \"PASS\"' DRYY_GPU_SHARD_EQUIVALENCE.json; "
                f"test $(find {root}/base_packets/dr -maxdepth 1 -name 'e3_dr_*_r00.json.gz' | wc -l) -eq 5; "
                f"test $(find {root}/base_packets/yy -maxdepth 1 -name 'e3_yy_*_r00.json.gz' | wc -l) -eq 5; "
                f"mkdir -p {root}/results; "
                f"test ! -e {root}/dryy.pid; test ! -e {root}/stsp.pid; "
                f"nohup {python} launch_dryy_stage1.py --result-root {root}/results "
                f"--dr-base {root}/base_packets/dr --yy-base {root}/base_packets/yy "
                f"--shard-count 8 --workers-per-gpu 4 > {root}/dryy_master.log 2>&1 & echo $! > {root}/dryy.pid; "
                f"nohup {python} launch_stsp_stage1.py --result-root {root}/results "
                f"--shard-count 8 --workers 8 > {root}/stsp_master.log 2>&1 & echo $! > {root}/stsp.pid; "
                f"cat {root}/dryy.pid {root}/stsp.pid",
            )
        elif args.action == "status-high-density":
            root = args.remote.rstrip("/")
            run(
                client,
                "echo GPU; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; "
                f"echo PIDS; for n in dryy stsp; do p=$(cat {root}/$n.pid 2>/dev/null || true); "
                "if test -n \"$p\" && kill -0 $p 2>/dev/null; then echo $n:$p:RUNNING; else echo $n:$p:EXITED; fi; done; "
                f"echo COUNTS; printf 'DRYY '; find {root}/results/stage1/shards -type f -name '*.json.gz' 2>/dev/null | wc -l; "
                f"printf 'STSP '; find {root}/results/stage1/stsp_shards -type f -name '*.json.gz' 2>/dev/null | wc -l; "
                f"echo DRYY_LOG; tail -n 12 {root}/dryy_master.log 2>/dev/null || true; "
                f"echo STSP_LOG; tail -n 12 {root}/stsp_master.log 2>/dev/null || true; "
                f"echo ERRORS; grep -RHE 'Traceback|failed tasks|CUDA out of memory' {root}/*_master.log {root}/results/stage1/*_launcher_logs 2>/dev/null | tail -n 20 || true",
            )
        elif args.action == "inspect-runtime":
            run(
                client,
                "echo DR_R00; find /home/zhijun/shared_data/yao_group "
                "-type f -name 'e3_dr_*_r00.json.gz' 2>/dev/null | sort; "
                "echo YY_R00; find /home/zhijun/shared_data/yao_group "
                "-type f -name 'e3_yy_*_r00.json.gz' 2>/dev/null | sort; "
                "echo PYTHONS; find /home/zhijun/shared_data/yao_group -maxdepth 5 "
                "\\( -type f -o -type l \\) -path '*/bin/python*' 2>/dev/null | sort | head -n 80; "
                "echo CUPY; for p in $(find /home/zhijun/shared_data/yao_group -maxdepth 5 "
                "\\( -type f -o -type l \\) -path '*/bin/python' 2>/dev/null); do "
                "$p -c 'import cupy; print(cupy.__version__)' >/dev/null 2>&1 && echo $p; done",
            )
        elif args.action == "gate":
            run(
                client,
                "set -e; ROOT=/home/zhijun/e3_erev_incremental_20260928; "
                "PY=/home/zhijun/shared_data/yao_group/.venvs/e3_yy_gpu_20260927/bin/python; "
                "mkdir -p $ROOT/pilot/results; "
                "$PY -m py_compile $ROOT/pilot/run_gpu_pilot.py $ROOT/pilot/summarize_erev.py; "
                "cd $ROOT/E3/methods/dr_yy; "
                "CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "$PY gpu_backend/equivalence_gate.py --output $ROOT/pilot/results/equivalence_gate.json",
            )
        elif args.action == "start":
            run(
                client,
                "set -e; ROOT=/home/zhijun/e3_erev_incremental_20260928/pilot; "
                "test \"$(/home/zhijun/shared_data/yao_group/.venvs/e3_yy_gpu_20260927/bin/python -c "
                "\"import json; print(json.load(open('$ROOT/results/equivalence_gate.json'))['status'])\")\" = PASS; "
                "bash $ROOT/launch_remote.sh",
            )
        elif args.action == "start-tail":
            run(
                client,
                "set -e; ROOT=/home/zhijun/e3_erev_incremental_20260928/pilot; "
                "mkdir -p $ROOT/results/logs; "
                "nohup bash $ROOT/worker_gpu1_tail.sh >$ROOT/results/logs/gpu1_tail.log 2>&1 & "
                "echo $! >$ROOT/results/logs/gpu1_tail.pid; "
                "echo gpu1_tail_pid=$(cat $ROOT/results/logs/gpu1_tail.pid)",
            )
        elif args.action == "status":
            run(
                client,
                "ROOT=/home/zhijun/e3_erev_incremental_20260928/pilot; "
                "echo GPU; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; "
                "echo PACKETS; find $ROOT/results/gpu_packets/packets -type f -name '*.json.gz' 2>/dev/null | sort; "
                "echo WORKERS; for f in $ROOT/results/logs/*.pid; do [ -f \"$f\" ] && "
                "echo $(basename $f)=$(cat $f):$(kill -0 $(cat $f) 2>/dev/null && echo RUNNING || echo EXITED); done; "
                "echo GPU0_LOG; tail -n 20 $ROOT/results/logs/gpu0.log 2>/dev/null; "
                "echo GPU1_LOG; tail -n 20 $ROOT/results/logs/gpu1.log 2>/dev/null; "
                "echo PROCESS_DETAIL; ps -eo pid,ppid,etime,pcpu,pmem,stat,args | "
                "grep -E 'e3_erev_incremental|run_gpu_pilot' | grep -v grep",
            )
        elif args.action == "packet-meta":
            packet_root = "/home/zhijun/e3_erev_incremental_20260928/pilot/results/gpu_packets/packets"
            sftp = client.open_sftp()
            try:
                for entry in sorted(sftp.listdir_attr(packet_root), key=lambda item: item.filename):
                    if not entry.filename.endswith(".json.gz"):
                        continue
                    with sftp.open(posixpath.join(packet_root, entry.filename), "rb") as handle:
                        packet = json.loads(gzip.decompress(handle.read()))
                    print(json.dumps({
                        "file": entry.filename,
                        "runtime_seconds": packet.get("runtime_seconds"),
                        "method": packet.get("method"),
                        "geometry": packet.get("geometry"),
                        "cells": len(packet.get("cells", [])),
                        "query_count": packet.get("query_count"),
                        "finished_utc": packet.get("finished_utc"),
                        "remote_mtime_epoch": entry.st_mtime,
                    }, sort_keys=True))
            finally:
                sftp.close()
        elif args.action == "download":
            remote_results = "/home/zhijun/e3_erev_incremental_20260928/pilot/results"
            local_results = Path(__file__).resolve().parent / "downloaded_remote_results"
            download_tree(client, remote_results, local_results)
            print(local_results)
        elif args.action == "quarantine-relaunch":
            run(
                client,
                "set -e; ROOT=/home/zhijun/e3_erev_incremental_20260928/pilot; "
                "PY=/home/zhijun/shared_data/yao_group/.venvs/e3_yy_gpu_20260927/bin/python; "
                "test -d $ROOT/results; test ! -e $ROOT/results_mismatched_numpy243; "
                "mv $ROOT/results $ROOT/results_mismatched_numpy243; "
                "mkdir -p $ROOT/results; "
                "cp $ROOT/results_mismatched_numpy243/equivalence_gate.json $ROOT/results/equivalence_gate.json; "
                "cd $ROOT; $PY -c \"import json,numpy as np,sys; "
                "sys.path.insert(0,'../E3/methods/dr_yy/src'); "
                "from dryy.geometry import array_hash; "
                "m=json.load(open('observations_numpy244/manifest.json')); "
                "assert all(array_hash(np.load('observations_numpy244/'+r['file']))==r['observed_sha256'] for r in m['records']); "
                "print('MATCHED_OBSERVATIONS_PASS=5')\"; "
                "bash $ROOT/launch_remote.sh",
            )
        else:
            local_pilot = Path(__file__).resolve().parent
            local_e3 = local_pilot.parent / "E3"
            upload_tree(client, local_pilot, posixpath.join(args.remote, "pilot"))
            upload_tree(client, local_e3, posixpath.join(args.remote, "E3"))
            print(args.remote)
    finally:
        client.close()


if __name__ == "__main__":
    main()
