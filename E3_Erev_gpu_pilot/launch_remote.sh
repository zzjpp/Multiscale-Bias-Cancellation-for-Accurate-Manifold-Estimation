#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhijun/e3_erev_incremental_20260928/pilot
mkdir -p "$ROOT/results/logs"
nohup bash "$ROOT/worker_gpu0.sh" >"$ROOT/results/logs/gpu0.log" 2>&1 &
echo $! >"$ROOT/results/logs/gpu0.pid"
nohup bash "$ROOT/worker_gpu1.sh" >"$ROOT/results/logs/gpu1.log" 2>&1 &
echo $! >"$ROOT/results/logs/gpu1.pid"
printf 'gpu0_pid=%s\ngpu1_pid=%s\n' "$(cat "$ROOT/results/logs/gpu0.pid")" "$(cat "$ROOT/results/logs/gpu1.pid")"
