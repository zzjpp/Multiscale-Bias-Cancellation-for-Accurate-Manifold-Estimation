#!/usr/bin/env python3
import argparse
import json
import _bootstrap  # noqa: F401

from e3.config import CONFIG
from e3.runner import run_parent


def main():
    parser = argparse.ArgumentParser(description="Run one E3 geometry x repeat parent or gate subset")
    parser.add_argument("--geometry", required=True, choices=[row["key"] for row in CONFIG["geometries"]])
    parser.add_argument("--repeat", required=True, type=int)
    parser.add_argument("--mode", default="smoke")
    parser.add_argument("--bandwidths", nargs="*", type=float)
    parser.add_argument("--query-count", type=int)
    parser.add_argument("--truth-count", type=int)
    parser.add_argument("--result-root")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--production-confirmed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_parent(args.geometry, args.repeat, args.result_root, args.mode, args.bandwidths, args.query_count, args.resume, args.production_confirmed, args.truth_count), indent=2))


if __name__ == "__main__":
    main()
