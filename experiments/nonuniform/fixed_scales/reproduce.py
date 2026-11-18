"""Fresh nonuniform_E2 experiment; no historical fitted outputs are required.

Run with --help for options or --output NEW to compute the paper profile.
Use --profile smoke for a reduced actual fitting check, not paper evidence.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from manifold_experiments.entrypoints import reproduce_experiment

if __name__ == "__main__":
    reproduce_experiment("nonuniform_E2")
