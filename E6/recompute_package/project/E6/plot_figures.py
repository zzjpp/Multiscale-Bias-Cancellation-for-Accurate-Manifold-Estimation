from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.insert(0, str(ROOT / "src"))

from e6_misspec.plotting import build

if __name__ == "__main__":
    build(ROOT)
    print("Generated exactly three E6 figures in figures/.")
