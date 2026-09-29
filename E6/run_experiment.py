from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from e6_misspec.pipeline import audit, freeze_hashes, run
from e6_misspec.plotting import build


if __name__ == "__main__":
    freeze_hashes(ROOT)
    payload = run(ROOT)
    report = audit(ROOT, payload)
    build(ROOT)
    print(report["status"])
    if report["status"] != "PASS":
        raise SystemExit(1)
