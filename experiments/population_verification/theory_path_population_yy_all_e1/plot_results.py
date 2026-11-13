from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from e1_yy_path.plotting import render


if __name__ == "__main__":
    print(json.dumps(render(ROOT / "results"), indent=2))
