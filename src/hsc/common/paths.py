"""Where the data and the results live. Both can be moved with environment variables.

HSC_DATA_DIR     folder that contains the TIME benchmark as TIME/<dataset>/<freq>/ (default: <repo>/data)
HSC_RESULTS_DIR  folder for every output (default: <repo>/results)
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(os.environ.get("HSC_RESULTS_DIR", ROOT / "results"))
TIME_DIR = Path(os.environ.get("HSC_DATA_DIR", ROOT / "data")) / "TIME"
