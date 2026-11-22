"""Machine-specific roots used by the released research scripts."""
import os
import sys
import tempfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parent
BASELINE = SOURCE / "I2MOE6_F1"
STORAGE = Path(os.environ.get("CZB_STORAGE", SOURCE.parent / "private-data")).resolve()
RUNS = Path(os.environ.get("CZB_RUNS", SOURCE.parent / "private-runs")).resolve()
STORED_PREFIX = os.environ.get("CZB_STORED_PREFIX", "/eeg")
PYTHON = sys.executable
LOCK_PREFIX = str(Path(tempfile.gettempdir()) / "czb_moe_baseline_gpu")
