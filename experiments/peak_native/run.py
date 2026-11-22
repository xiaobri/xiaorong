"""Run the peak_native ablation with its published configuration."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from launch_experiment import main

if __name__ == "__main__":
    main("peak_native")
