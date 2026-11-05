"""Freeze the M4 development evaluation; never opens a real training/acceptance gate."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_conveyor.simulation.m4_protocol import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
