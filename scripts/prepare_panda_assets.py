"""Prepare pinned Panda assets for the explicitly scoped M4 simulation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_conveyor.simulation.robot_assets import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
