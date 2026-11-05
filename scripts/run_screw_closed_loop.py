"""Execute explicitly paired Panda screw-grasp development episodes."""
from pathlib import Path
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_conveyor.simulation.closed_loop import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
