"""Review accepted schema-v2 capture images with four-point OpenCV OBB annotation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_conveyor.annotation.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
