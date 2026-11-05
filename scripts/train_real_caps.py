"""CLI for development-only Part A photo fine-tuning with the formal gate CLOSED."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_conveyor.real_photo_training import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
