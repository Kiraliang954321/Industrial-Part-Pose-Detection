"""Assemble a QA-clean Project Dataset V1 from an explicit session split plan."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_conveyor.datasets.project_v1 import build_project_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True, action="append",
                        metavar="WORKSPACE", help="QA-clean annotation workspace (repeatable)")
    parser.add_argument("--split-plan", type=Path, required=True, metavar="PLAN",
                        help="explicit whole-session train/val/test split JSON")
    parser.add_argument("--output", type=Path, required=True, metavar="OUTPUT",
                        help="new, nonoverlapping output directory")
    args = parser.parse_args(argv)
    try:
        build_project_dataset(args.annotation, args.split_plan, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"dataset assembly failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
