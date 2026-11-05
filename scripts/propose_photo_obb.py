"""Generate ROI-assisted OBB candidates for human review; never training labels."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_conveyor.photo_obb_proposals import propose_capture  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="immutable schema-v2 capture session")
    parser.add_argument("--rois", type=Path, required=True, help="JSON saved_path -> normalized [x0,y0,x1,y1]")
    parser.add_argument("--output", type=Path, required=True, help="entirely new, separate proposal directory")
    parser.add_argument("--foreground-color", choices=("none", "yellow"), default="none",
                        help="optional HSV color mask union with gray Otsu; candidates still require human review")
    args = parser.parse_args(argv)
    try:
        report = propose_capture(args.capture, args.rois, args.output, foreground_color=args.foreground_color)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"proposal generation failed: {exc}\n")
    print(f"Human-review candidates only: {args.output.resolve()}")
    print(f"Counts: {report['summary']}; formal training gate remains CLOSED")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
