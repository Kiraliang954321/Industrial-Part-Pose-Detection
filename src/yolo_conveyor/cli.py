"""Single-image experimental inference; no face accuracy or control claims."""

import argparse
import json
from pathlib import Path

from .config import DEFAULT_MODEL
from .geometry import long_axis_angle, point_in_polygon
from .types import Point
from .ultralytics_adapter import UltralyticsAdapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single-frame OBB inference (pretrained classes are not front/back).")
    parser.add_argument("image", type=Path, help="single image in the same pixel coordinates as the ROI")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--roi", type=Path, help="JSON file containing polygon [[x,y], ...] in image pixels")
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error("image does not exist")
    if not 0 <= args.confidence <= 1:
        parser.error("confidence must be in [0, 1]")
    roi: list[Point] | None = None
    if args.roi is not None:
        try:
            data = json.loads(args.roi.read_text(encoding="utf-8"))
            if not isinstance(data, list) or len(data) < 3 or any(len(p) != 2 for p in data):
                raise ValueError("expected at least three [x,y] vertices")
            roi = [(float(p[0]), float(p[1])) for p in data]
            point_in_polygon((0, 0), roi)  # validate finite coordinates
        except (OSError, ValueError, TypeError) as exc:
            parser.error(f"invalid ROI: {exc}")
    detections = UltralyticsAdapter(args.model).predict(args.image, args.confidence)
    # Keep unfiltered predictions to inspect boundary errors; in_roi only annotates them.
    print(json.dumps({"image": str(args.image), "model": args.model,
                      "detections": [{"class_index": d.class_index, "confidence": d.confidence,
                                      "corners": d.corners, "angle_deg": long_axis_angle(d.corners),
                                      "in_roi": point_in_polygon(d.center, roi) if roi else None}
                                     for d in detections]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
