"""Command-line interface for operator-controlled raw phone/video capture."""

import argparse
import math
from pathlib import Path
import re
import sys

from .collector import CaptureCollector, run_collector
from .duplicates import DuplicateGuard
from .quality import QualityPolicy
from .session import CaptureSession
from yolo_conveyor.label_contract import validate_part_id


def parse_source(text: str) -> int | str:
    """Interpret only nonnegative decimal integers as camera indices."""
    return int(text) if re.fullmatch(r"[0-9]+", text) else text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect unlabelled raw front/back JPEGs from a phone camera or video")
    parser.add_argument("--part-id", required=True, type=validate_part_id,
                        help="safe single-token part ID for this capture session")
    parser.add_argument("--source", default="0", help="nonnegative camera index or video path/URL (default: 0)")
    parser.add_argument("--output-root", type=Path, default=Path("data/captures"))
    parser.add_argument("--backend", choices=("auto", "dshow"), default="auto")
    parser.add_argument("--width", type=float)
    parser.add_argument("--height", type=float)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--auto-interval", type=float, default=0.0, help="seconds between captures when auto is toggled on")
    parser.add_argument("--min-blur-score", type=float, default=0.0)
    parser.add_argument("--min-brightness", type=float, default=0.0)
    parser.add_argument("--max-brightness", type=float, default=255.0)
    parser.add_argument("--quality-policy", choices=("warn", "reject"), default="warn")
    parser.add_argument("--duplicate-distance", type=int, default=5)
    parser.add_argument("--duplicate-history", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.auto_interval) or args.auto_interval < 0:
        parser.error("--auto-interval must be finite and nonnegative")
    try:
        quality = QualityPolicy(args.min_blur_score, args.min_brightness,
                                args.max_brightness, args.quality_policy == "reject")
        duplicates = DuplicateGuard(max_distance=args.duplicate_distance,
                                    history_size=args.duplicate_history)
        source = parse_source(args.source)
        session = CaptureSession.create(
            args.output_root, part_id=args.part_id, source=args.source, backend=args.backend,
            config={"width": args.width, "height": args.height, "fps": args.fps,
                    "auto_interval": args.auto_interval, "min_blur_score": args.min_blur_score,
                    "min_brightness": args.min_brightness, "max_brightness": args.max_brightness,
                    "quality_policy": args.quality_policy,
                    "duplicate_distance": args.duplicate_distance,
                    "duplicate_history": args.duplicate_history},
        )
        print(f"Capture session: {session.path}")
        run_collector(CaptureCollector(session, quality=quality, duplicates=duplicates), source,
                      backend=args.backend, width=args.width, height=args.height,
                      fps=args.fps, auto_interval=args.auto_interval)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Capture error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
