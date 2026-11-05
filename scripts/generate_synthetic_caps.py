"""Create independent synthetic-only cap scenes, OBB dataset and passage videos."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_conveyor.synthetic import generate_bundle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new bundle directory; never overwritten")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--images-per-session", type=int, default=120, help="even count; half each face")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--video-seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        result = generate_bundle(args.output, seed=args.seed, images_per_session=args.images_per_session,
                                 width=args.width, height=args.height,
                                 video_seconds=args.video_seconds, fps=args.fps,
                                 progress=lambda message: print(message, flush=True))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"synthetic generation failed: {exc}\n")
    print(f"Synthetic-only bundle ready: {args.output.resolve()}")
    print(f"Images: {sum(value['images'] for value in result['counts'].values())}; formal gate remains CLOSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
