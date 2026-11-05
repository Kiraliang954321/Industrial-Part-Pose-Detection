"""Import real-photo intake as separate Part A development and Part B reference sessions."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yolo_conveyor.photo_intake import import_photo_intake  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True, help="verified real-photo intake manifest")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; never overwritten")
    args = parser.parse_args(argv)
    try:
        report = import_photo_intake(args.intake, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"photo import failed: {exc}\n")
    print(f"Real-photo development intake: {args.output.resolve()}")
    for session in report["sessions"]:
        print(f"{session['part_id']}: {session['counts']}; role={session['role']}")
    print("One physical capture group; no formal split, training or gate opening.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
