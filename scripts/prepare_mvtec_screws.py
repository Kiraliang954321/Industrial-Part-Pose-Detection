"""Prepare MVTec Screws COCO-style JSON splits as Ultralytics OBB data.

Usage: python scripts/prepare_mvtec_screws.py SOURCE OUTPUT
"""

import argparse
from pathlib import Path

from yolo_conveyor.datasets import convert_mvtec_screws


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory containing split JSONs and source images")
    parser.add_argument("output", type=Path, help="new output directory")
    args = parser.parse_args(argv)
    try:
        manifest = convert_mvtec_screws(args.source, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for split, counts in manifest["splits"].items():
        print(f"{split}: {counts['images']} images, {counts['annotations']} annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
