"""Read-only Ultralytics OBB dataset structure/label syntax audit.

Usage: python scripts/audit_dataset.py PATH [--classes N]
PATH contains images/ and labels/ with matching relative stems. Empty label files
are allowed for negative images; this is not a semantic/quality audit.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


@dataclass(frozen=True)
class AuditResult:
    image_count: int
    label_count: int
    errors: tuple[str, ...]


def validate_row(row: str, class_count: int | None = None) -> str | None:
    """Return reason for an invalid class_index + eight normalized corners row."""
    fields = row.split()
    if len(fields) != 9:
        return "expected class_index followed by 8 corner coordinates"
    try:
        class_index = int(fields[0])
    except ValueError:
        return "class_index must be a non-negative integer"
    if class_index < 0 or (class_count is not None and class_index >= class_count):
        return "class_index out of range"
    try:
        coordinates = [float(field) for field in fields[1:]]
    except ValueError:
        return "corner coordinates must be numeric"
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in coordinates):
        return "corner coordinates must be finite and in [0, 1]"
    return None


def audit_dataset(root: Path, class_count: int | None = None) -> AuditResult:
    if class_count is not None and class_count <= 0:
        raise ValueError("class_count must be positive")
    root = Path(root)
    images_dir, labels_dir = root / "images", root / "labels"
    errors: list[str] = []
    if not images_dir.is_dir():
        errors.append("missing images/ directory")
    if not labels_dir.is_dir():
        errors.append("missing labels/ directory")
    if errors:
        return AuditResult(0, 0, tuple(errors))
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for directory, suffixes, target in ((images_dir, IMAGE_SUFFIXES, images),
                                         (labels_dir, frozenset({".txt"}), labels)):
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            key = path.relative_to(directory).with_suffix("").as_posix()
            if key in target:
                errors.append(f"duplicate stem: {path.relative_to(root).as_posix()}")
            else:
                target[key] = path
    if not images:
        errors.append("no images found")
    for key in sorted(images.keys() - labels.keys()):
        errors.append(f"missing label: {key}.txt")
    for key in sorted(labels.keys() - images.keys()):
        errors.append(f"missing image: {key}")
    for key in sorted(labels):
        path = labels[key]
        try:
            with path.open("r", encoding="utf-8") as stream:
                for number, row in enumerate(stream, start=1):
                    if not row.strip():
                        errors.append(f"{path.relative_to(root).as_posix()}:{number}: blank label row")
                        continue
                    reason = validate_row(row, class_count)
                    if reason:
                        errors.append(f"{path.relative_to(root).as_posix()}:{number}: {reason}")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(root).as_posix()}: unable to read label: {exc}")
    return AuditResult(len(images), len(labels), tuple(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--classes", type=int, help="optional number of class IDs (0 through N-1)")
    args = parser.parse_args(argv)
    try:
        result = audit_dataset(args.dataset, args.classes)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"image_count": result.image_count, "label_count": result.label_count,
                      "errors": result.errors}, indent=2))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
