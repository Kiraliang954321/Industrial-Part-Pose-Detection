"""Phase 1D sidecar contract: no numeric dataset class identifiers."""

import hashlib
import math
from pathlib import Path, PureWindowsPath
import re

from yolo_conveyor.geometry import long_axis_angle
from yolo_conveyor.label_contract import FACE_LABELS, canonical_class_name, validate_part_id

SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG frame dimensions without decoding pixels or adding dependencies."""
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError("source is not a JPEG")
        while True:
            marker = stream.read(1)
            if marker != b"\xff":
                raise ValueError("invalid JPEG marker")
            while marker == b"\xff":
                marker = stream.read(1)
            if marker in (b"", b"\xd9", b"\xda"):
                raise ValueError("JPEG has no frame dimensions")
            if marker == b"\x01" or 0xd0 <= marker[0] <= 0xd7:
                continue
            size_bytes = stream.read(2)
            if len(size_bytes) != 2:
                raise ValueError("truncated JPEG segment")
            size = int.from_bytes(size_bytes, "big")
            if size < 2:
                raise ValueError("invalid JPEG segment")
            if marker[0] in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
                             0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
                header = stream.read(5)
                if len(header) != 5 or size < 7:
                    raise ValueError("truncated JPEG frame")
                height, width = int.from_bytes(header[1:3], "big"), int.from_bytes(header[3:5], "big")
                if not width or not height:
                    raise ValueError("invalid JPEG dimensions")
                return width, height
            if len(stream.read(size - 2)) != size - 2:
                raise ValueError("truncated JPEG segment")


def safe_source_path(raw: str, face: str | None = None) -> Path:
    """Reject cross-platform absolute/traversal paths and non-image targets."""
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw or ":" in raw:
        raise ValueError("unsafe source path")
    path = Path(raw)
    parts = raw.split("/")
    if (path.is_absolute() or PureWindowsPath(raw).drive or
            len(parts) < 2 or parts[0] not in FACE_LABELS or
            any(part in ("", ".", "..") for part in parts) or
            path.suffix.lower() not in (".jpg", ".jpeg")):
        raise ValueError("unsafe source path")
    if face is not None and parts[0] != face:
        raise ValueError("source path does not match face")
    return path


def record_path(raw: str) -> Path:
    return Path("records") / safe_source_path(raw).with_suffix(".json")


def safe_file(root: Path, relative: Path) -> Path:
    """Resolve symlinks before accessing either source or record paths."""
    base = root.resolve()
    candidate = root / relative
    if not candidate.resolve().is_relative_to(base):
        raise ValueError(f"unsafe path: {relative}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"unsafe symlink: {relative}")
    return candidate


def validate_corners(corners: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(corners, (list, tuple)) or len(corners) != 4:
        raise ValueError("exactly four corners required")
    points = []
    for point in corners:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each corner must be an x,y pair")
        x, y = point
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or
               not math.isfinite(v) or not 0 <= v <= 1 for v in (x, y)):
            raise ValueError("corners must be finite and normalized to [0,1]")
        points.append((float(x), float(y)))
    if len(set(points)) != 4:
        raise ValueError("corners must be unique")
    # Strict convexity of successive perimeter edges also rejects crossed and
    # collinear/zero-area quadrilaterals. Either winding direction is valid.
    crosses = []
    for i in range(4):
        a, b, c = (points[(i + j) % 4] for j in range(3))
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) -
                       (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(v > 0 for v in crosses) or all(v < 0 for v in crosses)):
        raise ValueError("corners must describe a nonzero-area, convex perimeter")
    return tuple(points)


def angle_degrees(corners: object, width: int, height: int) -> float | None:
    points = validate_corners(corners)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise ValueError("image dimensions must be positive integers")
    return long_axis_angle([(x * width, y * height) for x, y in points])


def validate_record(record: object) -> dict:
    if not isinstance(record, dict):
        raise ValueError("review record must be an object")
    status = record.get("status")
    required = {"schema_version", "status", "part_id", "face", "class_name",
                "source_path", "source_image_sha256", "image_width", "image_height"}
    if status == "annotated":
        required |= {"corners"}
    elif status == "excluded":
        required |= {"reason"}
    else:
        raise ValueError("status must be annotated or excluded")
    if set(record) != required:
        raise ValueError(f"record fields must be exactly {sorted(required)}")
    if type(record["schema_version"]) is not int or record["schema_version"] != 1:
        raise ValueError("unsupported record schema_version")
    validate_part_id(record["part_id"])
    face = record["face"]
    if face not in FACE_LABELS or record["class_name"] != canonical_class_name(record["part_id"], face):
        raise ValueError("face/class name mismatch")
    safe_source_path(record["source_path"], face)
    if not isinstance(record["source_image_sha256"], str) or not SHA256.fullmatch(record["source_image_sha256"]):
        raise ValueError("invalid source image SHA-256")
    width, height = record["image_width"], record["image_height"]
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("image dimensions must be positive integers")
    if status == "annotated":
        angle_degrees(record["corners"], width, height)
    elif not isinstance(record["reason"], str) or not record["reason"].strip():
        raise ValueError("exclusion requires a nonempty reason")
    return record
