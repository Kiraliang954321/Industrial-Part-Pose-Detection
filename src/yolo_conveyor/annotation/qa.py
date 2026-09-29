"""Read-only annotation sidecar and current capture integrity audit."""

from dataclasses import asdict, dataclass
from pathlib import Path

from yolo_conveyor.label_contract import canonical_class_name

from .contract import SHA256, angle_degrees, jpeg_dimensions, record_path, safe_file, safe_source_path, sha256_file
from .store import AnnotationSession, load_source


@dataclass(frozen=True)
class QAResult:
    expected: int
    annotated: int
    excluded: int
    missing: int
    angle_evaluable: int
    angle_unevaluable: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def audit_annotations(workspace: str | Path) -> QAResult:
    """No writes; collect errors even if individual records are corrupt."""
    store = AnnotationSession(workspace)
    errors: list[str] = []
    warnings: list[str] = []
    expected: dict[str, str] = {}
    annotated = excluded = missing = evaluable = unevaluable = 0
    try:
        meta = store.metadata()
        source = store.source()
        for name, filename in (("source_session_sha256", "session.json"),
                               ("source_manifest_sha256", "manifest.jsonl")):
            value = meta.get(name)
            if not isinstance(value, str) or not SHA256.fullmatch(value) or value != sha256_file(source / filename):
                errors.append(f"{filename} hash drift or invalid metadata hash")
        session, expected = load_source(source)
        if type(meta.get("schema_version")) is not int or meta["schema_version"] != 1:
            errors.append("invalid annotation schema_version")
        if (meta.get("source_session_id") != session["session_id"] or
                meta.get("part_id") != session["part_id"] or
                meta.get("class_names") != session["class_names"]):
            errors.append("annotation metadata inconsistent with source session")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"cannot validate source/annotation metadata: {exc}")
        return QAResult(0, 0, 0, 0, 0, 0, tuple(errors), tuple(warnings))

    records_root = store.path / "records"
    expected_records = {record_path(name).as_posix(): name for name in expected}
    if not records_root.is_dir() or records_root.is_symlink():
        errors.append("missing or unsafe records directory")
    else:
        for entry in sorted(records_root.rglob("*")):
            relative = entry.relative_to(store.path).as_posix()
            try:
                safe_file(store.path, entry.relative_to(store.path))
                if entry.is_file() and relative not in expected_records:
                    errors.append(f"unexpected or unsafe review record: {relative}")
            except ValueError as exc:
                errors.append(f"unsafe record path {relative}: {exc}")
    for name, face in expected.items():
        try:
            target = store.record_file(name)
        except ValueError as exc:
            errors.append(f"{name}: unsafe review path: {exc}")
            continue
        if not target.exists():
            missing += 1
            errors.append(f"missing review record: {name}")
            continue
        try:
            record = store.read_record(name)
            if (record["part_id"] != session["part_id"] or record["face"] != face or
                    record["class_name"] != canonical_class_name(session["part_id"], face)):
                errors.append(f"{name}: record identity/face/class mismatch")
            image = safe_file(source, safe_source_path(name, face))
            if record["source_image_sha256"] != sha256_file(image):
                errors.append(f"{name}: source image hash drift")
            if jpeg_dimensions(image) != (record["image_width"], record["image_height"]):
                errors.append(f"{name}: source image dimensions mismatch")
            if record["status"] == "annotated":
                annotated += 1
                angle = angle_degrees(record["corners"], record["image_width"], record["image_height"])
                if angle is None:
                    unevaluable += 1
                    warnings.append(f"{name}: angle_evaluable=false (near-square)")
                else:
                    evaluable += 1
            else:
                excluded += 1
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"{name}: invalid review record or source image: {exc}")
    return QAResult(len(expected), annotated, excluded, missing, evaluable, unevaluable,
                    tuple(errors), tuple(warnings))
