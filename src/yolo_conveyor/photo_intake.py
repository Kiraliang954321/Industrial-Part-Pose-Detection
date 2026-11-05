"""Import verified real-photo copies into development-only V2 capture sessions."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import stat

import cv2
import numpy as np

from yolo_conveyor.capture.quality import measure_quality
from yolo_conveyor.capture.session import CaptureSession
from yolo_conveyor.label_contract import canonical_class_name, validate_part_id


CAPTURE_GROUP_ID = "observed-20261004-131421-131538"
PARTS = {
    "cap_a": ("cap_a_20261004_development01", "part_a_development", 4),
    "cap_b": ("cap_b_20261004_reference01", "part_b_candidate_reference_only", 3),
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FILENAME = re.compile(r"IMG_[0-9]{8}_[0-9]{6}\.jpg\Z", re.IGNORECASE)


def _unlinked_path(raw: str | Path) -> Path:
    """Reject links/reparse points before resolving input or output ancestors."""
    path = Path(raw).absolute()
    for item in (path, *path.parents):
        try:
            metadata = item.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(metadata.st_mode) or
                getattr(metadata, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"symlink or reparse point is not allowed: {item}")
        if item != path and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"path ancestor is not a directory: {item}")
    return path.resolve()


def _encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _preflight(intake: Path, output: Path) -> tuple[str, list[dict]]:
    if not intake.is_file():
        raise ValueError("intake must identify an existing manifest file")
    if output.exists():
        raise ValueError("output must be a new directory; existing paths are never overwritten")
    source_root = intake.parent
    if output.is_relative_to(source_root) or source_root.is_relative_to(output):
        raise ValueError("intake and output directories must not overlap")
    raw_manifest = intake.read_bytes()
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    manifest = json.loads(raw_manifest.decode("utf-8"))
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or
            manifest["schema_version"] != 1 or
            manifest.get("evidence_scope") != "real_photo_development" or
            manifest.get("formal_training_gate") != "closed" or
            manifest.get("source_originals_unchanged") is not True):
        raise ValueError("intake must be a schema-v1 real-photo development manifest with closed gate")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("intake files must be a nonempty list")
    rows, paths, families, hashes = [], set(), set(), set()
    counts = {canonical_class_name(part, face): 0 for part in PARTS for face in ("front", "back")}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each intake file must be an object")
        part = validate_part_id(entry.get("part_id"))
        face = entry.get("face")
        if part not in PARTS or face not in ("front", "back"):
            raise ValueError("only the declared cap_a/cap_b front/back intake is supported")
        class_name = canonical_class_name(part, face)
        filename = entry.get("filename")
        if not isinstance(filename, str) or not _FILENAME.fullmatch(filename):
            raise ValueError("invalid original JPEG filename")
        relative = f"originals/{class_name}/{filename}"
        if (entry.get("original_relative_path") != relative or
                entry.get("proposed_class") != class_name or entry.get("asset_kind") != "real_photo" or
                entry.get("capture_batch_id") != CAPTURE_GROUP_ID or
                entry.get("assigned_partition") != "development_unassigned"):
            raise ValueError(f"inconsistent real-photo source identity: {filename}")
        source = _unlinked_path(source_root / relative)
        if not source.is_relative_to(source_root) or not source.is_file():
            raise ValueError(f"source must be a regular file inside intake: {relative}")
        digest = entry.get("source_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid source SHA-256: {filename}")
        dimensions = (entry.get("width"), entry.get("height"))
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError(f"invalid declared dimensions: {filename}")
        notes = entry.get("quality_notes_visual")
        family = entry.get("source_family_id")
        if (not isinstance(notes, list) or any(not isinstance(note, str) for note in notes) or
                not isinstance(family, str) or not family.strip() or
                not isinstance(entry.get("label_status"), str) or
                entry.get("exif_datetime_timezone") != "unspecified"):
            raise ValueError(f"invalid source provenance: {filename}")
        capture_datetime = entry.get("capture_datetime_exif")
        if not isinstance(capture_datetime, str):
            raise ValueError(f"missing capture datetime: {filename}")
        datetime.strptime(capture_datetime, "%Y:%m:%d %H:%M:%S")
        if relative.casefold() in paths or family in families or digest in hashes:
            raise ValueError(f"duplicate source path, family or bytes: {filename}")
        paths.add(relative.casefold())
        families.add(family)
        hashes.add(digest)
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest or not payload.startswith(b"\xff\xd8"):
            raise ValueError(f"source SHA-256 or JPEG identity mismatch: {filename}")
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8),
                             cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if frame is None or (frame.shape[1], frame.shape[0]) != dimensions:
            raise ValueError(f"source JPEG decode or dimensions mismatch: {filename}")
        metrics = asdict(measure_quality(frame))
        provenance = {
            "original_relative_path": relative, "source_sha256": digest,
            "source_family_id": family, "asset_kind": "real_photo",
            "capture_datetime_exif": capture_datetime, "capture_datetime_timezone": "unspecified",
            "source_capture_group_id": CAPTURE_GROUP_ID,
            "source_label_status": entry["label_status"], "quality_notes_visual": notes,
        }
        rows.append({"part_id": part, "face": face, "class_name": class_name,
                     "filename": filename, "width": dimensions[0], "height": dimensions[1],
                     "payload": payload, "quality_metrics": metrics, "provenance": provenance})
        counts[class_name] += 1
    expected = {canonical_class_name(part, face): details[2]
                for part, details in PARTS.items() for face in ("front", "back")}
    if (counts != expected or manifest.get("counts") != expected or
            type(manifest.get("real_original_images")) is not int or
            manifest["real_original_images"] != sum(expected.values())):
        raise ValueError("intake counts must be cap_a 4 front/4 back and cap_b 3 front/3 back")
    # Ensure every future metadata value serializes before any output is created.
    _encoded([{key: value for key, value in row.items() if key != "payload"} for row in rows])
    return manifest_hash, rows


def import_photo_intake(intake: str | Path, output: str | Path) -> dict:
    """Prevalidate all images, then copy original bytes without assigning splits or GT."""
    intake, output = _unlinked_path(intake), _unlinked_path(output)
    manifest_hash, rows = _preflight(intake, output)
    import_time = datetime.now(timezone.utc).isoformat()
    sessions, image_map = [], []
    # Deliberately exclusive; a failed import is retained for inspection, never cleaned up.
    output.mkdir(parents=True, exist_ok=False)
    for part, (session_id, role, _) in PARTS.items():
        part_rows = [row for row in rows if row["part_id"] == part]
        config = {
            "evidence_scope": "real_photo_development", "role": role,
            "eligible_for_formal_dataset": False, "formal_training_gate": "closed",
            "capture_group_id": CAPTURE_GROUP_ID, "source_capture_group_id": CAPTURE_GROUP_ID,
            "conservative_physical_batch_count": 1,
            "source_intake_manifest_sha256": manifest_hash,
            "source_intake_manifest": str(intake), "imported_at": import_time,
            "started_at_semantics": "import_time_not_physical_capture_time",
            "capture_datetime_timezone": "unspecified",
            "quality_scope": "whole_frame", "quality_thresholds_calibrated": False,
            "quality_policy": "measure_only_no_automatic_rejection",
            "accepted_semantics": "imported_original_not_reviewed_annotation_or_ground_truth",
            "original_provenance": [row["provenance"] for row in part_rows],
        }
        session = CaptureSession.create(
            output / "captures", part_id=part, source=f"photo-import://{intake.parent.name}/{part}",
            backend="verified_jpeg_byte_copy", config=config, session_id=session_id,
        )
        for row in part_rows:
            relative = f"{row['face']}/{row['filename']}"
            target = session.path / relative
            with target.open("xb") as stream:
                stream.write(row["payload"])
            session.record_attempt(
                label=row["face"], mode="manual", accepted=True, saved_path=relative,
                quality_metrics=row["quality_metrics"], timestamp=import_time,
                warnings=["whole_frame_quality_uncalibrated", *row["provenance"]["quality_notes_visual"]],
            )
            image_map.append({
                **row["provenance"], "part_id": part, "face": row["face"],
                "class_name": row["class_name"], "session_id": session_id,
                "output_relative_path": target.relative_to(output).as_posix(),
                "image_width": row["width"], "image_height": row["height"],
                "quality_metrics": row["quality_metrics"], "bytes_preserved": True,
            })
        sessions.append({
            "part_id": part, "session_id": session_id, "role": role,
            "path": str(session.path), "relative_path": session.path.relative_to(output).as_posix(),
            "capture_group_id": CAPTURE_GROUP_ID, "source_capture_group_id": CAPTURE_GROUP_ID,
            "counts": {face: sum(row["face"] == face for row in part_rows) for face in ("front", "back")},
        })
    report = {
        "schema_version": 1, "evidence_scope": "real_photo_development", "imported_at": import_time,
        "source_intake_manifest": str(intake), "source_intake_manifest_sha256": manifest_hash,
        "sessions": sessions, "image_map": image_map, "real_original_images": len(rows),
        "conservative_physical_batch_count": 1, "session_folders_are_independent_physical_batches": False,
        "eligible_for_formal_dataset": False, "formal_training_gate": "closed",
        "part_b_is_phase5_formal_capture": False, "real_four_way_split_created": False,
        "training_started": False, "gate_opened": False,
        "ground_truth_written": False, "obb_annotations_written": False,
        "quality_scope": "whole_frame_uncalibrated", "quality_rejections": 0,
        "original_image_bytes_preserved_including_exif": True,
        "capture_datetime_timezone": "unspecified", "started_at_semantics": "import_time",
    }
    with (output / "import-report.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(_encoded(report))
    return report
