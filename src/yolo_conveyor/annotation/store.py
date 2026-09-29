"""Separate, replaceable review records over immutable raw capture sessions."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from yolo_conveyor.label_contract import FACE_LABELS, canonical_class_name, validate_part_id

from .contract import jpeg_dimensions, record_path, safe_file, safe_source_path, sha256_file, validate_record


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def load_source(source: Path) -> tuple[dict, dict[str, str]]:
    """Validate a V2 session and return accepted path -> face only."""
    session = json.loads((source / "session.json").read_text(encoding="utf-8"))
    if not isinstance(session, dict) or type(session.get("schema_version")) is not int or session["schema_version"] != 2:
        raise ValueError("source must be a schema-v2 capture session")
    if not isinstance(session.get("session_id"), str) or not _ID.fullmatch(session["session_id"]):
        raise ValueError("invalid source session ID")
    validate_part_id(session.get("part_id"))
    if (session.get("state") != "raw_capture" or session.get("face_labels") != list(FACE_LABELS) or
            session.get("manifest_label_semantics") != "face_label" or
            session.get("class_names") != {f: canonical_class_name(session["part_id"], f) for f in FACE_LABELS}):
        raise ValueError("inconsistent source session metadata")
    expected: dict[str, str] = {}
    stems: set[str] = set()
    with (source / "manifest.jsonl").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or type(row.get("accepted")) is not bool or row.get("label") not in FACE_LABELS:
                    raise ValueError("invalid attempt")
                if row["accepted"]:
                    if row.get("saved_path") is None:
                        raise ValueError("accepted attempt missing path")
                    path = safe_source_path(row["saved_path"], row["label"])
                    safe_file(source, path)
                    key = path.as_posix()
                    stem = path.with_suffix("").as_posix()
                    if key in expected or stem in stems:
                        raise ValueError("duplicate accepted image or review stem")
                    expected[key] = row["label"]
                    stems.add(stem)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"invalid manifest row {number}: {exc}") from exc
    return session, expected


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


class AnnotationSession:
    """One workspace per raw session; metadata and reviews never write into source."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def create(cls, path: str | Path, source: str | Path) -> "AnnotationSession":
        source = Path(source).resolve(strict=True)
        session, _ = load_source(source)
        path = Path(path).resolve()
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError("annotation workspace must be separate from source session")
        metadata = {
            "schema_version": 1, "source_session": str(source),
            "source_session_id": session["session_id"], "part_id": session["part_id"],
            "class_names": session["class_names"],
            "source_session_sha256": sha256_file(source / "session.json"),
            "source_manifest_sha256": sha256_file(source / "manifest.jsonl"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        encoded = _json_bytes(metadata)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()
        try:
            (path / "records").mkdir()
            (path / "annotation.json").write_bytes(encoded)
        except Exception:
            # Do not leave an apparently valid partial workspace.
            (path / "records").rmdir()
            path.rmdir()
            raise
        return cls(path)

    def metadata(self) -> dict:
        data = json.loads((self.path / "annotation.json").read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("annotation metadata must be an object")
        return data

    def source(self) -> Path:
        raw = self.metadata().get("source_session")
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise ValueError("invalid source location")
        return Path(raw)

    def record_file(self, source_path: str) -> Path:
        return safe_file(self.path, record_path(source_path))

    def read_record(self, source_path: str) -> dict:
        record = json.loads(self.record_file(source_path).read_text(encoding="utf-8"))
        validate_record(record)
        if record["source_path"] != source_path:
            raise ValueError("record path does not match source path")
        return record

    def write_record(self, record: dict) -> Path:
        validate_record(record)
        meta = self.metadata()
        source = self.source()
        session, expected = load_source(source)
        if (sha256_file(source / "session.json") != meta["source_session_sha256"] or
                sha256_file(source / "manifest.jsonl") != meta["source_manifest_sha256"]):
            raise ValueError("source metadata has drifted")
        path = record["source_path"]
        if (meta["source_session_id"] != session["session_id"] or
                meta["part_id"] != session["part_id"] or
                meta["class_names"] != session["class_names"] or
                expected.get(path) != record["face"] or
                record["part_id"] != session["part_id"] or
                record["class_name"] != session["class_names"][record["face"]]):
            raise ValueError("record does not match accepted source image")
        image = safe_file(source, safe_source_path(path))
        if sha256_file(image) != record["source_image_sha256"]:
            raise ValueError("source image hash mismatch")
        if jpeg_dimensions(image) != (record["image_width"], record["image_height"]):
            raise ValueError("source image dimensions mismatch")
        target = self.record_file(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Revalidate after mkdir to avoid following a newly introduced link.
        target = self.record_file(path)
        encoded = _json_bytes(record)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=".review-",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return target
