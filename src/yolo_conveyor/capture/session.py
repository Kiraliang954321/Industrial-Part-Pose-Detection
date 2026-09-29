"""Filesystem metadata for raw front/back capture sessions (no image I/O)."""

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Mapping
from uuid import uuid4


_LABELS = ("front", "back")
_MODES = ("manual", "auto")
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_line(value: object) -> str:
    # Serialize before touching the manifest: invalid JSON values cannot leave a partial row.
    return json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"


@dataclass(frozen=True)
class CaptureSession:
    """A session whose attempt log is append-only; callers save image bytes separately."""

    path: Path

    @classmethod
    def create(
        cls,
        root: str | Path,
        *,
        source: str,
        backend: str,
        config: Mapping[str, object],
        session_id: str | None = None,
    ) -> "CaptureSession":
        """Create a new session below root, failing if the ID is already in use."""
        if session_id is None:
            session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex
        if not isinstance(session_id, str) or not _ID_PATTERN.fullmatch(session_id):
            raise ValueError("session_id must be a safe single directory name")
        if not isinstance(source, str) or not isinstance(backend, str):
            raise ValueError("source and backend must be strings")
        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping")
        metadata = {
            "schema_version": 1,
            "session_id": session_id,
            "state": "raw_capture",
            "labels": list(_LABELS),
            "source": source,
            "backend": backend,
            "config": dict(config),
            "started_at": _timestamp(),
            "obb_labeled": False,
            "train_ready": False,
        }
        encoded = _json_line(metadata)
        base = Path(root)
        base.mkdir(parents=True, exist_ok=True)
        path = base / session_id
        path.mkdir()  # exclusive: never overwrite a pre-existing session
        (path / "front").mkdir()
        (path / "back").mkdir()
        (path / "session.json").write_text(encoded, encoding="utf-8")
        (path / "manifest.jsonl").touch(exist_ok=False)
        return cls(path)

    def record_attempt(
        self,
        *,
        label: str,
        mode: str,
        accepted: bool,
        saved_path: str | Path | None,
        quality_metrics: object,
        warnings: list[str] | tuple[str, ...] = (),
        rejection_reasons: list[str] | tuple[str, ...] = (),
        hash_value: str | None = None,
        nearest_duplicate_distance: int | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Append one attempt, validating the path without writing any image pixels."""
        if label not in _LABELS:
            raise ValueError("label must be 'front' or 'back'")
        if mode not in _MODES:
            raise ValueError("mode must be 'manual' or 'auto'")
        if not isinstance(accepted, bool):
            raise ValueError("accepted must be a boolean")
        if isinstance(quality_metrics, Mapping):
            metrics = dict(quality_metrics)
        elif is_dataclass(quality_metrics) and not isinstance(quality_metrics, type):
            metrics = asdict(quality_metrics)
        else:
            raise ValueError("quality_metrics must be a mapping or dataclass instance")
        if (not isinstance(warnings, (list, tuple)) or
                any(not isinstance(w, str) for w in warnings) or
                not isinstance(rejection_reasons, (list, tuple)) or
                any(not isinstance(r, str) for r in rejection_reasons)):
            raise ValueError("warnings and rejection_reasons must be sequences of strings")
        if hash_value is not None and not isinstance(hash_value, str):
            raise ValueError("hash_value must be a string or None")
        if (nearest_duplicate_distance is not None and
                (isinstance(nearest_duplicate_distance, bool) or
                 not isinstance(nearest_duplicate_distance, int) or
                 nearest_duplicate_distance < 0)):
            raise ValueError("nearest_duplicate_distance must be a nonnegative integer or None")
        if timestamp is not None and not isinstance(timestamp, str):
            raise ValueError("timestamp must be a string or None")
        if accepted:
            if not isinstance(saved_path, (str, Path)):
                raise ValueError("accepted attempts require a relative saved_path")
            raw = str(saved_path)
            candidate = Path(raw)
            # Windows drive and UNC paths must also fail on non-Windows hosts.
            if (not raw or candidate.is_absolute() or PureWindowsPath(raw).is_absolute() or
                    PureWindowsPath(raw).drive or "\\" in raw or
                    any(part in (".", "..") for part in raw.split("/")) or
                    not candidate.parts or candidate.parts[0] != label or
                    len(candidate.parts) < 2 or
                    not (self.path / candidate).resolve().is_relative_to(self.path.resolve())):
                raise ValueError("saved_path must be relative within the session label directory")
            relative_path: str | None = candidate.as_posix()
        else:
            if saved_path is not None:
                raise ValueError("rejected attempts must have saved_path=None")
            relative_path = None
        row = {
            "timestamp": timestamp if timestamp is not None else _timestamp(),
            "label": label,
            "mode": mode,
            "accepted": accepted,
            "saved_path": relative_path,
            "quality_metrics": metrics,
            "warnings": list(warnings),
            "rejection_reasons": list(rejection_reasons),
            "hash": hash_value,
            "nearest_duplicate_distance": nearest_duplicate_distance,
        }
        encoded = _json_line(row)
        with (self.path / "manifest.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
