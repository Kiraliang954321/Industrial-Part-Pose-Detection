"""Shared V2 part/face naming rules; no numeric class IDs are assigned here."""

import re


FACE_LABELS = ("front", "back")
_PART_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def validate_part_id(part_id: str) -> str:
    """Return a safe, nonempty single-token part ID or raise ValueError."""
    if not isinstance(part_id, str) or not _PART_ID_PATTERN.fullmatch(part_id):
        raise ValueError("part_id must match [A-Za-z0-9][A-Za-z0-9_-]*")
    return part_id


def canonical_class_name(part_id: str, face: str) -> str:
    """Derive the future class name for a validated part and face label."""
    validate_part_id(part_id)
    if face not in FACE_LABELS:
        raise ValueError("face must be 'front' or 'back'")
    return f"{part_id}_{face}"
