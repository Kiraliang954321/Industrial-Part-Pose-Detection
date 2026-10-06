"""Read-only preflight for the isolated synthetic demonstration dataset."""

import hashlib
import json
import math
from pathlib import Path

from yolo_conveyor.annotation.contract import sha256_file
from yolo_conveyor.annotation.qa import audit_annotations
from yolo_conveyor.annotation.store import AnnotationSession

SPLITS = ("train", "val", "test", "final_holdout")
PART_ID = "synthetic_cap_a"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")


def contained_path(root: Path, path: str | Path, *, directory: bool = False) -> Path:
    resolved = (root / path).resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"path escapes bundle: {path}")
    if (directory and not resolved.is_dir()) or (not directory and not resolved.is_file()):
        raise ValueError(f"invalid {'directory' if directory else 'file'}: {path}")
    return resolved


def _label(path: Path, class_id: int) -> list[float]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if len(rows) != 1:
        raise ValueError(f"expected one generated OBB per image: {path}")
    cells = rows[0].split()
    if len(cells) != 9 or cells[0] != str(class_id):
        raise ValueError(f"invalid OBB class or shape: {path}")
    points = [float(cell) for cell in cells[1:]]
    if any(not math.isfinite(point) or not 0 <= point <= 1 for point in points):
        raise ValueError(f"invalid normalized OBB: {path}")
    return points


def validate_bundle(bundle: str | Path) -> dict:
    """Recompute hashes and session assignments; the formal gate stays closed."""
    root = Path(bundle).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("bundle must be a directory")
    provenance_path = contained_path(root, "synthetic_provenance.json")
    provenance = read_json(provenance_path)
    if (provenance.get("evidence_scope") != "synthetic_only" or
            provenance.get("part_id") != PART_ID or
            provenance.get("orientation_semantics") != "NOT_APPLICABLE" or
            not provenance.get("generator_version") or type(provenance.get("seed")) is not int):
        raise ValueError("bundle requires explicit synthetic provenance and N/A orientation")
    dataset = contained_path(root, "dataset", directory=True)
    manifest_path = contained_path(root, "dataset/dataset_manifest.json")
    manifest = read_json(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != provenance.get("dataset_manifest_sha256"):
        raise ValueError("dataset manifest hash differs from synthetic provenance")
    names = {str(i): f"{PART_ID}_{face}" for i, face in enumerate(("front", "back"))}
    if (manifest.get("schema_version") != 2 or manifest.get("part_id") != PART_ID or
            manifest.get("training_gate") != "closed" or manifest.get("names") != names or
            set(manifest.get("splits", {})) != set(SPLITS)):
        raise ValueError("expected schema-v2 synthetic OBB dataset with formal gate closed")
    leakage_path = contained_path(root, "leakage_report.json")
    leakage = read_json(leakage_path)
    if sha256_file(leakage_path) != provenance.get("leakage_report_sha256"):
        raise ValueError("leakage report hash differs from synthetic provenance")
    if (type(leakage.get("exact_cross_split_duplicates")) is not int or
            leakage["exact_cross_split_duplicates"] != 0):
        raise ValueError("leakage report must declare zero exact cross-split duplicates")
    sessions = provenance.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("synthetic sessions are required")
    by_id = {}
    assignment = {}
    for split in SPLITS:
        ids = manifest["splits"][split]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{split} has no sessions")
        for session_id in ids:
            if not isinstance(session_id, str) or session_id in assignment:
                raise ValueError("duplicate or invalid session assignment")
            assignment[session_id] = split
    inputs = {item["source_session_id"]: item for item in manifest.get("inputs", [])}
    if set(inputs) != set(assignment):
        raise ValueError("manifest inputs differ from split assignments")
    for item in sessions:
        if (not isinstance(item, dict) or type(item.get("seed")) is not int or
                item.get("session_id") in by_id):
            raise ValueError("invalid or duplicate synthetic session")
        session_id = item["session_id"]
        if assignment.get(session_id) != item.get("split"):
            raise ValueError("synthetic provenance does not match session split")
        source = contained_path(root, item["source_session"], directory=True)
        source_json = contained_path(root, source / "session.json")
        source_manifest = contained_path(root, source / "manifest.jsonl")
        metadata = read_json(source_json)
        if (metadata.get("session_id") != session_id or metadata.get("part_id") != PART_ID or
                metadata.get("config", {}).get("evidence_scope") != "synthetic_only" or
                not metadata.get("source", "").startswith("synthetic://")):
            raise ValueError("every source session must be explicitly synthetic")
        entry = inputs[session_id]
        if (sha256_file(source_json) != entry["source_session_sha256"] or
                sha256_file(source_manifest) != entry["source_manifest_sha256"] or
                Path(entry["source_session"]).resolve() != source):
            raise ValueError("source session metadata drift")
        workspace = contained_path(root, entry["annotation_workspace"], directory=True)
        annotation_json = contained_path(root, workspace / "annotation.json")
        if sha256_file(annotation_json) != entry["annotation_sha256"]:
            raise ValueError("source annotation workspace drift")
        qa = audit_annotations(workspace)
        if qa.errors:
            raise ValueError(f"synthetic annotation QA failed for {session_id}: {'; '.join(qa.errors)}")
        video = contained_path(root, item["video_path"])
        if sha256_file(video) != item.get("video_sha256"):
            raise ValueError("source video hash drift")
        video_gt = contained_path(root, item["video_gt_path"])
        if sha256_file(video_gt) != item.get("video_gt_sha256"):
            raise ValueError("source video ground truth hash drift")
        by_id[session_id] = {**item, "source_session": str(source), "video_path": str(video),
                            "video_gt_path": str(video_gt), "annotation_workspace": str(workspace)}
    if set(by_id) != set(assignment):
        raise ValueError("synthetic provenance has missing or unknown sessions")
    image_hashes: dict[str, str] = {}
    snapshot = hashlib.sha256()
    checked_counts = {}
    for split in SPLITS:
        directory = contained_path(root, dataset / "images" / split, directory=True)
        samples = sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        face_counts = {"front": 0, "back": 0}
        for sample in samples:
            image = contained_path(root, sample)
            relative = sample.relative_to(directory)
            if (len(relative.parts) < 3 or assignment.get(relative.parts[0]) != split or
                    relative.parts[1] not in face_counts):
                raise ValueError(f"image violates session/face path contract: {sample}")
            session_id, face = relative.parts[:2]
            label = contained_path(root, dataset / "labels" / split / relative.with_suffix(".txt"))
            corners = _label(label, ("front", "back").index(face))
            image_hash = sha256_file(image)
            if image_hash in image_hashes and image_hashes[image_hash] != split:
                raise ValueError("recomputed image hashes reveal cross-split duplicates")
            image_hashes[image_hash] = split
            source_image = contained_path(root, Path(by_id[session_id]["source_session"]) /
                                          Path(*relative.parts[1:]))
            if image_hash != sha256_file(source_image):
                raise ValueError(f"dataset image differs from synthetic source: {sample}")
            store = AnnotationSession(by_id[session_id]["annotation_workspace"])
            raw = Path(*relative.parts[1:]).as_posix()
            record_file = contained_path(root, store.record_file(raw))
            if sha256_file(record_file) != inputs[session_id]["record_sha256"].get(raw):
                raise ValueError("annotation record hash drift")
            record = store.read_record(raw)
            if (record["status"] != "annotated" or record["source_image_sha256"] != image_hash or
                    corners != [coordinate for point in record["corners"] for coordinate in point]):
                raise ValueError("assembled OBB differs from synthetic annotation record")
            snapshot.update(sample.relative_to(dataset).as_posix().encode("utf-8"))
            snapshot.update(image_hash.encode("ascii"))
            snapshot.update(sha256_file(label).encode("ascii"))
            face_counts[face] += 1
        counts = {"images": len(samples), "faces": face_counts}
        if not samples or min(face_counts.values()) == 0 or counts != manifest["counts"][split]:
            raise ValueError(f"dataset count or face coverage mismatch: {split}")
        checked_counts[split] = counts
    return {"bundle": str(root), "dataset": str(dataset), "names": names,
            "sessions": by_id, "counts": checked_counts, "evidence_scope": "synthetic_only",
            "training_gate": "closed", "orientation_semantics": "NOT_APPLICABLE",
            "dataset_manifest_sha256": manifest_hash, "dataset_content_sha256": snapshot.hexdigest(),
            "synthetic_provenance_sha256": sha256_file(provenance_path),
            "exact_cross_split_duplicates": 0}


def training_yaml(path: Path, info: dict, *, holdout: bool = False) -> Path:
    """Write absolute paths; the normal training YAML never exposes Final Holdout."""
    dataset = Path(info["dataset"])
    lines = [f"path: {json.dumps(dataset.as_posix())}"]
    for split in ("train", "val", "test"):
        target = "final_holdout" if holdout and split == "test" else split
        lines.append(f"{split}: {json.dumps((dataset / 'images' / target).as_posix())}")
    lines.append("names:")
    lines.extend(f"  {key}: {json.dumps(value)}" for key, value in info["names"].items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path.resolve()
