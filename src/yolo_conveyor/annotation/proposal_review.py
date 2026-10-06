"""Read geometric candidates for operator review; never create accepted records."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import stat

from .contract import SHA256, jpeg_dimensions, safe_file, safe_source_path, sha256_file, validate_corners
from .store import load_source


@dataclass(frozen=True)
class ReviewProposal:
    corners: tuple[tuple[float, float], ...]
    source_image_sha256: str
    image_width: int
    image_height: int
    source_session_sha256: str
    source_manifest_sha256: str


def _no_links(path: Path) -> None:
    for item in (path, *path.parents):
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"symlink/reparse paths are not supported: {item}")
        if item != path and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"path ancestor is not a directory: {item}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate proposal key: {key}")
        result[key] = value
    return result


def _json_snapshot(path: Path) -> tuple[dict, str]:
    _no_links(path)
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("proposal metadata must be an object")
    return value, hashlib.sha256(payload).hexdigest()


def load_review_proposals(proposal_dir: Path, source: Path) -> dict[str, ReviewProposal]:
    """Fail closed before a workspace opens; return immutable corners and source bindings."""
    for path in (source.absolute(), proposal_dir.absolute()):
        _no_links(path)
    source, proposal_dir = source.resolve(strict=True), proposal_dir.resolve(strict=True)
    if not source.is_dir() or not proposal_dir.is_dir():
        raise ValueError("source and proposal paths must be directories")
    session_path = safe_file(source, Path("session.json"))
    manifest_path = safe_file(source, Path("manifest.jsonl"))
    snapshot_session, session_hash = _json_snapshot(session_path)
    _no_links(manifest_path)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    session, accepted = load_source(source)
    if session != snapshot_session or not accepted:
        raise ValueError("source metadata differs from its snapshot or has no accepted images")
    common = {"source_session": str(source), "source_session_id": session["session_id"],
              "part_id": session["part_id"], "source_session_sha256": session_hash,
              "source_manifest_sha256": manifest_hash}

    def validate_identity(value: dict, artifact_type: str) -> None:
        if (type(value.get("schema_version")) is not int or value["schema_version"] != 1 or
                value.get("artifact_type") != artifact_type or
                any(value.get(key) != expected for key, expected in common.items()) or
                value.get("state") != "candidate" or value.get("operator_review_required") is not True or
                value.get("operator_review_complete") is not False or value.get("formal_gt") is not False or
                value.get("formal_training_gate") != "closed" or value.get("angle_gt_evaluable") is not False):
            raise ValueError("proposal identity or mandatory review state does not match source")

    report_path = safe_file(proposal_dir, Path("proposal-report.json"))
    report, report_hash = _json_snapshot(report_path)
    validate_identity(report, "photo_obb_proposals")
    if not isinstance(report.get("orientation_semantics"), str) or not report["orientation_semantics"]:
        raise ValueError("proposal report must declare orientation semantics")
    common["orientation_semantics"] = report["orientation_semantics"]
    if (type(report.get("expected_images")) is not int or report["expected_images"] != len(accepted) or
            report.get("errors") != [] or report.get("rejected") != [] or
            not isinstance(report.get("candidates"), list) or len(report["candidates"]) != len(accepted) or
            report.get("summary") != {"candidates": len(accepted), "rejected": 0, "errors": 0}):
        raise ValueError("proposal report must contain every accepted image without errors or rejection")
    roi_hash = report.get("roi_input_sha256")
    if not isinstance(roi_hash, str) or not SHA256.fullmatch(roi_hash):
        raise ValueError("invalid frozen proposal ROI hash")
    config = session.get("config")
    provenance = config.get("original_provenance") if isinstance(config, dict) else None
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("source must include frozen config.original_provenance")
    frozen = {}
    for entry in provenance:
        if not isinstance(entry, dict):
            raise ValueError("invalid original provenance entry")
        original, digest = entry.get("original_relative_path"), entry.get("source_sha256")
        if not isinstance(original, str) or not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError("invalid original provenance path/hash")
        faces = [face for face in ("front", "back")
                 if original.startswith(f"originals/{session['class_names'][face]}/")]
        if len(faces) != 1:
            raise ValueError("original provenance does not match source part/face")
        face = faces[0]
        prefix = f"originals/{session['class_names'][face]}/"
        name = safe_source_path(f"{face}/{original[len(prefix):]}", face).as_posix()
        if name in frozen or accepted.get(name) != face:
            raise ValueError("duplicate or unexpected original provenance source")
        frozen[name] = digest
    if set(frozen) != set(accepted):
        raise ValueError("original provenance must cover every accepted image")

    proposals, snapshots = {}, [(report_path, report_hash), (session_path, session_hash),
                                 (manifest_path, manifest_hash)]
    for entry in report["candidates"]:
        if not isinstance(entry, dict) or set(entry) != {"source_path", "candidate_path"}:
            raise ValueError("candidate report entry must identify exactly source and candidate paths")
        name = safe_source_path(entry["source_path"]).as_posix()
        relative = Path("candidates") / safe_source_path(name).with_suffix(".json")
        if entry["candidate_path"] != relative.as_posix() or name not in accepted or name in proposals:
            raise ValueError("duplicate, unexpected or unsafe candidate path")
        candidate_path = safe_file(proposal_dir, relative)
        candidate, candidate_hash = _json_snapshot(candidate_path)
        snapshots.append((candidate_path, candidate_hash))
        validate_identity(candidate, "photo_obb_candidate")
        face = accepted[name]
        if (candidate.get("source_path") != name or candidate.get("face") != face or
                candidate.get("class_name") != session["class_names"][face] or
                candidate.get("roi_input_sha256") != roi_hash or
                candidate.get("source_image_sha256") != frozen[name]):
            raise ValueError(f"candidate does not match accepted source image: {name}")
        image_path = safe_file(source, safe_source_path(name, face))
        _no_links(image_path)
        width, height = jpeg_dimensions(image_path)
        if (any(type(candidate.get(key)) is not int for key in ("image_width", "image_height")) or
                (candidate["image_width"], candidate["image_height"]) != (width, height) or
                sha256_file(image_path) != frozen[name]):
            raise ValueError(f"source image differs from frozen candidate hash/dimensions: {name}")
        corners = validate_corners(candidate.get("proposed_corners"))
        pixels = candidate.get("proposed_corners_pixels")
        if not isinstance(pixels, list) or len(pixels) != 4:
            raise ValueError("candidate must include four pixel corners")
        for point, (x, y) in zip(pixels, corners):
            if (not isinstance(point, list) or len(point) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point) or
                    not 0 <= x * width <= width or not 0 <= y * height <= height or
                    not math.isclose(point[0], x * width, abs_tol=1e-4) or
                    not math.isclose(point[1], y * height, abs_tol=1e-4)):
                raise ValueError("candidate pixel corners must match normalized corners within image bounds")
        proposals[name] = ReviewProposal(corners, frozen[name], width, height, session_hash, manifest_hash)
    if set(proposals) != set(accepted):
        raise ValueError("candidate set must exactly cover accepted source images")
    for path, digest in snapshots:
        _no_links(path)
        if sha256_file(path) != digest:
            raise ValueError("source/proposal metadata changed during validation")
    return proposals
