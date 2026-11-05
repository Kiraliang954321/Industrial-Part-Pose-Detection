"""ROI-assisted geometric candidates; never accepted annotations or training GT."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import stat

import cv2
import numpy as np

from .annotation.contract import SHA256, safe_file, safe_source_path, sha256_file, validate_corners
from .annotation.store import load_source


def _no_links(path: Path) -> None:
    for item in (path, *path.parents):
        try:
            metadata = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"symlink/reparse paths are not supported: {item}")
        if item != path and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"path ancestor is not a directory: {item}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate ROI key: {key}")
        result[key] = value
    return result


def _roi_pixels(roi: object, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(roi, list) or len(roi) != 4 or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) for v in roi):
        raise ValueError("ROI must be normalized [x0,y0,x1,y1] with finite numbers")
    try:
        values = [float(v) for v in roi]
    except (ValueError, OverflowError) as exc:
        raise ValueError("ROI values must be representable finite numbers in [0,1]") from exc
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("ROI values must be finite numbers in [0,1]")
    x0, y0, x1, y1 = values
    if x0 >= x1 or y0 >= y1:
        raise ValueError("ROI must have positive width and height")
    pixels = (math.floor(x0 * width), math.floor(y0 * height),
              math.ceil(x1 * width), math.ceil(y1 * height))
    if pixels[2] - pixels[0] < 3 or pixels[3] - pixels[1] < 3:
        raise ValueError("ROI must span at least 3 pixels on each axis")
    return pixels


def _read_image(payload: bytes, name: str) -> np.ndarray:
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError(f"source is not a JPEG: {name}")
    try:
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    except cv2.error as exc:
        raise ValueError(f"cannot decode JPEG: {name}") from exc
    if image is None:
        raise ValueError(f"cannot decode JPEG: {name}")
    return image


def _frozen_hashes(session: dict, accepted: dict[str, str]) -> dict[str, str]:
    config = session.get("config")
    provenance = config.get("original_provenance") if isinstance(config, dict) else None
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("photo capture must include frozen config.original_provenance")
    frozen = {}
    for entry in provenance:
        if not isinstance(entry, dict):
            raise ValueError("invalid original provenance entry")
        original, digest = entry.get("original_relative_path"), entry.get("source_sha256")
        if not isinstance(original, str) or not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError("original provenance requires a safe path and valid SHA-256")
        matches = [face for face in ("front", "back")
                   if original.startswith(f"originals/{session['class_names'][face]}/")]
        if len(matches) != 1:
            raise ValueError("original provenance does not match source part/face")
        face = matches[0]
        prefix = f"originals/{session['class_names'][face]}/"
        name = safe_source_path(f"{face}/{original[len(prefix):]}", face).as_posix()
        if name in frozen or accepted.get(name) != face:
            raise ValueError("duplicate or unexpected original provenance source path")
        frozen[name] = digest
    if set(frozen) != set(accepted):
        raise ValueError("original provenance must cover every accepted source image")
    return frozen


def _write_json(path: Path, value: dict) -> None:
    _no_links(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    _no_links(path)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError(f"cannot encode preview: {path.name}")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())


def _propose(image: np.ndarray, roi: tuple[int, int, int, int], foreground_color: str = "none") -> tuple[list, dict]:
    x0, y0, x1, y1 = roi
    crop = image[y0:y1, x0:x1]
    scale = min(1.0, 900 / max(crop.shape[:2]))
    size = (max(3, round(crop.shape[1] * scale)), max(3, round(crop.shape[0] * scale)))
    reduced = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    yellow_lower, yellow_upper = [12, 70, 35], [40, 255, 255]
    if foreground_color == "yellow":
        hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array(yellow_lower, dtype=np.uint8), np.array(yellow_upper, dtype=np.uint8))
        mask = cv2.bitwise_or(mask, yellow)  # Otsu retains pale/white highlights; HSV adds darker yellow sides.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no external contour found inside operator ROI")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        raise ValueError("largest ROI contour has zero area")
    corners = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
    # Use actual rounded resize dimensions rather than the nominal scale.
    corners[:, 0] = corners[:, 0] * crop.shape[1] / size[0] + x0
    corners[:, 1] = corners[:, 1] * crop.shape[0] / size[1] + y0
    height, width = image.shape[:2]
    normalized = (corners / np.array([width, height])).tolist()
    validate_corners(normalized)  # No clipping: out-of-bounds fits are explicit errors.
    return normalized, {
        "method": ("gray_otsu_union_yellow_hsv_open_close_largest_external_contour_min_area_rect"
                   if foreground_color == "yellow" else "gray_otsu_open_close_largest_external_contour_min_area_rect"),
        "foreground_color": foreground_color,
        "yellow_hsv_lower": yellow_lower if foreground_color == "yellow" else None,
        "yellow_hsv_upper": yellow_upper if foreground_color == "yellow" else None,
        "hsv_convention": "OpenCV H=0..179 S=0..255 V=0..255",
        "threshold": float(threshold), "threshold_type": "BINARY_OTSU",
        "max_roi_edge": 900, "processed_roi_width": size[0], "processed_roi_height": size[1],
        "morphology_kernel": [3, 3], "morphology_iterations": 1,
        "contour_area_processed_pixels": float(cv2.contourArea(contour)),
        "geometry_angle_only": True,
    }


def _preview(image: np.ndarray, roi: tuple, corners: list | None, name: str, error: str | None) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, 900 / max(height, width))
    preview = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    cv2.rectangle(preview, (round(roi[0] * scale), round(roi[1] * scale)),
                  (round(roi[2] * scale), round(roi[3] * scale)), (255, 160, 0), 2)
    if corners is not None:
        points = np.rint(np.array(corners) * [width * scale, height * scale]).astype(np.int32)
        cv2.polylines(preview, [points], True, (0, 255, 0), 2)
    lines = ["CANDIDATE / HUMAN REVIEW REQUIRED", name,
             "ERROR: " + error if error else "NOT GT / FORMAL TRAINING GATE CLOSED"]
    for index, line in enumerate(lines):
        y = 24 + index * 22
        cv2.putText(preview, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(preview, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return preview


def propose_capture(capture: Path, rois_file: Path, output: Path, *, foreground_color: str = "none") -> dict:
    """Prevalidate inputs, then produce separate candidates with mandatory human review."""
    if foreground_color not in ("none", "yellow"):
        raise ValueError("foreground_color must be none or yellow")
    for path in (capture.absolute(), rois_file.absolute(), output.absolute()):
        _no_links(path)
    source = capture.resolve(strict=True)
    rois_file = rois_file.resolve(strict=True)
    target = output.resolve()
    if not source.is_dir() or not rois_file.is_file():
        raise ValueError("capture must be a directory and ROIs must be a JSON file")
    if target.exists() or target.is_symlink():
        raise ValueError("output must be an entirely new directory")
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("proposal output must not overlap source capture")
    session_path = safe_file(source, Path("session.json"))
    manifest_path = safe_file(source, Path("manifest.jsonl"))
    _no_links(session_path)
    _no_links(manifest_path)
    session_payload, manifest_payload, roi_payload = session_path.read_bytes(), manifest_path.read_bytes(), rois_file.read_bytes()
    hashes = {"source_session_sha256": hashlib.sha256(session_payload).hexdigest(),
              "source_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
              "roi_input_sha256": hashlib.sha256(roi_payload).hexdigest()}
    session, accepted = load_source(source)
    snapshot_session = json.loads(session_payload.decode("utf-8"))
    snapshot_rows = [json.loads(line) for line in manifest_payload.decode("utf-8").splitlines()]
    if (session != snapshot_session or any(not isinstance(row, dict) or
            type(row.get("accepted")) is not bool or row.get("label") not in ("front", "back") or
            (row["accepted"] and not isinstance(row.get("saved_path"), str)) for row in snapshot_rows) or
            accepted != {row.get("saved_path"): row.get("label") for row in snapshot_rows if row.get("accepted")}):
        raise ValueError("source metadata snapshot differs from load_source validation")
    frozen = _frozen_hashes(session, accepted)
    rois = json.loads(roi_payload.decode("utf-8"), object_pairs_hook=_unique_object)
    if not accepted or not isinstance(rois, dict) or set(rois) != set(accepted):
        raise ValueError("ROI JSON must cover exactly every accepted source saved_path")
    inputs = []
    for name, face in accepted.items():
        image_path = safe_file(source, safe_source_path(name, face))
        _no_links(image_path)
        if not image_path.is_file():
            raise ValueError(f"source image is missing: {name}")
        payload = image_path.read_bytes()
        image_hash = hashlib.sha256(payload).hexdigest()
        if image_hash != frozen[name]:
            raise ValueError(f"source image differs from frozen original SHA-256: {name}")
        image = _read_image(payload, name)
        height, width = image.shape[:2]
        roi = _roi_pixels(rois[name], width, height)
        inputs.append((name, face, image, width, height, roi, image_hash))
    _no_links(session_path)
    _no_links(manifest_path)
    if sha256_file(session_path) != hashes["source_session_sha256"] or sha256_file(manifest_path) != hashes["source_manifest_sha256"]:
        raise ValueError("source metadata changed during input validation")
    report = {
        "schema_version": 1, "artifact_type": "photo_obb_proposals", "source_session": str(source),
        "source_session_id": session["session_id"], "part_id": session["part_id"], **hashes,
        "created_at": datetime.now(timezone.utc).isoformat(), "state": "candidate",
        "operator_review_required": True, "operator_review_complete": False,
        "formal_gt": False, "formal_training_gate": "closed",
        "orientation_semantics": "NOT_APPLICABLE" if session["part_id"] == "cap_a" else "TBD",
        "angle_gt_evaluable": False, "foreground_color": foreground_color,
        "expected_images": len(inputs), "candidates": [], "errors": [],
        "rejected": [],  # Exclusion is a human decision; this tool never excludes images.
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    _no_links(target)
    target.mkdir()
    thumbnails = []
    for name, face, image, width, height, roi, image_hash in inputs:
        corners, error = None, None
        try:
            corners, method = _propose(image, roi, foreground_color)
            record = {key: report[key] for key in (
                "schema_version", "source_session", "source_session_id", "part_id",
                "source_session_sha256", "source_manifest_sha256", "roi_input_sha256", "state", "operator_review_required",
                "operator_review_complete", "formal_gt", "formal_training_gate",
                "orientation_semantics", "angle_gt_evaluable")}
            record.update(artifact_type="photo_obb_candidate", face=face,
                          class_name=session["class_names"][face], source_path=name,
                          source_image_sha256=image_hash, image_width=width, image_height=height,
                          roi=rois[name], roi_pixels=list(roi), proposed_corners=corners,
                          proposed_corners_pixels=(np.array(corners) * [width, height]).tolist(),
                          proposal_method=method, method_parameters={key: value for key, value in method.items() if key != "method"})
            candidate = safe_file(target, Path("candidates") / safe_source_path(name).with_suffix(".json"))
            candidate.parent.mkdir(parents=True, exist_ok=True)
            _write_json(candidate, record)
            report["candidates"].append({"source_path": name, "candidate_path": candidate.relative_to(target).as_posix()})
        except (ValueError, cv2.error) as exc:
            error = str(exc)
            report["errors"].append({"source_path": name, "error": error})
        preview = _preview(image, roi, corners, name, error)
        preview_path = safe_file(target, Path("previews") / safe_source_path(name).with_suffix(".jpg"))
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        _write_jpeg(preview_path, preview)
        tile = np.full((300, 360, 3), 30, dtype=np.uint8)
        ratio = min(360 / preview.shape[1], 276 / preview.shape[0])
        thumb = cv2.resize(preview, (round(preview.shape[1] * ratio), round(preview.shape[0] * ratio)))
        top, left = (276 - thumb.shape[0]) // 2, (360 - thumb.shape[1]) // 2
        tile[top:top + thumb.shape[0], left:left + thumb.shape[1]] = thumb
        cv2.putText(tile, f"{face}/{Path(name).stem}", (6, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        thumbnails.append(tile)
    _no_links(session_path)
    _no_links(manifest_path)
    if sha256_file(session_path) != hashes["source_session_sha256"] or sha256_file(manifest_path) != hashes["source_manifest_sha256"]:
        report["errors"].append({"error": "source session/manifest hash drifted during proposal generation"})
    sheet = np.full((math.ceil(len(thumbnails) / 4) * 300, 4 * 360, 3), 30, dtype=np.uint8)
    for index, tile in enumerate(thumbnails):
        row, column = divmod(index, 4)
        sheet[row * 300:(row + 1) * 300, column * 360:(column + 1) * 360] = tile
    _write_jpeg(target / "contact-sheet.jpg", sheet)
    report["summary"] = {"candidates": len(report["candidates"]), "rejected": len(report["rejected"]),
                         "errors": len(report["errors"])}
    _write_json(target / "proposal-report.json", report)
    return report
