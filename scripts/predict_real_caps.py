"""Apply the selected development YOLO + Part A yellow-support perception filter."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


SCOPE = "real_photo_development"
NAMES = {0: "cap_a_front", 1: "cap_a_back"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
LIMITATIONS = [
    "YOLO + cap_a_color_filter is a calibrated development perception pipeline",
    "Raw YOLO detections are retained separately from filtered detections",
    "Yellow support does not resolve competing front/back detections or prove correctness",
    "No Ground Truth, accuracy, formal Test, event decision, latency or throughput evaluation",
    "This filter is specific to yellow Part A; Part B changeover is not supported",
    "Part A orientation is NOT_APPLICABLE; the formal Training Gate remains CLOSED",
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate selection JSON key: {key}")
        result[key] = value
    return result


def _unlinked(raw: str | Path, *, file: bool = False) -> Path:
    path = Path(raw).absolute()
    for item in (path, *path.parents):
        try:
            metadata = item.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(metadata.st_mode) or
                getattr(metadata, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"links/reparse paths are not supported: {item}")
        if item != path and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"path ancestor is not a directory: {item}")
    path = path.resolve()
    if file and not path.is_file():
        raise ValueError(f"expected an existing regular file: {path}")
    return path


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def _number(value: object, name: str, *, positive: bool = True) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not (0 < value <= 1 if positive else 0 <= value <= 1)):
        raise ValueError(f"{name} must be a finite number in {'(0,1]' if positive else '[0,1]'}")
    return float(value)


def _selection(raw: str | Path) -> tuple[Path, bytes, dict, Path, bytes]:
    path = _unlinked(raw, file=True)
    payload = path.read_bytes()
    config = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    if (not isinstance(config, dict) or type(config.get("schema_version")) is not int or
            config["schema_version"] != 1 or config.get("evidence_scope") != SCOPE or
            config.get("formal_training_gate") != "closed" or config.get("part_id") != "cap_a" or
            config.get("orientation_semantics") != "NOT_APPLICABLE"):
        raise ValueError("selection must be a schema-v1 Part A development selection with formal gate CLOSED")
    model_raw = config.get("model_path")
    if not isinstance(model_raw, str) or not Path(model_raw).is_absolute():
        raise ValueError("model_path must identify an existing absolute local checkpoint")
    model_path = _unlinked(model_raw, file=True)
    digest = config.get("model_sha256")
    if model_path.suffix.lower() != ".pt" or not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise ValueError("model must be a local .pt checkpoint with a SHA-256 declaration")
    model_payload = model_path.read_bytes()
    if _sha(model_payload) != digest:
        raise ValueError("selected checkpoint SHA-256 mismatch")
    _number(config.get("confidence"), "confidence")
    _number(config.get("nms_iou"), "nms_iou")
    if type(config.get("imgsz")) is not int or config["imgsz"] <= 0:
        raise ValueError("imgsz must be a positive integer")
    color = config.get("color_filter")
    if not isinstance(color, dict) or set(color) != {"hsv_lower", "hsv_upper", "min_yellow_fraction"}:
        raise ValueError("color_filter must declare HSV bounds and min_yellow_fraction")
    for name in ("hsv_lower", "hsv_upper"):
        values = color[name]
        if (not isinstance(values, list) or len(values) != 3 or
                any(type(value) is not int or not 0 <= value <= maximum
                    for value, maximum in zip(values, (179, 255, 255)))):
            raise ValueError(f"invalid OpenCV HSV bound: {name}")
    if any(low > high for low, high in zip(color["hsv_lower"], color["hsv_upper"])):
        raise ValueError("HSV lower bound cannot exceed upper bound")
    _number(color["min_yellow_fraction"], "min_yellow_fraction")
    return path, payload, config, model_path, model_payload


def _sources(raw: str | Path) -> tuple[Path, list[dict]]:
    source = _unlinked(raw)
    if source.is_file():
        paths = [source]
        directory = source.parent
    elif source.is_dir():
        directory = source
        paths = []
        for item in sorted(source.iterdir(), key=lambda path: path.name.casefold()):
            checked = _unlinked(item)
            if checked.is_file() and checked.suffix.lower() in (".jpg", ".jpeg"):
                paths.append(checked)
    else:
        raise ValueError("source must be one JPEG or an existing flat directory of JPEGs")
    if not paths:
        raise ValueError("source directory contains no JPEG files; nested directories are not scanned")
    rows = []
    for path in paths:
        if path.suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError(f"only JPEG sources are supported: {path}")
        payload = _unlinked(path, file=True).read_bytes()
        if not payload.startswith(b"\xff\xd8"):
            raise ValueError(f"source is not a JPEG: {path}")
        rows.append({"source_path": str(path), "source_image_sha256": _sha(payload)})
    return directory, rows


def target_yellow_fraction(frame, corners: list, color: dict) -> tuple[float | None, str | None]:
    """Measure yellow pixels in the in-image part of a convex predicted polygon."""
    import cv2
    import numpy as np

    points = np.asarray(corners, dtype=np.float64)
    if (points.shape != (4, 2) or not np.isfinite(points).all() or
            np.max(np.abs(points)) > 1e8):
        return None, "invalid_polygon"
    points = points.astype(np.float32)
    if not cv2.isContourConvex(points) or abs(float(cv2.contourArea(points))) <= 0:
        return None, "invalid_polygon"
    height, width = frame.shape[:2]
    bounds = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    area, clipped = cv2.intersectConvexConvex(points, bounds)
    if clipped is None or not math.isfinite(area) or area <= 0:
        return None, "empty_polygon_mask"
    polygon = clipped.reshape(-1, 2)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
    denominator = int(cv2.countNonZero(mask))
    if denominator <= 0:
        return None, "empty_polygon_mask"
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array(color["hsv_lower"], dtype=np.uint8),
                         np.array(color["hsv_upper"], dtype=np.uint8))
    numerator = int(cv2.countNonZero(cv2.bitwise_and(yellow, mask)))
    return float(numerator / denominator), None


def _detections(result, frame, color: dict) -> tuple[list, list, list]:
    raw, accepted, rejected = [], [], []
    if result.obb is None:
        return raw, accepted, rejected
    corner_rows = result.obb.xyxyxyxy.cpu().tolist()
    classes = result.obb.cls.cpu().tolist()
    scores = result.obb.conf.cpu().tolist()
    if not len(corner_rows) == len(classes) == len(scores):
        raise ValueError("model returned inconsistent OBB tensor lengths")
    for index, (corners, class_value, score) in enumerate(zip(corner_rows, classes, scores)):
        class_id = int(class_value) if math.isfinite(class_value) and class_value in NAMES else None
        fraction, reason = target_yellow_fraction(frame, corners, color)
        if class_id is None or not math.isfinite(score) or not 0 <= score <= 1:
            reason = "invalid_model_prediction"
        face = NAMES[class_id].rsplit("_", 1)[-1] if class_id is not None else None
        record = {"detection_index": index, "class_id": class_id,
                  "class_name": NAMES.get(class_id), "face": face,
                  "confidence": float(score) if math.isfinite(score) else None,
                  "corners_pixels": [[float(value) if math.isfinite(value) else None
                                      for value in point] for point in corners],
                  "yellow_fraction": fraction, "angle": None, "angle_label": "N/A",
                  "orientation_semantics": "NOT_APPLICABLE"}
        raw.append(record)
        if reason is not None:
            rejected.append({**record, "rejection_reason": reason})
        elif fraction < color["min_yellow_fraction"]:
            rejected.append({**record, "rejection_reason": "rejected_yellow_support"})
        else:
            accepted.append(record)
    return raw, accepted, rejected


def _preview(frame, accepted: list, rejected: list, target: Path) -> str:
    import cv2
    import numpy as np

    plotted = frame.copy()
    height, width = plotted.shape[:2]
    scale = min(2.0, max(0.6, width / 1200))
    thickness = max(2, round(scale * 2))
    for detections, keep in ((rejected, False), (accepted, True)):
        for row in detections:
            points = row["corners_pixels"]
            if any(value is None for point in points for value in point):
                continue
            polygon = np.asarray(points, dtype=np.float64)
            if polygon.shape != (4, 2) or np.max(np.abs(polygon)) > 1e8:
                continue
            draw = np.clip(polygon, [-width, -height], [2 * width, 2 * height]).astype(np.int32)
            color = ((60, 210, 90) if row["face"] == "front" else (255, 170, 40)) if keep else (160, 160, 160)
            cv2.polylines(plotted, [draw], True, color, thickness, cv2.LINE_AA)
            fraction, score = row["yellow_fraction"], row["confidence"]
            text = f"{'KEEP' if keep else 'REJECT'} {row['face']}"
            if score is not None:
                text += f" {score:.2f}"
            if fraction is not None:
                text += f" yellow={fraction:.2f}"
            x = max(0, min(width - 1, int(np.min(polygon[:, 0]))))
            y = max(round(30 * scale), min(height - 1, int(np.min(polygon[:, 1])) - 8))
            cv2.putText(plotted, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        thickness, cv2.LINE_AA)
    header = "YOLO + cap_a_color_filter | DEVELOPMENT | Angle N/A | Formal gate CLOSED"
    cv2.putText(plotted, header, (12, max(24, round(28 * scale))), cv2.FONT_HERSHEY_SIMPLEX,
                scale * 0.7, (255, 255, 255), thickness, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise OSError("could not encode prediction preview")
    payload = encoded.tobytes()
    _write_new(target, payload)
    return _sha(payload)


def run(args) -> dict:
    selection_path, selection_payload, config, model_path, model_payload = _selection(args.selection)
    source_directory, inputs = _sources(args.source)
    output = _unlinked(args.output)
    if (os.path.lexists(output) or output == source_directory or
            output.is_relative_to(source_directory) or source_directory.is_relative_to(output) or
            model_path.is_relative_to(output) or selection_path.is_relative_to(output)):
        raise ValueError("output must be entirely new and separate from sources, selection and checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    config_directory = output / "ultralytics-config"
    config_directory.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.pop("ULTRALYTICS_API_KEY", None)
    # Local checkpoints are trusted pickle inputs; snapshot bytes before loading.
    snapshot_directory = output / "input-snapshot"
    snapshot_directory.mkdir()
    weight_snapshot = snapshot_directory / "model.pt"
    _write_new(weight_snapshot, model_payload)
    _write_new(snapshot_directory / "operating-selection.json", selection_payload)
    if _sha(weight_snapshot.read_bytes()) != config["model_sha256"]:
        raise ValueError("checkpoint snapshot SHA-256 mismatch")
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS.update({key: False for key in ("sync", "wandb", "comet", "mlflow", "clearml", "dvc",
                                           "raytune", "tensorboard") if key in SETTINGS})
    SETTINGS.update({"api_key": ""})
    model = YOLO(str(weight_snapshot))
    if model.task != "obb" or {int(key): value for key, value in model.names.items()} != NAMES:
        raise ValueError("selected model must be Part A OBB with cap_a_front/cap_a_back class mapping")
    previews = output / "previews"
    previews.mkdir()
    rows = []
    for index, original in enumerate(inputs):
        source = _unlinked(original["source_path"], file=True)
        payload = source.read_bytes()
        if _sha(payload) != original["source_image_sha256"]:
            raise ValueError(f"source changed after preflight: {source}")
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8),
                             cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if frame is None or frame.ndim != 3 or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError(f"source JPEG cannot be decoded: {source}")
        result = model.predict(frame, imgsz=config["imgsz"], device=args.device,
                               conf=config["confidence"], iou=config["nms_iou"],
                               agnostic_nms=False, augment=False, verbose=False)[0]
        raw, accepted, rejected = _detections(result, frame, config["color_filter"])
        preview = previews / f"{index:04d}_{source.stem}.jpg"
        preview_hash = _preview(frame, accepted, rejected, preview)
        rows.append({**original, "image_width": int(frame.shape[1]), "image_height": int(frame.shape[0]),
                     "coordinate_system": "JPEG encoded pixel orientation; EXIF orientation ignored",
                     "raw_detections": raw, "accepted_detections": accepted, "rejected": rejected,
                     "preview_path": str(preview), "preview_sha256": preview_hash})
    report = {"schema_version": 1, "evidence_scope": SCOPE, "formal_training_gate": "closed",
              "part_id": "cap_a", "pipeline": "YOLO+cap_a_color_filter", "completed_at": _utc(),
              "selection_path": str(selection_path), "selection_sha256": _sha(selection_payload),
              "selection": config, "model_path": str(model_path), "model_sha256": config["model_sha256"],
              "device": args.device, "orientation_semantics": "NOT_APPLICABLE", "image_count": len(rows),
              "formal_dataset_v1": False, "test_evaluated": False, "final_holdout_evaluated": False,
              "filter_scope": "In-image OBB polygon yellow pixel fraction only; no GT or class suppression",
              "files": rows, "limitations": LIMITATIONS}
    _write_new(output / "predictions.json",
               (json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="frozen development operating selection")
    parser.add_argument("--source", type=Path, required=True, help="one JPEG or a flat directory of JPEGs")
    parser.add_argument("--output", type=Path, required=True, help="entirely new result directory")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"development photo inference failed: {exc}\n")
    print(f"Saved {report['image_count']} development photo predictions: {args.output.resolve() / 'predictions.json'}")
    print("Pipeline: YOLO+cap_a_color_filter. Formal gate CLOSED; Angle N/A; no GT/event metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
