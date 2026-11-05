"""Single-frame synthetic video inference; never reports event-level acceptance."""

import json
import math
from pathlib import Path

from .synthetic_bundle import write_json


def _wilson(errors: int, total: int) -> dict:
    if not total:
        return {"x": errors, "n": total, "proportion": None, "wilson_95_ci": None}
    z = 1.959963984540054
    p = errors / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"x": errors, "n": total, "proportion": p,
            "wilson_95_ci": [max(0, centre - width), min(1, centre + width)]}


def _contact_sheet(frames: list, path: Path) -> None:
    import cv2
    import numpy as np

    if not frames:
        return
    tiles = []
    for frame in frames[:8]:
        scale = min(480 / frame.shape[1], 270 / frame.shape[0])
        image = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
        tile = np.zeros((270, 480, 3), dtype=np.uint8)
        x, y = (480 - image.shape[1]) // 2, (270 - image.shape[0]) // 2
        tile[y:y + image.shape[0], x:x + image.shape[1]] = image
        tiles.append(tile)
    blank = np.zeros_like(tiles[0])
    tiles.extend(blank.copy() for _ in range(8 - len(tiles)))
    sheet = np.concatenate([np.concatenate(tiles[row:row + 4], axis=1)
                            for row in (0, 4)], axis=0)
    if not cv2.imwrite(str(path), sheet):
        raise OSError(f"could not save inference contact sheet: {path}")


def validation_contact_sheet(model, dataset: Path, output: Path, *, imgsz: int,
                             device: str, confidence: float, iou: float) -> Path:
    """A small Val-only visual check for each model, before looking at Test."""
    import cv2

    samples = []
    for session in sorted((dataset / "images" / "val").iterdir())[:2]:
        for face in ("front", "back"):
            samples.extend(sorted(p for p in (session / face).rglob("*")
                                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:2])
    frames = []
    for sample in samples:
        result = model.predict(str(sample), imgsz=imgsz, device=device, conf=confidence,
                               iou=iou, verbose=False)[0]
        frame = result.plot()
        cv2.putText(frame, "SYNTHETIC / VAL / Angle N/A", (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(frame)
    target = output / "val-inference-contact-sheet.jpg"
    _contact_sheet(frames, target)
    return target


def predict_video(model, video: Path, output: Path, *, gt_path: Path, imgsz: int, device: str,
                  confidence: float, iou: float) -> dict:
    """Write MP4 and prediction JSONL; GT is used only for single-frame metrics."""
    import cv2
    import numpy as np

    ground_truth = {}
    with gt_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            frame_id = row["frame_id"]
            if type(frame_id) is not int or frame_id < 0 or frame_id in ground_truth:
                raise ValueError("invalid or duplicate video GT frame_id")
            if type(row.get("visible")) is not bool:
                raise ValueError("video GT visible must be boolean")
            if row.get("face") is None:
                if row["visible"] or row.get("corners") is not None:
                    raise ValueError("blank video GT cannot have a visible cap or corners")
            else:
                corners = row.get("corners")
                if (row["face"] not in ("front", "back") or not isinstance(corners, list) or
                        len(corners) != 4 or any(not isinstance(p, list) or len(p) != 2 for p in corners)):
                    raise ValueError("invalid video GT face or corner shape")
                if any(not isinstance(v, (int, float)) or not math.isfinite(v)
                       for point in corners for v in point):
                    raise ValueError("video GT corners must contain finite normalized coordinates")
            ground_truth[frame_id] = row
    if not ground_truth:
        raise ValueError("validated video ground truth cannot be empty")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"could not open synthetic video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("video has invalid fps or dimensions")
    target = output / "prediction-video.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV mp4v writer is unavailable")
    counts = {"visible_gt_frames": 0, "localized_frames": 0, "correct_face_frames": 0,
              "blank_gt_frames": 0, "blank_frames_with_detection": 0,
              "partial_visibility_frames": 0}
    frames = []
    timing = []
    frame_id = 0
    try:
        with (output / "frame-predictions.jsonl").open("w", encoding="utf-8") as stream:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.predict(frame, imgsz=imgsz, device=device, conf=confidence,
                                       iou=iou, verbose=False)[0]
                predictions = []
                if result.obb is not None:
                    for corners, cls, score in zip(result.obb.xyxyxyxy.cpu().tolist(),
                                                    result.obb.cls.cpu().tolist(),
                                                    result.obb.conf.cpu().tolist()):
                        class_id = int(cls)
                        predictions.append({"class_id": class_id, "class_name": result.names[class_id],
                                            "confidence": float(score), "corners_pixels": corners,
                                            "angle": None, "orientation_semantics": "NOT_APPLICABLE"})
                gt = ground_truth.get(frame_id)
                match_iou = None
                match = None
                if gt is not None and gt.get("visible") is True:
                    counts["visible_gt_frames"] += 1
                    expected = np.asarray(gt["corners"], dtype=np.float32) * [width, height]
                    expected = expected.astype(np.float32)
                    for prediction in predictions:
                        predicted = np.asarray(prediction["corners_pixels"], dtype=np.float32)
                        intersection = float(cv2.intersectConvexConvex(expected, predicted)[0])
                        union = abs(cv2.contourArea(expected)) + abs(cv2.contourArea(predicted)) - intersection
                        overlap = intersection / union if union > 0 else 0.0
                        if match_iou is None or overlap > match_iou:
                            match_iou, match = overlap, prediction
                    if match_iou is not None and match_iou >= 0.5:
                        counts["localized_frames"] += 1
                        if match["class_name"].endswith("_" + gt["face"]):
                            counts["correct_face_frames"] += 1
                elif gt is not None and gt.get("face") is None:
                    counts["blank_gt_frames"] += 1
                    counts["blank_frames_with_detection"] += bool(predictions)
                elif gt is not None and gt.get("visible") is False:
                    counts["partial_visibility_frames"] += 1
                speeds = {key: float(value) for key, value in result.speed.items()}
                timing.append(sum(speeds.values()))
                stream.write(json.dumps({"frame_id": frame_id, "evidence_scope": "synthetic_only",
                                         "predictions": predictions, "ultralytics_speed_ms": speeds,
                                         "gt": gt, "best_localization_iou": match_iou},
                                        allow_nan=False) + "\n")
                plotted = result.plot()
                cv2.putText(plotted, "SYNTHETIC ONLY | Angle N/A", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(plotted)
                if frame_id % max(1, int(fps)) == 0 and len(frames) < 8:
                    frames.append(plotted.copy())
                frame_id += 1
    finally:
        writer.release()
        cap.release()
    if not frame_id:
        raise ValueError("video contains no decodable frames")
    if set(ground_truth) != set(range(frame_id)):
        raise ValueError("video GT does not exactly cover decoded frames")
    _contact_sheet(frames, output / "video-inference-contact-sheet.jpg")
    visible, localized = counts["visible_gt_frames"], counts["localized_frames"]
    summary = {"evidence_scope": "synthetic_only", "video": str(target), "source_video": str(video),
               "frames": frame_id, "fps": fps, "gt_available": bool(ground_truth), "counts": counts,
               "orientation_semantics": "NOT_APPLICABLE", "matching_iou": 0.5,
               "interval_scope": "Wilson intervals describe correlated consecutive synthetic frames; "
                                 "they are not independent physical-event confidence intervals",
               "single_frame_detection_coverage": _wilson(localized, visible),
               "single_frame_miss_rate": _wilson(visible - localized, visible),
               "single_frame_face_accuracy_among_localized": _wilson(counts["correct_face_frames"], localized),
               "single_frame_face_and_localization_accuracy": _wilson(counts["correct_face_frames"], visible),
               "blank_frame_false_detection_rate": _wilson(counts["blank_frames_with_detection"],
                                                           counts["blank_gt_frames"]),
               "ultralytics_single_frame_p95_ms": float(np.percentile(timing, 95)),
               "timing_scope": "Ultralytics single-frame preprocessing/inference/postprocessing; excludes "
                               "video decoding, rendering, writing, event decision and end-to-end latency"}
    write_json(output / "video-summary.json", summary)
    return summary
