"""Explicit image/scene-level development metrics, including misses and rejections."""

import math

import numpy as np

from .geometry import bbox_iou, directed_error


def rate(x: int, n: int) -> dict:
    if not 0 <= x <= n:
        raise ValueError("invalid proportion counts")
    if n == 0:
        return {"x": x, "n": n, "estimate": None, "wilson95": None}
    z = 1.959963984540054
    p = x / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"x": x, "n": n, "estimate": p, "wilson95": [max(0, center - half), min(1, center + half)]}


def distribution(values: list[float]) -> dict:
    return {"n": len(values), "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None}


def score_scene(row: dict, predictions: list[dict], latency_ms: float, iou_threshold=0.5) -> dict:
    objects = row["objects"]
    record = {"scene_id": row["scene_id"], "condition": row["condition"], "positive": bool(objects),
              "prediction_count": len(predictions), "matched": False, "rejected": False,
              "angle_error_deg": None, "endpoint_mean_px": None, "endpoint_normalized": None,
              "latency_ms": latency_ms, "correct_image_10deg": not objects and not predictions}
    if not objects or not predictions:
        return record
    obj = objects[0]
    index = max(range(len(predictions)), key=lambda i: bbox_iou(obj["bbox_xyxy"], predictions[i]["bbox_xyxy"]))
    chosen = predictions[index]
    overlap = bbox_iou(obj["bbox_xyxy"], chosen["bbox_xyxy"])
    record["matched_iou"] = overlap
    if overlap < iou_threshold:
        return record
    record["matched"] = True
    record["rejected"] = chosen["heading_deg"] is None
    if record["rejected"]:
        return record
    angle = directed_error(chosen["heading_deg"], obj["heading_deg"])
    error = np.linalg.norm(np.asarray(chosen["keypoints_xy"]) - np.asarray(obj["keypoints_xy"]), axis=1)
    axis_length = float(np.linalg.norm(np.subtract(*obj["keypoints_xy"])))
    record.update(angle_error_deg=angle, endpoint_mean_px=float(error.mean()),
                  endpoint_normalized=float(error.mean() / axis_length),
                  correct_image_10deg=len(predictions) == 1 and angle <= 10)
    return record


def summarize(records: list[dict]) -> dict:
    positive = [r for r in records if r["positive"]]
    negative = [r for r in records if not r["positive"]]
    matched = [r for r in positive if r["matched"]]
    usable = [r for r in matched if r["angle_error_deg"] is not None]
    total_predictions = sum(r["prediction_count"] for r in records)
    angles = [r["angle_error_deg"] for r in usable]
    return {
        "images": len(records), "gt_objects": len(positive), "negative_images": len(negative),
        "predictions": total_predictions, "matched": len(matched), "missed": len(positive) - len(matched),
        "extra_or_unmatched_predictions": total_predictions - len(matched),
        "direction_rejected_matches": sum(r["rejected"] for r in matched),
        "detection_recall_iou50": rate(len(matched), len(positive)),
        "orientation_coverage_all_gt": rate(len(usable), len(positive)),
        "halfplane_errors_conditional": rate(sum(a > 90 for a in angles), len(usable)),
        "within_angle_all_gt": {str(t): rate(sum(a <= t for a in angles), len(positive)) for t in (5, 10, 15, 30)},
        "negative_false_positive_images": rate(sum(r["prediction_count"] > 0 for r in negative), len(negative)),
        "correct_images_10deg": rate(sum(r["correct_image_10deg"] for r in records), len(records)),
        "angle_error_deg_conditional": distribution(angles),
        "endpoint_mean_px_conditional": distribution([r["endpoint_mean_px"] for r in usable]),
        "endpoint_normalized_conditional": distribution([r["endpoint_normalized"] for r in usable]),
        "latency_ms_read_preprocess_predict_postprocess": distribution([r["latency_ms"] for r in records]),
    }
