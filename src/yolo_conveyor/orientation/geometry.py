"""Directed image-coordinate geometry. A head/tip swap is a 180 degree error."""

import math


def heading(keypoints, *, min_length: float = 1.0) -> float | None:
    if not math.isfinite(min_length) or min_length <= 0:
        raise ValueError("min_length must be finite and positive")
    if len(keypoints) != 2 or any(len(p) != 2 for p in keypoints):
        raise ValueError("expected head and tip x,y coordinates")
    if not all(math.isfinite(v) for p in keypoints for v in p):
        return None
    dx, dy = keypoints[0][0] - keypoints[1][0], keypoints[0][1] - keypoints[1][1]
    if math.hypot(dx, dy) < min_length:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0


def directed_error(predicted: float, expected: float) -> float:
    if not math.isfinite(predicted) or not math.isfinite(expected):
        raise ValueError("angles must be finite")
    return abs((predicted - expected + 180.0) % 360.0 - 180.0)


def bbox_iou(a, b) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(
        0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def prediction(bbox, points, confidence, point_confidences, *, min_point_conf=0.25,
               min_length=8.0, reason=None) -> dict:
    numeric = [*bbox, confidence, *point_confidences, *(v for p in points for v in p)]
    if (len(bbox) != 4 or len(points) != 2 or len(point_confidences) != 2 or
            any(len(p) != 2 for p in points) or not all(math.isfinite(v) for v in numeric)):
        raise ValueError("malformed prediction")
    if not 0 <= min_point_conf <= 1 or not all(0 <= v <= 1 for v in [confidence, *point_confidences]):
        raise ValueError("invalid confidence")
    angle = heading(points, min_length=min_length)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        reason = "invalid_box"
    elif min(point_confidences) < min_point_conf:
        reason = "low_keypoint_confidence"
    elif angle is None:
        reason = "short_or_degenerate_projection"
    return {"bbox_xyxy": [float(v) for v in bbox], "keypoints_xy": [[float(v) for v in p] for p in points],
            "confidence": float(confidence), "keypoint_confidences": [float(v) for v in point_confidences],
            "heading_deg": None if reason else angle, "rejection_reason": reason}
