"""Pure 2.5D perception-to-plan interface; accepts no simulator object state."""

from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class GraspObservation:
    center_xy: tuple[float, float]
    heading_rad: float
    source: str

    def record(self):
        return asdict(self)


def ray_to_plane(pixel, camera, z):
    k = np.asarray(camera["K"], dtype=float)
    t = np.asarray(camera["T_world_camera_cv"], dtype=float)
    if k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(t).all():
        raise ValueError("invalid calibration")
    direction = t[:3, :3] @ np.linalg.solve(k, [*pixel, 1.0])
    if abs(direction[2]) < 1e-8:
        raise ValueError("ray parallel to support plane")
    scale = (z - t[2, 3]) / direction[2]
    if scale <= 0:
        raise ValueError("plane is behind the camera")
    return t[:3, 3] + scale * direction


def from_prediction(predictions, camera):
    if len(predictions) != 1:
        raise ValueError("expected exactly one screw prediction")
    predicted = predictions[0]
    if predicted["heading_deg"] is None:
        raise ValueError(f"direction rejected: {predicted['rejection_reason']}")
    # Fixed proxy/support priors, not per-scene ground-truth depth.
    head = ray_to_plane(predicted["keypoints_xy"][0], camera, 0.006)
    tip = ray_to_plane(predicted["keypoints_xy"][1], camera, 0.003)
    length = np.linalg.norm(head[:2] - tip[:2])
    center = (head[:2] + tip[:2]) / 2
    if not 0.033 <= length <= 0.053 or np.any(np.abs(center) > 0.065):
        raise ValueError("prediction outside declared proxy length/workspace")
    return GraspObservation(tuple(map(float, center)), math.atan2(head[1]-tip[1], head[0]-tip[0]), "vision_prediction")


def from_ground_truth(anchors):
    head, tip = np.asarray(anchors)
    return GraspObservation(tuple(map(float, (head[:2]+tip[:2])/2)),
                            math.atan2(head[1]-tip[1], head[0]-tip[0]), "gt_xy_heading_reference")


def downward_rotation(heading):
    x = np.array([math.cos(heading), math.sin(heading), 0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])
