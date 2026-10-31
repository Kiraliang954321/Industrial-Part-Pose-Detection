"""Convert only camera pixels/predictions and fixed calibration into event evidence."""

import math
import numpy as np

from .events import FrameEvidence
from yolo_conveyor.simulation.grasp_geometry import from_prediction


def frame_evidence(frame_id, time_s, image, predictions, camera):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or not image.size or image.dtype != np.uint8:
        return FrameEvidence(frame_id, time_s, None, None, "invalid_camera_frame")
    if float(np.max(np.std(image,axis=(0,1)))) < 1.0:
        return FrameEvidence(frame_id, time_s, None, None, "flat_camera_frame")
    try:
        observation = from_prediction(predictions, camera)
    except (ValueError, KeyError, TypeError) as exc:
        return FrameEvidence(frame_id, time_s, None, None, str(exc))
    return FrameEvidence(frame_id, time_s, math.degrees(observation.heading_rad) % 360, observation.center_xy)
