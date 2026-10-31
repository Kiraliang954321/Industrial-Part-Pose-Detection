"""Small, bounded per-view endpoint residual calibration; no runtime truth access."""
import json
from pathlib import Path
import numpy as np
from yolo_conveyor.orientation.geometry import prediction
from yolo_conveyor.orientation.common import sha256

WEIGHT_SHA = 'e78fc24c4dd5b758aa0646e8da7ec8b7ba251f382ac327c17dcf85725903cb03'
MAX_CORRECTION_PX = 6.0


def orientation_group(points):
    return 'head_up' if points[0][1] < points[1][1] else 'head_down'


def fit(records):
    groups = {}
    for row in records:
        if not row['objects'] or len(row['raw']) != 1:
            continue
        pred = row['raw'][0]
        if pred['confidence'] < .6 or pred['heading_deg'] is None:
            continue
        key = row['view'] + '/' + orientation_group(pred['keypoints_xy'])
        groups.setdefault(key, []).append(row)
    parameters = {}
    for key, rows in groups.items():
        count = len({row['episode_id'] for row in rows})
        if count < 10:
            raise ValueError('need ten independent fitting episodes: ' + key)
        residuals = np.array([np.subtract(row['objects'][0]['keypoints_xy'], row['raw'][0]['keypoints_xy']) for row in rows])
        offset = np.median(residuals, axis=0)
        if not np.isfinite(offset).all() or np.max(np.linalg.norm(offset, axis=1)) > MAX_CORRECTION_PX:
            raise ValueError('residual outside bounded calibration model: ' + key)
        parameters[key] = dict(offset_px=offset.tolist(),episodes=count,images=len(rows),
                               residual_median_norm_px=float(np.median(np.linalg.norm(residuals-offset,axis=2))))
    return parameters


class Calibrator:
    def __init__(self, parameters):
        self.parameters = parameters
        for value in parameters.values():
            offset = np.asarray(value['offset_px'])
            if offset.shape != (2,2) or not np.isfinite(offset).all() or np.max(np.linalg.norm(offset,axis=1)) > MAX_CORRECTION_PX:
                raise ValueError('invalid correction')

    @classmethod
    def load(cls, path, weights):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data['scope'] != 'H2_DEVELOPMENT_RESIDUAL_CALIBRATION' or data['weights_sha256'] != sha256(Path(weights)):
            raise ValueError('calibrator/checkpoint mismatch')
        return cls(data['parameters'])

    def apply(self, view, predictions):
        output = []
        for item in predictions:
            key = view + '/' + orientation_group(item['keypoints_xy'])
            if key not in self.parameters:
                raise ValueError('uncalibrated_view_orientation')
            points = np.asarray(item['keypoints_xy']) + self.parameters[key]['offset_px']
            reason = 'corrected_keypoint_outside_image' if np.any(points < 0) or np.any(points >= 640) else item['rejection_reason']
            output.append(prediction(item['bbox_xyxy'],points.tolist(),item['confidence'],item['keypoint_confidences'],
                                     min_point_conf=.5,reason=reason))
        return output
