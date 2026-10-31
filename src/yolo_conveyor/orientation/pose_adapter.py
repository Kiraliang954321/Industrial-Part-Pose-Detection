"""Local-checkpoint two-keypoint adapter. No simulator truth or automatic downloads."""

from pathlib import Path

from .common import KEYPOINTS, unlinked
from .geometry import prediction


def verify_model(model) -> None:
    names = {int(k): v for k, v in model.names.items()}
    shape = getattr(model.model, "kpt_shape", None)
    if shape is None:
        shape = getattr(model.model.model[-1], "kpt_shape", None)
    kpt_names = getattr(model.model, "kpt_names", None)
    if (model.task != "pose" or names != {0: "screw"} or list(shape or []) != [2, 3] or
            not kpt_names or {int(k): v for k, v in kpt_names.items()} != {0: KEYPOINTS}):
        raise ValueError("checkpoint must declare screw Pose with ordered head/tip keypoints")


class PosePredictor:
    def __init__(self, weights: str | Path, *, device="0", imgsz=640, confidence=0.25,
                 point_confidence=0.25, min_length=8.0):
        import torch
        from ultralytics import YOLO
        weights = unlinked(weights)
        if not weights.is_file() or weights.suffix.lower() != ".pt":
            raise ValueError("an existing local .pt checkpoint is required")
        if not 0 < confidence < 1 or not 0 <= point_confidence <= 1:
            raise ValueError("invalid prediction thresholds")
        self.model = YOLO(str(weights))
        verify_model(self.model)
        self.device, self.imgsz = device, imgsz
        self.confidence, self.point_confidence, self.min_length = confidence, point_confidence, min_length
        self.torch = torch

    def synchronize(self):
        if self.device != "cpu" and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def predict(self, image) -> list[dict]:
        result = self.model.predict(image, imgsz=self.imgsz, device=self.device,
                                    conf=self.confidence, verbose=False)[0]
        if result.keypoints is None or result.boxes is None:
            return []
        coords = result.keypoints.data.cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        if not len(coords) == len(boxes) == len(scores):
            raise ValueError("inconsistent model output lengths")
        output = []
        for box, points, score in zip(boxes, coords, scores):
            if len(points) != 2 or any(len(p) != 3 for p in points):
                raise ValueError("expected two x,y,confidence keypoints")
            xy = [p[:2] for p in points]
            outside = any(not (0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]) for p in xy)
            output.append(prediction(box, xy, score, [p[2] for p in points],
                                     min_point_conf=self.point_confidence, min_length=self.min_length,
                                     reason="keypoint_outside_image" if outside else None))
        return output
