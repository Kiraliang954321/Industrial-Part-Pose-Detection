"""Minimal single-image OBB adapter. Importing this module never loads Ultralytics."""

from pathlib import Path

from .config import DEFAULT_MODEL
from .types import Detection


class UltralyticsAdapter:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model_name = model
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Inference requires Ultralytics; install the inference extra") from exc
            self._model = YOLO(self.model_name)
        return self._model

    def predict(self, image: str | Path, confidence: float = 0.25) -> list[Detection]:
        """Return all single-frame OBB detections in original image pixel coordinates."""
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        results = self._load().predict(source=str(image), conf=confidence, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.obb is None:
                continue
            for corners, cls, conf in zip(result.obb.xyxyxyxy.tolist(),
                                          result.obb.cls.tolist(), result.obb.conf.tolist()):
                if len(corners) != 4 or any(len(point) != 2 for point in corners):
                    raise ValueError("unexpected Ultralytics OBB corner shape")
                points = tuple((float(x), float(y)) for x, y in corners)
                detections.append(Detection(int(cls), float(conf), points))  # type: ignore[arg-type]
        return detections
