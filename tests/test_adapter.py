import builtins
import sys
from types import SimpleNamespace

from yolo_conveyor.config import DEFAULT_MODEL
from yolo_conveyor.ultralytics_adapter import UltralyticsAdapter


def test_lazy_import_and_default(monkeypatch):
    original_import = builtins.__import__

    def reject_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ultralytics)
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    assert DEFAULT_MODEL == "yolo26n-obb.pt"
    adapter = UltralyticsAdapter()
    assert adapter.model_name == DEFAULT_MODEL
    assert adapter._model is None
    try:
        adapter.predict("dummy.jpg")
    except RuntimeError as exc:
        assert "Ultralytics" in str(exc)
    else:
        assert False, "inference must fail if dependency is absent"


def test_conversion_uses_obb_corners_and_preserves_all_predictions(monkeypatch):
    class Tensor:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    class FakeYOLO:
        def __init__(self, name):
            assert name == DEFAULT_MODEL

        def predict(self, **kwargs):
            assert kwargs["source"] == "dummy.jpg"
            return [SimpleNamespace(obb=SimpleNamespace(
                xyxyxyxy=Tensor([[[0, 0], [2, 0], [2, 1], [0, 1]]]),
                cls=Tensor([3]), conf=Tensor([0.75])))]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    predictions = UltralyticsAdapter().predict("dummy.jpg")
    assert len(predictions) == 1
    assert predictions[0].class_index == 3  # Not mapped to front/back.
    assert predictions[0].center == (1, 0.5)
    assert predictions[0].confidence == 0.75
