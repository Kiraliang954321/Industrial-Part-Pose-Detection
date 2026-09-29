"""Pure baseline configuration constants."""

from dataclasses import dataclass

DEFAULT_MODEL = "yolo26n-obb.pt"
FACE_CLASSES = ("front", "back")  # Only valid for a separately verified/fine-tuned model.


@dataclass(frozen=True)
class InferenceConfig:
    model: str = DEFAULT_MODEL
    confidence: float = 0.25

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must not be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
