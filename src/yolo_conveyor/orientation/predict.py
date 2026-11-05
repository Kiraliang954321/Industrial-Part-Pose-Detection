"""Prediction-only replay from RGB images; does not open annotation metadata."""

import argparse
import json
import os
from pathlib import Path

from .common import TASK, SCOPE, new_directory, sha256, unlinked, write_json
from .evaluate import add_prediction_arguments, draw_predictions, make_predictor


def main() -> int:
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    add_prediction_arguments(parser)
    args = parser.parse_args()
    source = unlinked(args.input)
    images = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not images:
        raise ValueError("no input images")
    images = [unlinked(p) for p in images]
    out = new_directory(args.output, [source] + ([args.weights] if args.weights else []))
    (out / "yolo-config").mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(out / "yolo-config")
    predictor = make_predictor(args)
    (out / "frames").mkdir()
    writer = None
    counts = 0
    try:
        with (out / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for index, path in enumerate(images):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"cannot decode {path}")
                predictions = predictor.predict(image)
                stream.write(json.dumps({"source": str(path), "sha256": sha256(path), "predictions": predictions}, allow_nan=False) + "\n")
                annotated = draw_predictions(image, predictions, f"{args.method} | SIM DEVELOPMENT | image {index + 1}")
                if not cv2.imwrite(str(out / "frames" / f"{index:05d}.jpg"), annotated):
                    raise RuntimeError("frame write failed")
                frame = cv2.resize(annotated, (640, 640))
                if writer is None:
                    writer = cv2.VideoWriter(str(out / "replay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4, (640, 640))
                    if not writer.isOpened():
                        raise RuntimeError("video writer unavailable")
                writer.write(frame)
                counts += 1
    except BaseException as exc:
        write_json(out / "failure.json", {"error": str(exc), "frames_written": counts})
        raise
    finally:
        if writer:
            writer.release()
    capture = cv2.VideoCapture(str(out / "replay.mp4"))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    if decoded != counts:
        raise RuntimeError(f"video decode verification failed: {decoded}/{counts}")
    write_json(out / "replay.json", {"task": TASK, "scope": SCOPE, "frames": counts, "decoded_frames": decoded, "fps": 4,
                                    "checkpoint_sha256": sha256(Path(args.weights)) if args.weights else None,
                                    "parameters": vars(args), "claims": "Still-image prediction replay, not a physical motion recording"})
    return 0
