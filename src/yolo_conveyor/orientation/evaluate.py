"""Evaluate Train/Val only; independent Test requires a separate frozen protocol."""

import argparse
import json
import os
from pathlib import Path
import time

from yolo_conveyor.datasets.screw_orientation import audit_dataset, load_dataset
from .common import TASK, SCOPE, dataset_file, environment, new_directory, sha256, source_hashes, utc_now, write_json
from .evaluation import score_scene, summarize


def make_predictor(args):
    if args.method == "cv":
        from .traditional import TraditionalPredictor
        return TraditionalPredictor(threshold=args.cv_threshold, width_ratio=args.width_ratio)
    if not args.weights:
        raise ValueError("--weights is required for Pose")
    from .pose_adapter import PosePredictor
    return PosePredictor(args.weights, device=args.device, imgsz=args.imgsz,
                         confidence=args.confidence, point_confidence=args.point_confidence)


def draw_predictions(image, predictions, title):
    import cv2
    import numpy as np
    output = image.copy()
    for pred in predictions:
        color = (30, 210, 40) if pred["heading_deg"] is not None else (20, 140, 240)
        box = np.round(pred["bbox_xyxy"]).astype(int)
        cv2.rectangle(output, tuple(box[:2]), tuple(box[2:]), color, 1)
        head, tip = [tuple(np.round(p).astype(int)) for p in pred["keypoints_xy"]]
        cv2.circle(output, head, 4, (40, 40, 240), -1)
        cv2.circle(output, tip, 4, (240, 140, 40), -1)
        if pred["heading_deg"] is not None:
            cv2.arrowedLine(output, tip, head, color, 2, tipLength=0.12)
        text = pred["rejection_reason"] or f"{pred['heading_deg']:.1f} deg"
        cv2.putText(output, text, (max(0, box[0]), max(40, box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(output, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 220), 1)
    return output


def evaluate(args) -> dict:
    import cv2
    import numpy as np
    if args.split not in ("train", "val"):
        raise ValueError("development evaluation only supports Train/Val")
    root = Path(args.dataset).resolve()
    audit = audit_dataset(root)
    if audit["errors"]:
        raise ValueError(f"dataset audit failed: {audit['errors'][:5]}")
    meta, rows = load_dataset(root)
    rows = [row for row in rows if row["split"] == args.split]
    sources = [root] + ([args.weights] if args.weights else [])
    out = new_directory(args.output, sources)
    (out / "yolo-config").mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(out / "yolo-config")
    cv2.setNumThreads(4)
    try:
        predictor = make_predictor(args)
        predictor.predict(np.zeros((meta["image_size"], meta["image_size"], 3), dtype=np.uint8))
        if hasattr(predictor, "synchronize"):
            predictor.synchronize()
        (out / "previews").mkdir()
        scored, preview_paths = [], []
        with (out / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for index, row in enumerate(rows):
                path = dataset_file(root, row["image"])
                started = time.perf_counter()
                image = cv2.imread(str(path))
                predictions = predictor.predict(image)
                if hasattr(predictor, "synchronize"):
                    predictor.synchronize()
                latency = (time.perf_counter() - started) * 1000
                # Only the scorer receives GT; the predictor receives the decoded image alone.
                record = score_scene(row, predictions, latency)
                scored.append(record)
                stream.write(json.dumps({"scene_id": row["scene_id"], "image_sha256": row["image_sha256"],
                                         "predictions": predictions, "score": record}, allow_nan=False) + "\n")
                if index < 24 or (not record["correct_image_10deg"] and len(preview_paths) < 48):
                    path = out / "previews" / f"{row['scene_id']}.jpg"
                    preview = draw_predictions(image, predictions, f"{args.method} | {args.split} | {row['condition']} | SIM DEV")
                    if not cv2.imwrite(str(path), preview):
                        raise RuntimeError("preview image could not be written")
                    preview_paths.append(path)
        thumbnails = [cv2.resize(cv2.imread(str(p)), (320, 320)) for p in preview_paths[:12]]
        while len(thumbnails) % 3:
            thumbnails.append(np.zeros((320, 320, 3), dtype=np.uint8))
        if thumbnails:
            sheet = np.concatenate([np.concatenate(thumbnails[i:i + 3], axis=1) for i in range(0, len(thumbnails), 3)], axis=0)
            if not cv2.imwrite(str(out / "contact-sheet.jpg"), sheet):
                raise RuntimeError("contact sheet write failed")
        report = {"task": TASK, "scope": SCOPE, "formal_training_gate": "closed", "status": "completed",
                  "split": args.split, "method": args.method, "parameters": vars(args),
                  "dataset_manifest_sha256": sha256(root / "dataset.json"),
                  "checkpoint_sha256": sha256(Path(args.weights)) if args.weights else None,
                  "environment": environment(), "source_hashes": source_hashes(), "created_at": utc_now(),
                  "overall": summarize(scored), "by_condition": {
                      condition: summarize([r for r in scored if r["condition"] == condition])
                      for condition in sorted({r["condition"] for r in scored})},
                  "semantics": {"unit": "one independent synthetic scene/image; not a conveyor event",
                                "matching": "one GT maximum, highest box IoU >= 0.5; unmatched detections count as extra",
                                "angle": "directed tail-to-head, 360 degrees; no absolute dot product",
                                "correct_images_10deg": "positive: exactly one matched, usable prediction <=10deg; negative: no detections",
                                "10deg": "development selection diagnostic, not an industrial acceptance tolerance",
                                "latency": "warm model; read/decode through synchronized prediction; scoring/drawing/model load excluded",
                                "claims": "Train/Val development diagnostics only; no test or real-world generalization"}}
        write_json(out / "report.json", report)
        return report
    except BaseException as exc:
        write_json(out / "failure.json", {"error": str(exc), "updated_at": utc_now()})
        raise


def add_prediction_arguments(parser):
    parser.add_argument("--method", choices=("cv", "pose"), required=True)
    parser.add_argument("--weights")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--point-confidence", type=float, default=0.25)
    parser.add_argument("--cv-threshold", type=int, default=35)
    parser.add_argument("--width-ratio", type=float, default=1.2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    add_prediction_arguments(parser)
    report = evaluate(parser.parse_args())
    print(json.dumps(report["overall"], indent=2))
    return 0
