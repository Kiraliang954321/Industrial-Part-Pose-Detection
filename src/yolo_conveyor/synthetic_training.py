"""Train and evaluate synthetic-only cap models while the formal gate stays closed."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import time

from yolo_conveyor.annotation.contract import sha256_file

from .synthetic_bundle import read_json, training_yaml, validate_bundle, write_json
from .synthetic_video import predict_video, validation_contact_sheet

CONFIDENCE = 0.25
NMS_IOU = 0.7


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git() -> dict:
    repository = Path(__file__).resolve().parents[2]
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=repository, check=True,
                                  capture_output=True, encoding="utf-8").stdout.strip()
        return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "dirty_paths": git("status", "--porcelain", "--untracked-files=normal").splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty_paths": None}


def _environment(torch) -> dict:
    versions = {}
    for package in ("ultralytics", "torch", "numpy", "opencv-python", "pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cpu_threads": torch.get_num_threads()}


def _metrics(metrics, names: dict, counts: dict) -> dict:
    box = metrics.box
    indices = [int(index) for index in box.ap_class_index]
    per_class = {}
    for class_id, name in names.items():
        class_id = int(class_id)
        samples = counts["faces"]["front" if class_id == 0 else "back"]
        if class_id in indices:
            index = indices.index(class_id)
            values = [float(value) for value in box.class_result(index)]
        else:
            values = [0.0, 0.0, 0.0, 0.0]
        per_class[name] = dict(zip(("precision", "recall", "map50", "map50_95"), values))
        per_class[name]["ground_truth_instances"] = samples
    return {"evidence_scope": "synthetic_only", "images": counts["images"],
            "precision": float(box.mp), "recall": float(box.mr),
            "map50": float(box.map50), "map50_95": float(box.map), "per_class": per_class,
            "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
            "timing_scope": "Ultralytics single-frame stage timings; no event or end-to-end claim",
            "precision_recall_scope": "Ultralytics AP validation curves (default low confidence); "
                                      "frozen video operating point is reported separately"}


def _training_epochs(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return len(rows)


def _fresh_output(output: Path, bundle: Path) -> Path:
    if os.path.lexists(output):
        raise FileExistsError(f"output must be an entirely new directory: {output}")
    output = output.resolve()
    if output == bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("training output must be separate from the bundle")
    return output


def _evaluate_holdout(args, info: dict, output: Path, YOLO) -> dict:
    """A separate, explicit untuned evaluation of a previously frozen selection."""
    if args.frozen_run is None:
        raise ValueError("--evaluate-holdout requires --frozen-run")
    frozen = args.frozen_run.resolve(strict=True)
    selection = read_json(frozen / "selection.json")
    if (selection.get("dataset_manifest_sha256") != info["dataset_manifest_sha256"] or
            selection.get("dataset_content_sha256") != info["dataset_content_sha256"] or
            selection.get("evidence_scope") != "synthetic_only"):
        raise ValueError("frozen selection differs from this synthetic dataset")
    weight = Path(selection["selected_best_pt"]).resolve(strict=True)
    if not weight.is_relative_to(frozen) or sha256_file(weight) != selection["selected_best_sha256"]:
        raise ValueError("selected model path or checksum differs from frozen selection")
    if selection.get("confidence") != CONFIDENCE or selection.get("nms_iou") != NMS_IOU:
        raise ValueError("unexpected frozen operating point")
    if args.imgsz != selection.get("imgsz"):
        raise ValueError("holdout image size must match the frozen selection")
    yaml = training_yaml(output / "holdout-evaluation-data.yaml", info, holdout=True)
    write_json(output / "holdout-evaluation-config.json", {
        "evidence_scope": "synthetic_only", "purpose": "untuned final validation",
        "frozen_selection_sha256": sha256_file(frozen / "selection.json"),
        "selection": selection, "dataset": info, "git": _git(), "started_at": _utc()})
    model = YOLO(str(weight))
    metrics = model.val(data=str(yaml), split="test", imgsz=args.imgsz, batch=args.batch,
                        device=args.device, workers=0, iou=NMS_IOU,
                        project=str(output), name="final-holdout", exist_ok=False, plots=True)
    summary = {"evidence_scope": "synthetic_only", "training_gate": "closed",
               "purpose": "untuned final validation", "tuning_allowed": False,
               "model_sha256": sha256_file(weight), "final_holdout":
                   _metrics(metrics, info["names"], info["counts"]["final_holdout"])}
    write_json(output / "result-summary.json", summary)
    return summary


def run(args) -> dict:
    """Preflight before creating output, loading weights or touching Test metrics."""
    if min(args.epochs, args.batch, args.imgsz, args.cpu_threads) <= 0 or args.patience < 0:
        raise ValueError("epochs, batch, imgsz, threads must be positive; patience cannot be negative")
    if args.workers != 0:
        raise ValueError("this reproducible Windows route requires --workers 0")
    if len(args.models) != len(set(args.models)):
        raise ValueError("models must not be duplicated")
    info = validate_bundle(args.bundle)
    output = _fresh_output(args.output, Path(info["bundle"]))
    if args.evaluate_holdout and args.frozen_run is None:
        raise ValueError("--evaluate-holdout requires --frozen-run")
    if not args.evaluate_holdout and args.frozen_run is not None:
        raise ValueError("--frozen-run is only valid with --evaluate-holdout")
    output.mkdir(parents=True, exist_ok=False)
    config_dir = output / "ultralytics-config"
    config_dir.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    os.environ["WANDB_MODE"] = "disabled"
    # Platform uploads are independent of sync/logger flags. Limit this process to local artifacts.
    os.environ.pop("ULTRALYTICS_API_KEY", None)
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
    except ImportError as exc:
        raise RuntimeError("synthetic training requires the existing inference extra (Ultralytics)") from exc
    SETTINGS.update({key: False for key in ("sync", "wandb", "comet", "mlflow", "clearml", "dvc",
                                           "raytune", "tensorboard") if key in SETTINGS})
    SETTINGS.update({"api_key": ""})
    torch.set_num_threads(args.cpu_threads)
    if args.evaluate_holdout:
        return _evaluate_holdout(args, info, output, YOLO)
    yaml = training_yaml(output / "training-data.yaml", info)
    config = {"evidence_scope": "synthetic_only", "formal_training_gate": "closed",
              "orientation_semantics": "NOT_APPLICABLE", "dataset": info, "git": _git(),
              "environment": _environment(torch), "started_at": _utc(),
              "models": args.models, "epochs": args.epochs, "batch": args.batch,
              "imgsz": args.imgsz, "device": args.device, "workers": 0, "seed": args.seed,
              "patience": args.patience, "confidence": CONFIDENCE, "nms_iou": NMS_IOU,
              "selection_rule": "highest Val mAP50-95; ties keep CLI model order",
              "test_policy": "evaluate only after all training and selection have been frozen",
              "final_holdout_policy": "not evaluated; separate explicit --evaluate-holdout only"}
    write_json(output / "run-config.json", config)
    results = {}
    started = time.perf_counter()
    for variant in args.models:
        name = f"yolo26{variant}-obb"
        model = YOLO(name + ".pt")
        pretrain_path = Path(getattr(model, "ckpt_path", name + ".pt"))
        pretrain_hash = sha256_file(pretrain_path) if pretrain_path.is_file() else None
        model_started = time.perf_counter()
        model.train(data=str(yaml), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                    device=args.device, workers=0, seed=args.seed, deterministic=True,
                    patience=args.patience, project=str(output), name=name, exist_ok=False,
                    pretrained=True, val=True, plots=True, cache=False)
        train_seconds = time.perf_counter() - model_started
        model_dir = Path(model.trainer.save_dir).resolve()
        if not model_dir.is_relative_to(output):
            raise ValueError("Ultralytics output escaped run directory")
        best, last = model_dir / "weights/best.pt", model_dir / "weights/last.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError(f"training did not produce best/last checkpoints: {name}")
        selected = YOLO(str(best))
        metrics = selected.val(data=str(yaml), split="val", imgsz=args.imgsz, batch=args.batch,
                               device=args.device, workers=0, iou=NMS_IOU,
                               project=str(output), name=name + "-val", exist_ok=False, plots=True)
        contact = validation_contact_sheet(selected, Path(info["dataset"]), model_dir,
                                           imgsz=args.imgsz, device=args.device,
                                           confidence=CONFIDENCE, iou=NMS_IOU)
        results[name] = {"model": name, "pretrained_checkpoint_sha256": pretrain_hash,
                         "best_pt": str(best), "last_pt": str(last),
                         "best_sha256": sha256_file(best), "last_sha256": sha256_file(last),
                         "actual_epochs": _training_epochs(model_dir / "results.csv"),
                         "train_seconds": train_seconds, "val_contact_sheet": str(contact),
                         "validation": _metrics(metrics, info["names"], info["counts"]["val"])}
        write_json(output / "training-progress.json", {"models": results,
                                                       "evidence_scope": "synthetic_only"})
    winner = max(results, key=lambda name: results[name]["validation"]["map50_95"])
    selection = {"evidence_scope": "synthetic_only", "training_gate": "closed",
                 "frozen_at": _utc(), "selection_uses": "Val mAP50-95 only",
                 "selected_model": winner, "selected_best_pt": results[winner]["best_pt"],
                 "selected_best_sha256": results[winner]["best_sha256"],
                 "confidence": CONFIDENCE, "nms_iou": NMS_IOU,
                 "imgsz": args.imgsz,
                 "dataset_manifest_sha256": info["dataset_manifest_sha256"],
                 "dataset_content_sha256": info["dataset_content_sha256"],
                 "validation_scores": {name: value["validation"]["map50_95"]
                                       for name, value in results.items()},
                 "final_holdout_evaluated": False}
    write_json(output / "selection.json", selection)
    # No Test evaluation may occur before the above frozen record is persisted.
    for name, record in results.items():
        model = YOLO(record["best_pt"])
        metrics = model.val(data=str(yaml), split="test", imgsz=args.imgsz, batch=args.batch,
                            device=args.device, workers=0, iou=NMS_IOU,
                            project=str(output), name=name + "-test", exist_ok=False, plots=True)
        record["test"] = _metrics(metrics, info["names"], info["counts"]["test"])
    video_session = next(item for item in info["sessions"].values() if item["split"] == "test")
    selected = YOLO(results[winner]["best_pt"])
    video = predict_video(selected, Path(video_session["video_path"]), output,
                          gt_path=Path(video_session["video_gt_path"]), imgsz=args.imgsz,
                          device=args.device, confidence=CONFIDENCE, iou=NMS_IOU)
    summary = {"evidence_scope": "synthetic_only", "training_gate": "closed",
               "orientation_semantics": "NOT_APPLICABLE", "completed_at": _utc(),
               "total_seconds": time.perf_counter() - started, "models": results,
               "selection": selection, "video": video, "final_holdout_evaluated": False,
               "limitations": ["Synthetic detection and face classification only; no real-camera evidence",
                               "No tracking, event decision, risk rejection or validated event throughput",
                               "Video Wilson intervals are descriptive for correlated frames; "
                               "not independent physical-event intervals",
                               "Circular cap orientation is N/A; OBB is localization only"]}
    write_json(output / "result-summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new independent run directory")
    parser.add_argument("--models", choices=("n", "s"), nargs="+", default=["n", "s"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--evaluate-holdout", action="store_true",
                        help="separate untuned evaluation of a frozen run; never retrains")
    parser.add_argument("--frozen-run", type=Path, help="completed run containing selection.json")
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"synthetic training failed: {exc}\n")
    print(f"Synthetic-only result saved: {args.output.resolve() / 'result-summary.json'}")
    if summary.get("final_holdout_evaluated") is False:
        print("Formal gate remains CLOSED. Final Holdout was not evaluated.")
    return 0
