"""Bounded, auditable development training of the two-keypoint screw model."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time

from yolo_conveyor.datasets.screw_orientation import audit_dataset, load_dataset
from .common import TASK, SCOPE, environment, new_directory, sha256, source_hashes, unlinked, utc_now, write_json

OFFICIAL_SHA256 = "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9"
OFFICIAL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt"


def train(args) -> dict:
    if (not 1 <= args.epochs <= 30 or not 1 <= args.batch <= 32 or
            not 128 <= args.imgsz <= 1024 or args.imgsz % 32 or
            not 1 <= args.max_minutes <= 60 or not 0 <= args.seed < 2**32 or
            args.device not in ("0", "cpu")):
        raise ValueError("invalid training settings or approved development budget exceeded")
    root, initial = unlinked(args.dataset), unlinked(args.initial_weights)
    if not initial.is_file() or initial.suffix != ".pt" or sha256(initial) != OFFICIAL_SHA256:
        raise ValueError("initial checkpoint must match the recorded official YOLO26n-Pose SHA-256")
    audit = audit_dataset(root)
    if audit["errors"]:
        raise ValueError(f"dataset audit failed: {audit['errors'][:5]}")
    meta, _ = load_dataset(root)
    out = new_directory(args.output, [root, initial])
    (out / "yolo-config").mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(out / "yolo-config")
    started = time.perf_counter()
    deadline_reached = False
    latest_progress = 0.0
    try:
        from ultralytics import YOLO, settings
        import torch
        settings.update({k: False for k in ("sync", "wandb", "mlflow", "comet", "clearml", "neptune", "hub", "tensorboard") if k in settings})
        from .pose_adapter import verify_model
        torch.set_num_threads(8)
        if args.device == "0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if args.precision == "bf16" and (args.device == "cpu" or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("BF16 requires a supported CUDA device; choose --precision fp32")
        if args.device == "0":
            torch.cuda.reset_peak_memory_stats()
        snapshots = out / "inputs"
        snapshots.mkdir()
        for name in ("dataset.json", "records.jsonl", "split-plan.json", "asset.json", "scene.xml", "data.yaml"):
            shutil.copyfile(root / name, snapshots / name)
        weight_snapshot = snapshots / "initial-yolo26n-pose.pt"
        shutil.copyfile(initial, weight_snapshot)
        if sha256(weight_snapshot) != OFFICIAL_SHA256:
            raise ValueError("checkpoint snapshot hash mismatch")
        write_json(out / "preflight-audit.json", audit)
        manifest = {"task": TASK, "scope": SCOPE, "formal_training_gate": "closed", "created_at": utc_now(),
                    "arguments": vars(args), "dataset_manifest_sha256": sha256(root / "dataset.json"),
                    "initial_weight": {"url": OFFICIAL_URL, "sha256": OFFICIAL_SHA256},
                    "environment": environment(), "source_hashes": source_hashes(),
                    "gpu": torch.cuda.get_device_name(0) if args.device == "0" else None,
                    "checkpoint_selection": "Ultralytics best fitness on development Val; direct angle selection is a separate report",
                    "limitations": meta["limitations"]}
        write_json(out / "run-manifest.json", manifest)
        project_root = Path(__file__).resolve().parents[3]
        for relative, expected_hash in manifest["source_hashes"].items():
            source = project_root / relative
            snapshot = snapshots / "source" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, snapshot)
            if sha256(snapshot) != expected_hash:
                raise ValueError("source changed while creating the run snapshot")

        def progress(trainer, phase="training"):
            # Ultralytics increments epoch once for its extra final-best validation callback.
            reported_epoch = int(trainer.epoch) + 1
            write_json(out / "progress.json", {
                "status": "final_validation" if reported_epoch > args.epochs else phase,
                "epoch": min(reported_epoch, args.epochs), "requested_epochs": args.epochs,
                "seconds": time.perf_counter() - started, "time_budget_reached": deadline_reached,
                "updated_at": utc_now()}, replace=True)

        def batch_end(trainer):
            nonlocal deadline_reached, latest_progress
            now = time.perf_counter()
            if now - started >= args.max_minutes * 60:
                deadline_reached = True
                trainer.stop = True
            if (out / "STOP_REQUESTED").exists():
                trainer.stop = True
            if now - latest_progress > 15:
                progress(trainer)
                latest_progress = now

        model = YOLO(str(weight_snapshot))
        if model.task != "pose":
            raise ValueError("initial model must be a Pose model")
        model.add_callback("on_train_batch_end", batch_end)
        model.add_callback("on_fit_epoch_end", lambda trainer: progress(trainer, "epoch_validated"))
        model.train(data=str(root / "data.yaml"), epochs=args.epochs, batch=args.batch,
                    imgsz=args.imgsz, device=args.device, workers=0, seed=args.seed,
                    deterministic=True, project=str(out), name="fit", exist_ok=False,
                    optimizer="AdamW", lr0=0.001, lrf=0.01, patience=0,
                    amp="bf16" if args.precision == "bf16" else False,
                    cache=False, plots=False, save=True, save_period=-1,
                    mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.05, scale=0.15,
                    fliplr=0.5, flipud=0.5, hsv_h=0.015, hsv_s=0.2, hsv_v=0.2,
                    close_mosaic=0, val=True, verbose=False)
        fit = unlinked(model.trainer.save_dir)
        if not fit.is_relative_to(out):
            raise ValueError("trainer outputs escaped the run directory")
        checkpoints = {}
        for name in ("best", "last"):
            path = fit / "weights" / f"{name}.pt"
            verify_model(YOLO(str(path)))
            checkpoints[name] = {"path": str(path), "sha256": sha256(path)}
        with (fit / "results.csv").open(encoding="utf-8", newline="") as stream:
            epochs = list(csv.DictReader(stream))
        final_audit = audit_dataset(root)
        if final_audit["errors"]:
            raise ValueError("dataset changed or became invalid during training")
        if sha256(root / "dataset.json") != manifest["dataset_manifest_sha256"]:
            raise ValueError("dataset manifest changed during training")
        summary = {"task": TASK, "scope": SCOPE, "formal_training_gate": "closed", "status": "completed",
                   "actual_logged_epochs": len(epochs), "requested_epochs": args.epochs,
                   "actual_batch_size": int(model.trainer.batch_size),
                   "stop_requested": (out / "STOP_REQUESTED").exists(),
                   "time_budget_reached": deadline_reached, "seconds": time.perf_counter() - started,
                   "checkpoints": checkpoints,
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "0" else None,
                   "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "0" else None,
                   "last_csv_metrics": epochs[-1] if epochs else {}, "completed_at": utc_now(),
                   "claims": "Completed development optimization only; direct task evaluation remains separate"}
        write_json(out / "training-summary.json", summary)
        write_json(out / "progress.json", {"status": "completed", "epochs": len(epochs), "updated_at": utc_now()}, replace=True)
        return summary
    except BaseException as exc:
        write_json(out / "progress.json", {"status": "failed", "error": str(exc), "updated_at": utc_now()}, replace=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-weights", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", choices=("0", "cpu"), default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-minutes", type=int, default=60)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    result = train(parser.parse_args())
    print(json.dumps(result, indent=2))
    return 0
