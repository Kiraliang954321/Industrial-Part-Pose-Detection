"""Isolated Part A photo fine-tuning; never opens the formal experiment gate."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import stat
import subprocess
import time

from yolo_conveyor.annotation.contract import jpeg_dimensions, sha256_file, validate_corners


SCOPE = "real_photo_development"
NAMES = {"0": "cap_a_front", "1": "cap_a_back"}
SPLITS = ("train", "val")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
AUGMENTATION = {"degrees": 180.0, "translate": 0.1, "scale": 0.15,
                "hsv_h": 0.015, "hsv_s": 0.3, "hsv_v": 0.2,
                "mosaic": 0.2, "mixup": 0.0, "copy_paste": 0.0}
LIMITATIONS = [
    "Development photo fine-tuning only; the formal Training Gate remains CLOSED",
    "Validation uses an earlier development photo batch; no independent final performance claim",
    "Assistant-reviewed training labels are not operator-accepted unless human_accepted is true",
    "No Test or Final Holdout is created, evaluated, or used for model selection",
    "No real motion, event decision, risk rejection, latency or validated throughput evidence",
    "Part A orientation is NOT_APPLICABLE; OBB is used for localization and face classification",
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _unlinked(raw: str | Path, *, file: bool = False) -> Path:
    path = Path(raw).absolute()
    for item in (path, *path.parents):
        try:
            metadata = item.lstat()
        except FileNotFoundError:
            continue
        if (stat.S_ISLNK(metadata.st_mode) or
                getattr(metadata, "st_file_attributes", 0) & 0x400):
            raise ValueError(f"links/reparse paths are not allowed: {item}")
        if item != path and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"path ancestor is not a directory: {item}")
    path = path.resolve()
    if file and not path.is_file():
        raise ValueError(f"expected an existing regular file: {path}")
    return path


def _contained(root: Path, raw: str) -> Path:
    if (not isinstance(raw, str) or not raw or "\\" in raw or ":" in raw or
            "\x00" in raw or Path(raw).is_absolute() or PureWindowsPath(raw).drive or
            any(part in ("", ".", "..") for part in raw.split("/"))):
        raise ValueError(f"invalid relative bundle path: {raw!r}")
    path = _unlinked(root / raw, file=True)
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes bundle: {raw}")
    return path


def _digest(path: Path, expected: object) -> str:
    if not isinstance(expected, str) or not SHA256.fullmatch(expected):
        raise ValueError(f"invalid SHA-256 declaration: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return actual


def _label(path: Path, face: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[0].split() if len(lines) == 1 else []
    if len(cells) != 9 or cells[0] != ("0" if face == "front" else "1"):
        raise ValueError(f"expected one nine-token OBB with matching face class: {path}")
    values = [float(value) for value in cells[1:]]
    validate_corners([values[index:index + 2] for index in range(0, 8, 2)])


def _yaml_evidence(path: Path, root: Path, rows: list[dict]) -> None:
    # The source YAML is checked as evidence only; the trainer receives a new snapshot YAML.
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or set(value) - {"path", "train", "val", "names", "nc"} or
            not isinstance(value.get("names"), dict) or
            {str(key): item for key, item in value["names"].items()} != NAMES or
            value.get("nc", 2) != 2):
        raise ValueError("data YAML must describe only the two declared classes and Train/Val")
    raw_base = value.get("path", ".")
    if not isinstance(raw_base, str):
        raise ValueError("data YAML path must be a string")
    base = _unlinked(path.parent / raw_base)
    if not base.is_dir() or not base.is_relative_to(root):
        raise ValueError("data YAML root must stay inside the manifest bundle")
    for split in SPLITS:
        relative = value.get(split)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"data YAML {split} must be a directory path")
        directory = _unlinked(base / relative)
        if not directory.is_relative_to(root) or not directory.is_dir():
            raise ValueError(f"data YAML {split} directory must stay inside the bundle")
        expected = {row["absolute_image_path"] for row in rows if row["split"] == split}
        actual = set()
        for candidate in directory.rglob("*"):
            _unlinked(candidate)
            if candidate.is_file() and candidate.suffix.lower() in (".jpg", ".jpeg"):
                actual.add(str(candidate.resolve()))
        if actual != expected:
            raise ValueError(f"data YAML {split} images do not match manifest rows")


def validate_manifest(raw: str | Path) -> dict:
    """Verify source identity, derived bytes, OBB geometry and batch-level separation."""
    manifest_path = _unlinked(raw, file=True)
    root = manifest_path.parent
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or
            manifest["schema_version"] != 1 or manifest.get("evidence_scope") != SCOPE or
            manifest.get("formal_training_gate") != "closed" or
            manifest.get("formal_dataset_v1") is not False or manifest.get("names") != NAMES or
            manifest.get("split_strategy") != "whole_real_capture_batches"):
        raise ValueError("expected schema-v1 development-only Part A manifest with formal gate CLOSED")
    groups = manifest.get("split_capture_groups")
    if not isinstance(groups, dict) or set(groups) != set(SPLITS):
        raise ValueError("capture groups must define Train and Val only")
    for split, ids in groups.items():
        if (not isinstance(ids, list) or not ids or
                any(not isinstance(item, str) or not item.strip() for item in ids) or
                len(set(ids)) != len(ids)):
            raise ValueError(f"invalid capture groups: {split}")
    if set(groups["train"]) & set(groups["val"]):
        raise ValueError("a real capture group cannot occur in both Train and Val")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest files must be a nonempty list")
    rows, paths, source_identity = [], set(), {}
    families = {split: set() for split in SPLITS}
    originals = {split: set() for split in SPLITS}
    exports = {split: set() for split in SPLITS}
    observed_groups = {split: set() for split in SPLITS}
    counts = {split: {"images": 0, "faces": {"front": 0, "back": 0}} for split in SPLITS}
    authors = {split: {"human_accepted": 0, "not_human_accepted": 0,
                      "label_authors": {}} for split in SPLITS}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each manifest file must be an object")
        split, face = entry.get("split"), entry.get("face")
        family, group = entry.get("source_family_id"), entry.get("capture_group_id")
        if (split not in SPLITS or face not in ("front", "back") or
                not isinstance(family, str) or not family.strip() or group not in groups[split]):
            raise ValueError("invalid row split, face, source family or capture group")
        image = _contained(root, entry.get("image_path"))
        label = _contained(root, entry.get("label_path"))
        parts = list(Path(entry["image_path"]).parts)
        if image.suffix.lower() not in (".jpg", ".jpeg") or "images" not in parts:
            raise ValueError("derived images must be JPEGs below an images directory")
        parts[parts.index("images")] = "labels"
        if Path(*parts).with_suffix(".txt") != Path(entry["label_path"]):
            raise ValueError("image/label paths must follow Ultralytics images-to-labels mapping")
        for path in (image, label):
            identity = str(path).casefold()
            if identity in paths:
                raise ValueError(f"duplicate derived image or label path: {path}")
            paths.add(identity)
        source_raw = entry.get("absolute_source_image_path")
        if not isinstance(source_raw, str) or not Path(source_raw).is_absolute():
            raise ValueError("absolute_source_image_path must identify an archived original")
        source = _unlinked(source_raw, file=True)
        source_hash = _digest(source, entry.get("source_image_sha256"))
        image_hash = _digest(image, entry.get("image_sha256"))
        label_hash = _digest(label, entry.get("label_sha256"))
        dimensions = (entry.get("exported_width"), entry.get("exported_height"))
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("exported dimensions must be positive integers")
        if jpeg_dimensions(image) != dimensions:
            raise ValueError(f"derived JPEG dimensions differ from manifest: {image}")
        jpeg_dimensions(source)
        _label(label, face)
        author, accepted = entry.get("label_author"), entry.get("human_accepted")
        if not isinstance(author, str) or not author.strip() or type(accepted) is not bool:
            raise ValueError("every label requires explicit author and human acceptance provenance")
        if "derivation_path" in entry or "derivation_sha256" in entry:
            _digest(_contained(root, entry.get("derivation_path")), entry.get("derivation_sha256"))
        source_key = str(source).casefold()
        if source_key in source_identity and source_identity[source_key] != source_hash:
            raise ValueError("one source path cannot declare multiple source hashes")
        source_identity[source_key] = source_hash
        families[split].add(family)
        originals[split].add(source_hash)
        exports[split].add(image_hash)
        observed_groups[split].add(group)
        counts[split]["images"] += 1
        counts[split]["faces"][face] += 1
        authors[split]["human_accepted" if accepted else "not_human_accepted"] += 1
        author_counts = authors[split]["label_authors"]
        author_counts[author] = author_counts.get(author, 0) + 1
        rows.append({**entry, "absolute_image_path": str(image), "absolute_label_path": str(label),
                     "absolute_source_image_path": str(source), "source_image_sha256": source_hash,
                     "image_sha256": image_hash, "label_sha256": label_hash})
    for split in SPLITS:
        if not all(counts[split]["faces"].values()) or observed_groups[split] != set(groups[split]):
            raise ValueError(f"{split} must contain both faces and all declared capture groups")
    for identities in (families, originals, exports):
        if identities["train"] & identities["val"]:
            raise ValueError("source family, source hash or derived image hash leaks across Train/Val")
    if manifest.get("split_counts") != counts:
        raise ValueError("split_counts differ from verified manifest rows")
    yaml_path = _contained(root, manifest.get("data_yaml"))
    _yaml_evidence(yaml_path, root, rows)
    content = json.dumps(rows, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    return {"manifest_path": str(manifest_path), "bundle": str(root), "names": NAMES,
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "dataset_content_sha256": hashlib.sha256(content).hexdigest(),
            "source_data_yaml": str(yaml_path), "source_data_yaml_sha256": sha256_file(yaml_path),
            "split_counts": counts, "split_capture_groups": groups,
            "label_provenance": authors, "files": rows}


def _copy_verified(source: Path, target: Path, digest: str) -> None:
    payload = _unlinked(source, file=True).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"input changed before snapshot copy: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)
    _digest(target, digest)


def _snapshot(info: dict, output: Path) -> Path:
    import yaml

    _digest(Path(info["manifest_path"]), info["manifest_sha256"])
    _digest(Path(info["source_data_yaml"]), info["source_data_yaml_sha256"])
    snapshots = []
    for index, row in enumerate(info["files"]):
        _digest(_unlinked(row["absolute_source_image_path"], file=True), row["source_image_sha256"])
        stem = f"{index:05d}_{row['face']}"
        image = output / "data/images" / row["split"] / (stem + ".jpg")
        label = output / "data/labels" / row["split"] / (stem + ".txt")
        _copy_verified(Path(row["absolute_image_path"]), image, row["image_sha256"])
        _copy_verified(Path(row["absolute_label_path"]), label, row["label_sha256"])
        if jpeg_dimensions(image) != (row["exported_width"], row["exported_height"]):
            raise ValueError("copied JPEG does not match declared export dimensions")
        _label(label, row["face"])
        snapshots.append({"image_path": str(image), "label_path": str(label),
                          "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"],
                          "source_family_id": row["source_family_id"], "split": row["split"]})
    path = output / "training-data.yaml"
    path.write_text(yaml.safe_dump({"path": str(output / "data"), "train": "images/train",
                                   "val": "images/val", "names": {int(k): v for k, v in NAMES.items()}},
                                  sort_keys=False, allow_unicode=True), encoding="utf-8")
    info["training_snapshot"] = snapshots
    info["training_data_yaml_sha256"] = sha256_file(path)
    return path


def _git() -> dict:
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[2],
                                  check=True, capture_output=True, encoding="utf-8").stdout.strip()
        return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "dirty_paths": git("status", "--porcelain", "--untracked-files=normal").splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty_paths": None}


def _environment(torch) -> dict:
    versions = {}
    for package in ("ultralytics", "torch", "numpy", "opencv-python", "pillow", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cpu_threads": torch.get_num_threads()}


def _metrics(metrics, counts: dict) -> dict:
    box, per_class = metrics.box, {}
    indices = [int(index) for index in box.ap_class_index]
    for class_id, name in NAMES.items():
        class_id = int(class_id)
        values = ([float(value) for value in box.class_result(indices.index(class_id))]
                  if class_id in indices else [0.0, 0.0, 0.0, 0.0])
        per_class[name] = dict(zip(("precision", "recall", "map50", "map50_95"), values))
        per_class[name]["ground_truth_instances"] = counts["faces"]["front" if class_id == 0 else "back"]
    result = {"evidence_scope": SCOPE, "formal_training_gate": "closed", "images": counts["images"],
              "precision": float(box.mp), "recall": float(box.mr), "map50": float(box.map50),
              "map50_95": float(box.map), "per_class": per_class,
              "scope": "Earlier photo batch used as development Val only; no formal Test metrics"}
    if any(not math.isfinite(result[key]) for key in ("precision", "recall", "map50", "map50_95")):
        raise ValueError("validation metrics are not finite")
    return result


def _epochs(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def _checkpoint_names(model) -> dict:
    names = {str(key): value for key, value in model.names.items()}
    if (model.task != "obb" or set(names) != {"0", "1"} or
            not all(isinstance(value, str) for value in names.values()) or
            not names["0"].endswith("_front") or not names["1"].endswith("_back")):
        raise ValueError("initial checkpoint must be OBB with class 0 front and class 1 back")
    return names


def _verify_snapshot(info: dict, yaml_path: Path) -> None:
    _digest(_unlinked(yaml_path, file=True), info["training_data_yaml_sha256"])
    for row in info["training_snapshot"]:
        _digest(_unlinked(row["image_path"], file=True), row["image_sha256"])
        _digest(_unlinked(row["label_path"], file=True), row["label_sha256"])


def run(args) -> dict:
    if min(args.epochs, args.batch, args.imgsz, args.cpu_threads) <= 0 or args.patience < 0:
        raise ValueError("epochs, batch, imgsz and threads must be positive; patience cannot be negative")
    if args.workers != 0:
        raise ValueError("this reproducible Windows route requires --workers 0")
    info = validate_manifest(args.manifest)
    weights = [_unlinked(path, file=True) for group in args.initial_weight for path in group]
    if not weights or len({str(path).casefold() for path in weights}) != len(weights):
        raise ValueError("initial weights must be nonempty and cannot be duplicated")
    if any(path.suffix.lower() != ".pt" for path in weights):
        raise ValueError("initial weights must be existing local .pt checkpoints")
    weight_info = [{"path": str(path), "sha256": sha256_file(path)} for path in weights]
    output = _unlinked(args.output)
    bundle = Path(info["bundle"])
    if (os.path.lexists(output) or output == bundle or output.is_relative_to(bundle) or
            bundle.is_relative_to(output)):
        raise ValueError("output must be an entirely new directory separate from the source bundle")
    output.mkdir(parents=True, exist_ok=False)
    config_directory = output / "ultralytics-config"
    config_directory.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.pop("ULTRALYTICS_API_KEY", None)
    results = {}
    def progress(status, **details):
        _write(output / "training-progress.json", {"evidence_scope": SCOPE,
               "formal_training_gate": "closed", "formal_dataset_v1": False,
               "updated_at": _utc(), "status": status, "models": results, **details})
    progress("preparing")
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS

        SETTINGS.update({key: False for key in ("sync", "wandb", "comet", "mlflow", "clearml", "dvc",
                                               "raytune", "tensorboard") if key in SETTINGS})
        SETTINGS.update({"api_key": ""})
        torch.set_num_threads(args.cpu_threads)
        yaml_path = _snapshot(info, output)
        for index, record in enumerate(weight_info, start=1):
            copied = output / "initial-weights" / f"model-{index:02d}.pt"
            _copy_verified(Path(record["path"]), copied, record["sha256"])
            record["snapshot_path"] = str(copied)
            record["initial_names"] = _checkpoint_names(YOLO(str(copied)))
        config = {"evidence_scope": SCOPE, "formal_training_gate": "closed", "formal_dataset_v1": False,
                  "orientation_semantics": "NOT_APPLICABLE", "started_at": _utc(),
                  "dataset": info, "git": _git(), "environment": _environment(torch),
                  "initial_weights": weight_info, "epochs": args.epochs, "imgsz": args.imgsz,
                  "batch": args.batch, "device": args.device, "workers": 0,
                  "cpu_threads": args.cpu_threads, "seed": args.seed, "patience": args.patience,
                  "augmentation": AUGMENTATION, "validation_augmentation": False,
                  "selection_rule": "highest development Val mAP50-95; ties keep CLI weight order",
                  "test_evaluated": False, "final_holdout_evaluated": False, "limitations": LIMITATIONS}
        _write(output / "training-config.json", config)
        started = time.perf_counter()
        for index, initial in enumerate(weight_info, start=1):
            name = f"model-{index:02d}"
            _verify_snapshot(info, yaml_path)
            _digest(_unlinked(initial["snapshot_path"], file=True), initial["sha256"])
            model = YOLO(initial["snapshot_path"])
            initial_names = _checkpoint_names(model)
            progress("training", current_model=name, completed_epochs=0, requested_epochs=args.epochs)
            model.add_callback("on_fit_epoch_end", lambda trainer: progress(
                "training", current_model=name, completed_epochs=int(trainer.epoch) + 1,
                requested_epochs=args.epochs))
            model_started = time.perf_counter()
            model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                        device=args.device, workers=0, seed=args.seed, deterministic=True,
                        patience=args.patience, project=str(output), name=name, exist_ok=False,
                        pretrained=True, val=True, plots=True, cache=False, **AUGMENTATION)
            seconds = time.perf_counter() - model_started
            model_dir = _unlinked(model.trainer.save_dir)
            if not model_dir.is_relative_to(output):
                raise ValueError("trainer output escaped the run directory")
            best = _unlinked(model_dir / "weights/best.pt", file=True)
            last = _unlinked(model_dir / "weights/last.pt", file=True)
            selected = YOLO(str(best))
            if selected.task != "obb" or {str(k): v for k, v in selected.names.items()} != NAMES:
                raise ValueError("trained model did not preserve the declared Part A OBB class mapping")
            _verify_snapshot(info, yaml_path)
            metrics = selected.val(data=str(yaml_path), split="val", imgsz=args.imgsz, batch=args.batch,
                                   device=args.device, workers=0, augment=False, iou=0.7,
                                   project=str(output), name=name + "-val", exist_ok=False, plots=True)
            results[name] = {"model": name, "initial_weight": initial,
                             "class_remapping": {initial_names[k]: v for k, v in NAMES.items()},
                             "best_pt": str(best), "last_pt": str(last),
                             "best_sha256": sha256_file(best), "last_sha256": sha256_file(last),
                             "actual_epochs": _epochs(model_dir / "results.csv"),
                             "train_seconds": seconds,
                             "validation": _metrics(metrics, info["split_counts"]["val"])}
            progress("model_completed", current_model=name)
        winner = max(results, key=lambda name: results[name]["validation"]["map50_95"])
        selection = {"evidence_scope": SCOPE, "formal_training_gate": "closed", "formal_dataset_v1": False,
                     "frozen_at": _utc(), "selection_uses": "development Val mAP50-95 only",
                     "selected_model": winner, "selected_best_pt": results[winner]["best_pt"],
                     "selected_best_sha256": results[winner]["best_sha256"],
                     "dataset_manifest_sha256": info["manifest_sha256"],
                     "dataset_content_sha256": info["dataset_content_sha256"],
                     "validation_scores": {key: value["validation"]["map50_95"]
                                           for key, value in results.items()},
                     "test_evaluated": False, "final_holdout_evaluated": False,
                     "limitations": LIMITATIONS}
        _write(output / "selection.json", selection)
        summary = {"evidence_scope": SCOPE, "formal_training_gate": "closed", "formal_dataset_v1": False,
                   "completed_at": _utc(), "total_seconds": time.perf_counter() - started,
                   "orientation_semantics": "NOT_APPLICABLE", "models": results,
                   "selection": selection, "split_counts": info["split_counts"],
                   "label_provenance": info["label_provenance"], "test_evaluated": False,
                   "final_holdout_evaluated": False, "limitations": LIMITATIONS}
        _write(output / "result-summary.json", summary)
        progress("completed", selected_model=winner)
        return summary
    except Exception as exc:
        progress("failed", error_type=type(exc).__name__, error=str(exc))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--initial-weight", type=Path, nargs="+", action="append", required=True,
                        help="existing local OBB checkpoints; repeat option or provide multiple paths")
    parser.add_argument("--output", type=Path, required=True, help="entirely new isolated run directory")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"real-photo development training failed: {exc}\n")
    print(f"Development result saved: {args.output.resolve() / 'result-summary.json'}")
    print("Formal Training Gate remains CLOSED; Test and Final Holdout were not evaluated.")
    return 0
