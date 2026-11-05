"""Read-only audit of the two-keypoint simulation development dataset."""

from collections import Counter
import json
from pathlib import Path

from yolo_conveyor.orientation.common import TASK, SCOPE, KEYPOINTS, dataset_file, sha256, unlinked
from yolo_conveyor.orientation.geometry import heading, directed_error


def data_yaml(root: Path) -> dict:
    # JSON is a YAML subset. No URLs, download hooks, Test, or Final Holdout.
    return {"path": root.as_posix(), "train": "images/train", "val": "images/val",
            "names": {"0": "screw"}, "kpt_shape": [2, 3], "flip_idx": [0, 1],
            "kpt_names": {"0": KEYPOINTS}}


def label_text(objects: list, size: int) -> str:
    if not objects:
        return ""
    if len(objects) != 1:
        raise ValueError("V1 supports zero or one screw per image")
    obj = objects[0]
    x1, y1, x2, y2 = obj["bbox_xyxy"]
    values = [0, (x1 + x2) / (2 * size), (y1 + y2) / (2 * size),
              (x2 - x1) / size, (y2 - y1) / size]
    for point, visible in zip(obj["keypoints_xy"], obj["visibility"]):
        values.extend([point[0] / size, point[1] / size, visible])
    return " ".join(f"{v:.8f}" for v in values) + "\n"


def load_dataset(path: str | Path) -> tuple[dict, list[dict]]:
    root = unlinked(path)
    meta = json.loads(dataset_file(root, "dataset.json").read_text(encoding="utf-8"))
    if (meta.get("schema_version") != 1 or meta.get("task") != TASK or meta.get("scope") != SCOPE or
            meta.get("status") != "completed" or meta.get("formal_training_gate") != "closed" or
            meta.get("keypoint_names") != KEYPOINTS or set(meta.get("counts", {})) != {"train", "val"}):
        raise ValueError("unsupported or incomplete dataset contract")
    for name, key in (("records.jsonl", "records_sha256"), ("data.yaml", "yaml_sha256"),
                      ("scene.xml", "scene_sha256"), ("asset.json", "asset_sha256")):
        if sha256(dataset_file(root, name)) != meta.get(key):
            raise ValueError(f"dataset metadata/hash drift: {name}")
    if json.loads(dataset_file(root, "data.yaml").read_text(encoding="utf-8")) != data_yaml(root):
        raise ValueError("data.yaml must use this dataset's Train/Val only, with the exact two-point mapping")
    if json.loads(dataset_file(root, "asset.json").read_text(encoding="utf-8")) != meta.get("asset"):
        raise ValueError("asset metadata does not match asset.json")
    rows = [json.loads(line) for line in dataset_file(root, "records.jsonl").read_text(encoding="utf-8").splitlines()]
    return meta, rows


def audit_dataset(path: str | Path) -> dict:
    import cv2
    import numpy as np
    root = unlinked(path)
    report = {"task": TASK, "scope": SCOPE, "formal_training_gate": "closed", "errors": [],
              "counts": {}, "conditions": {}, "heading_bins_45deg": {}, "visibility_counts": {},
              "independence": "one generated snapshot per independently seeded scene; same known asset"}
    try:
        meta, rows = load_dataset(root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report["errors"].append(str(exc))
        return report
    size = meta.get("image_size")
    if type(size) is not int or not 64 <= size <= 2048:
        report["errors"].append("invalid image size")
        return report
    seen_ids, seen_images, seen_hashes = set(), set(), set()
    split_counts, conditions, bins, visibility = Counter(), Counter(), Counter(), Counter()
    positives = Counter()
    for row in rows:
        try:
            split = row["split"]
            if (split not in ("train", "val") or row["source_family"] != row["scene_id"] or
                    row["domain"] != "simulation" or row["condition"] not in meta["conditions"]):
                raise ValueError("unsupported split or non-independent scene family")
            if row["scene_id"] in seen_ids or row["image"] in seen_images or row["image_sha256"] in seen_hashes:
                raise ValueError("duplicate scene, image path, or image bytes")
            seen_ids.add(row["scene_id"])
            seen_images.add(row["image"])
            seen_hashes.add(row["image_sha256"])
            if not row["image"].startswith(f"images/{split}/") or not row["label"].startswith(f"labels/{split}/"):
                raise ValueError("row path does not match split")
            image_path, labels = dataset_file(root, row["image"]), dataset_file(root, row["label"])
            if sha256(image_path) != row["image_sha256"] or sha256(labels) != row["label_sha256"]:
                raise ValueError("image or label hash drift")
            image = cv2.imread(str(image_path))
            if image is None or image.shape != (size, size, 3):
                raise ValueError("invalid decoded image dimensions")
            objects = row["objects"]
            if not isinstance(objects, list) or len(objects) > 1:
                raise ValueError("expected zero or one object")
            if labels.read_text(encoding="utf-8") != label_text(objects, size):
                raise ValueError("YOLO label does not match independent metadata")
            cam = row["camera"]
            k = np.asarray(cam["K"], dtype=float)
            twc = np.asarray(cam["T_world_camera_cv"], dtype=float)
            if (k.shape != (3, 3) or twc.shape != (4, 4) or not np.isfinite(k).all() or
                    not np.isfinite(twc).all() or k[0, 0] <= 0 or k[1, 1] <= 0):
                raise ValueError("invalid camera calibration")
            for obj in objects:
                if obj["class_id"] != 0 or obj["orientation_gt_status"] != "VALID":
                    raise ValueError("only screw class with valid synthetic orientation GT is supported")
                points = np.asarray(obj["keypoints_xy"], dtype=float)
                box = np.asarray(obj["bbox_xyxy"], dtype=float)
                if (points.shape != (2, 2) or not np.isfinite(points).all() or
                        np.any(points < 0) or np.any(points >= size) or
                        box.shape != (4,) or not np.isfinite(box).all() or np.any(box < 0) or np.any(box > size) or
                        box[2] <= box[0] or box[3] <= box[1]):
                    raise ValueError("invalid or out-of-image geometry")
                if len(obj["visibility"]) != 2 or any(v not in (1, 2) for v in obj["visibility"]):
                    raise ValueError("both projected points must be known; visibility is 1 or 2")
                angle = heading(points)
                if angle is None or directed_error(angle, obj["heading_deg"]) > 1e-6:
                    raise ValueError("heading is inconsistent with ordered points")
                t = np.asarray(obj["T_world_object"], dtype=float)
                anchors = np.asarray(meta["asset"]["keypoints_object_m"], dtype=float)
                if t.shape != (4, 4) or not np.isfinite(t).all():
                    raise ValueError("invalid object transform")
                for transform in (twc, t):
                    rotation = transform[:3, :3]
                    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
                            not np.allclose(transform[3], [0, 0, 0, 1]) or
                            np.dot(np.cross(rotation[:, 0], rotation[:, 1]), rotation[:, 2]) < 0.999):
                        raise ValueError("transform must be a proper rigid transform")
                world_points = anchors @ t[:3, :3].T + t[:3, 3]
                camera_points = (world_points - twc[:3, 3]) @ twc[:3, :3]
                if np.any(camera_points[:, 2] <= 0):
                    raise ValueError("keypoint behind camera")
                projected = camera_points @ k.T
                projected = projected[:, :2] / projected[:, 2:3]
                if not np.isfinite(projected).all() or not np.allclose(projected, points, atol=1e-5):
                    raise ValueError("3D-to-2D reprojection mismatch")
                positives[split] += 1
                bins[f"{split}:{int(angle // 45) * 45}"] += 1
                visibility.update(map(str, obj["visibility"]))
            split_counts[split] += 1
            conditions[f"{split}:{row['condition']}"] += 1
        except (ValueError, OSError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
            report["errors"].append(f"{row.get('scene_id', '?')}: {exc}")
    if dict(split_counts) != meta["counts"] or any(positives[s] == 0 for s in ("train", "val")):
        report["errors"].append("split counts mismatch or a split has no positive examples")
    for split in ("train", "val"):
        expected_images = {root / row["image"] for row in rows if row["split"] == split}
        expected_labels = {root / row["label"] for row in rows if row["split"] == split}
        if set((root / "images" / split).glob("*")) != expected_images or set((root / "labels" / split).glob("*")) != expected_labels:
            report["errors"].append(f"unlisted or missing files in {split}")
    report.update(counts=dict(split_counts), positive_counts=dict(positives), conditions=dict(conditions),
                  heading_bins_45deg=dict(bins), visibility_counts=dict(visibility))
    return report
