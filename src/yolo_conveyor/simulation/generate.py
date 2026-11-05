"""Generate auditable, group-isolated Train/Val snapshots of a screw proxy."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from yolo_conveyor.datasets.screw_orientation import audit_dataset, data_yaml, label_text
from yolo_conveyor.orientation.common import (TASK, SCOPE, KEYPOINTS, environment, new_directory,
                                             sha256, source_hashes, utc_now, write_json)
from yolo_conveyor.orientation.geometry import heading
from .screw_scene import ASSET, CONDITIONS, ScrewScene


def generate(args) -> dict:
    import cv2
    import numpy as np
    if (not 0 <= args.seed < 2**32 or not 128 <= args.size <= 1024 or args.size % 32 or
            min(args.train_count, args.val_count) < 6 or max(args.train_count, args.val_count) > 10000):
        raise ValueError("require seed uint32, size 128..1024 divisible by 32, counts 6..10000")
    out = new_directory(args.output)
    started = time.perf_counter()
    scene = None
    try:
        scene = ScrewScene(args.size)
        (out / "scene.xml").write_text(scene.xml, encoding="utf-8")
        write_json(out / "asset.json", ASSET)
        write_json(out / "environment.json", environment())
        write_json(out / "data.yaml", data_yaml(out))
        (out / "previews").mkdir()
        rows, hashes, retries = [], set(), Counter()
        written_bytes = 0
        counts = {"train": args.train_count, "val": args.val_count}
        for split_index, (split, count) in enumerate(counts.items()):
            (out / "images" / split).mkdir(parents=True)
            (out / "labels" / split).mkdir(parents=True)
            for index in range(count):
                condition = CONDITIONS[index % len(CONDITIONS)]
                scene_id = f"seed{args.seed}_{split}_{index:05d}"
                for attempt in range(20):
                    seed = [args.seed, split_index, index, attempt]
                    rng = np.random.default_rng(np.random.SeedSequence(seed))
                    # Deterministic negatives in every condition, without conditioning on direction.
                    has_object = (index // len(CONDITIONS)) % 10 != 9
                    bgr, mask, points, vis, camera, pose, params = scene.render(rng, condition, has_object)
                    objects = []
                    if has_object:
                        ys, xs = np.nonzero(mask)
                        if (len(xs) < 40 or min(xs) <= 1 or min(ys) <= 1 or max(xs) >= args.size - 2 or
                                max(ys) >= args.size - 2 or not np.isfinite(points).all() or
                                np.any(points < 0) or np.any(points >= args.size) or 0 in vis or
                                heading(points, min_length=12) is None):
                            retries["clipped_or_degenerate"] += 1
                            continue
                        objects = [{"class_id": 0, "bbox_xyxy": [int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1],
                                    "keypoints_xy": points.tolist(), "visibility": vis,
                                    "heading_deg": heading(points), "orientation_gt_status": "VALID",
                                    "T_world_object": pose.tolist(), "visible_mask_pixels": int(mask.sum())}]
                    elif mask.any():
                        raise ValueError("negative scene unexpectedly contains the screw")
                    success, encoded = cv2.imencode(".png", bgr)
                    if not success:
                        raise RuntimeError("PNG encoding failed")
                    payload = encoded.tobytes()
                    image_hash = hashlib.sha256(payload).hexdigest()
                    if image_hash in hashes:
                        retries["duplicate_image"] += 1
                        continue
                    if written_bytes + len(payload) > 10 * 1024**3:
                        raise RuntimeError("10 GiB generation budget reached; partial outputs preserved")
                    break
                else:
                    raise RuntimeError(f"could not produce an auditable scene: {scene_id}")
                hashes.add(image_hash)
                image_rel, label_rel = f"images/{split}/{scene_id}.png", f"labels/{split}/{scene_id}.txt"
                with (out / image_rel).open("xb") as stream:
                    stream.write(payload)
                with (out / label_rel).open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(label_text(objects, args.size))
                written_bytes += len(payload)
                row = {"scene_id": scene_id, "source_family": scene_id, "split": split, "condition": condition,
                       "domain": "simulation", "seed_sequence": seed, "image": image_rel, "label": label_rel,
                       "image_sha256": image_hash, "label_sha256": sha256(out / label_rel),
                       "objects": objects, "camera": camera, "generation_parameters": params}
                rows.append(row)
                if index < 12:
                    overlay = bgr.copy()
                    if objects:
                        head, tip = [tuple(np.round(p).astype(int)) for p in points]
                        cv2.arrowedLine(overlay, tip, head, (50, 230, 60), 2, tipLength=0.15)
                        cv2.circle(overlay, head, 4, (20, 20, 240), -1)
                        cv2.circle(overlay, tip, 4, (240, 150, 20), -1)
                    cv2.putText(overlay, f"GT ONLY | {split} | {condition}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if not cv2.imwrite(str(out / "previews" / f"{scene_id}.jpg"), overlay):
                        raise RuntimeError("preview write failed")
                if index % 100 == 0 or index == count - 1:
                    progress = {"status": "generating", "split": split, "completed": index + 1, "requested": count,
                                "seconds": time.perf_counter() - started, "updated_at": utc_now()}
                    write_json(out / "progress.json", progress, replace=True)
                    print(json.dumps(progress), flush=True)
        with (out / "records.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        write_json(out / "split-plan.json", {"schema_version": 1, "splits": {
            split: [r["scene_id"] for r in rows if r["split"] == split] for split in counts},
            "test": "not generated; requires frozen simulation evaluation protocol",
            "final_holdout": "not generated; never exposed to development training"})
        meta = {"schema_version": 1, "task": TASK, "scope": SCOPE, "status": "completed",
                "formal_training_gate": "closed", "keypoint_names": KEYPOINTS, "counts": counts,
                "image_size": args.size, "seed": args.seed, "asset": ASSET, "conditions": list(CONDITIONS),
                "records_sha256": sha256(out / "records.jsonl"), "yaml_sha256": sha256(out / "data.yaml"),
                "scene_sha256": sha256(out / "scene.xml"), "asset_sha256": sha256(out / "asset.json"),
                "source_hashes": source_hashes(), "resampled_scenes": dict(retries), "created_at": utc_now(),
                "generation_seconds": time.perf_counter() - started, "image_bytes": written_bytes,
                "limitations": ["Known procedural proxy, not physical Part A", "Train/Val development only",
                                "One independent rendered snapshot per scene", "No real-world or 6D claim",
                                "Occluder is a visual-only obstacle; blur is a fixed 5px horizontal image filter"]}
        write_json(out / "dataset.json", meta)
        audit = audit_dataset(out)
        write_json(out / "audit.json", audit)
        if audit["errors"]:
            raise ValueError(f"dataset audit failed: {audit['errors'][:5]}")
        write_json(out / "progress.json", {"status": "completed", "counts": counts, "updated_at": utc_now()}, replace=True)
        return meta
    except BaseException as exc:
        write_json(out / "progress.json", {"status": "failed", "error": str(exc), "updated_at": utc_now()}, replace=True)
        raise
    finally:
        if scene is not None:
            scene.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=400)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    meta = generate(args)
    print(json.dumps({"status": "completed", "counts": meta["counts"], "seconds": meta["generation_seconds"]}))
    return 0
