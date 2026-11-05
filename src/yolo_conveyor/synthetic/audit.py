"""Content hashes and bounded perceptual-similarity candidates for synthetic assets."""

from collections import defaultdict
import hashlib
from pathlib import Path

import cv2
import numpy as np

from yolo_conveyor.annotation.contract import sha256_file


def image_fingerprints(path: Path) -> tuple[str, str, int]:
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode generated JPEG: {path}")
    decoded = hashlib.sha256(image.tobytes()).hexdigest()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
    return sha256_file(path), decoded, dhash


def audit_leakage(items: list[dict], *, threshold: int = 4) -> dict:
    """Exact equality is an error; small dHash distance is only a review candidate."""
    exact_groups = []
    for key in ("jpeg_sha256", "decoded_pixel_sha256"):
        groups = defaultdict(list)
        for item in items:
            groups[item[key]].append(item)
        for digest, entries in groups.items():
            if len({entry["split"] for entry in entries}) > 1:
                exact_groups.append({"hash_kind": key, "hash": digest,
                                     "images": [entry["path"] for entry in entries]})
    pairs = set()
    for group in exact_groups:
        for a in group["images"]:
            for b in group["images"]:
                if a < b:
                    left = next(item for item in items if item["path"] == a)
                    right = next(item for item in items if item["path"] == b)
                    if left["split"] != right["split"]:
                        pairs.add((a, b))
    candidate_count = 0
    candidates = []
    for index, left in enumerate(items):
        for right in items[index + 1:]:
            if left["split"] == right["split"]:
                continue
            distance = (left["dhash"] ^ right["dhash"]).bit_count()
            if distance <= threshold:
                candidate_count += 1
                if len(candidates) < 100:
                    candidates.append({"left": left["path"], "right": right["path"],
                                       "dhash_distance": distance})
    return {
        "schema_version": 1,
        "evidence_scope": "synthetic_only",
        "exact_cross_split_duplicates": len(pairs),
        "cross_split_exact_duplicates": exact_groups,
        "counts": {split: sum(item["split"] == split for item in items)
                   for split in ("train", "val", "test", "final_holdout")},
        "hash_audit": [{key: value for key, value in item.items() if key != "dhash"}
                       for item in items],
        "near_duplicate_candidates": {
            "method": "64-bit full-frame dHash",
            "maximum_hamming_distance": threshold,
            "candidate_count": candidate_count,
            "examples": candidates,
            "examples_truncated": candidate_count > len(candidates),
            "interpretation": "Candidates require visual review; not proof of leakage. "
                              "Small objects on similar conveyor backgrounds can have similar hashes.",
        },
        "shared_templates": {
            "cap_surface": "All splits share the same procedural cap topology and face-graphic templates.",
            "background": "All splits share renderer equations, but session seeds, texture, illumination "
                          "and camera compression are independently sampled.",
            "limitation": "Session separation measures simulator generalization only; it does not "
                          "establish independence from the rendering family or physical generalization.",
        },
        "status": "pass_exact_hash_audit" if not pairs else "fail_exact_cross_split_duplicates",
    }
