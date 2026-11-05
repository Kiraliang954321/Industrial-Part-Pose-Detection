"""Create an auditable synthetic-only OBB demo using existing session contracts."""

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from yolo_conveyor.annotation import AnnotationSession
from yolo_conveyor.annotation.contract import sha256_file
from yolo_conveyor.capture.session import CaptureSession
from yolo_conveyor.datasets.project_v1 import build_project_dataset
from yolo_conveyor.label_contract import canonical_class_name

from .audit import audit_leakage, image_fingerprints
from .render import CapPose, Scene, cap_corners, conveyor_background, render_cap

GENERATOR_VERSION = "1.0.0"
PART_ID = "synthetic_cap_a"
SPLIT_SESSIONS = {"train": 6, "val": 2, "test": 2, "final_holdout": 2}


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8")


def _jpeg(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise OSError(f"JPEG encoder failed: {path}")
    path.write_bytes(encoded.tobytes())


def _sample_pose(scene: Scene, rng: np.random.Generator) -> CapPose:
    diameter = float(rng.uniform(72, min(220, scene.height * 0.46)))
    compression = float(np.clip(scene.camera_compression + rng.uniform(-0.08, 0.08), 0.60, 1.0))
    angle = float(rng.uniform(-math.pi, math.pi))
    object_height = diameter * (272 / 256) * compression
    extent_x = (abs(math.cos(angle)) * diameter + abs(math.sin(angle)) * object_height) / 2
    extent_y = (abs(math.sin(angle)) * diameter + abs(math.cos(angle)) * object_height) / 2
    return CapPose(float(rng.uniform(extent_x + 12, scene.width - extent_x - 12)),
                   float(rng.uniform(extent_y + 24, scene.height - extent_y - 24)),
                   diameter, compression, angle, int(rng.integers(0, 2**32)),
                   brightness=float(rng.uniform(0.83, 1.16)),
                   blur_sigma=float(rng.choice([0.0, 0.0, 0.0, 0.4, 0.7, 1.0])),
                   occlusion=bool(rng.random() < 0.15))


def _write_video(output: Path, scene: Scene, background: np.ndarray, seed: int,
                 seconds: float, fps: int) -> dict:
    """Synthetic video GT uses complete visibility; partial entry/exit frames are excluded."""
    rng = np.random.default_rng(seed)
    frame_count = round(seconds * fps)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (scene.width, scene.height))
    if not writer.isOpened():
        raise OSError(f"MP4 writer unavailable: {output}")
    events = max(2, round(seconds / 3))
    faces = ["front" if index % 2 == 0 else "back" for index in range(events)]
    rng.shuffle(faces)
    poses = [_sample_pose(scene, rng) for _ in range(events)]
    gt_path = output.with_suffix(".jsonl")
    visible_count = 0
    try:
        with gt_path.open("w", encoding="utf-8", newline="\n") as stream:
            for frame_id in range(frame_count):
                fraction = frame_id / frame_count * events
                event_index = min(events - 1, int(fraction))
                progress = fraction - event_index
                base = poses[event_index]
                # Leave a blank interval before and after each independent passage.
                active = 0.10 <= progress < 0.90
                pose = None
                corners = None
                full_visible = False
                face = faces[event_index] if active else None
                if active:
                    across = (progress - 0.10) / 0.80
                    pose = replace(base,
                                   center_x=-base.diameter + across * (scene.width + 2 * base.diameter),
                                   center_y=scene.height / 2 + math.sin(across * math.pi) * 18,
                                   angle=base.angle + (across - 0.5) * 0.16,
                                   blur_sigma=max(base.blur_sigma, 0.3))
                    corners = cap_corners(scene, pose)
                    full_visible = all(0 <= value <= 1 for point in corners for value in point)
                frame = render_cap(scene, background, face, pose, rng)
                writer.write(frame)
                visible_count += full_visible
                row = {"frame_id": frame_id, "face": face, "corners": corners,
                       "visible": full_visible, "event_id": event_index if active else None,
                       "synthetic_event_id": event_index if active else None,
                       "condition_id": f"synthetic-session-{scene.seed}",
                       "evidence_scope": "synthetic_only",
                       "visibility_semantics": "full_object_and_face_cue_visible",
                       "physical_event_evidence": False}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    finally:
        writer.release()
    # Check codec output and frame count rather than assuming write() succeeded.
    capture = cv2.VideoCapture(str(output))
    decoded_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (scene.height, scene.width):
                raise ValueError(f"video dimensions changed: {output}")
            decoded_count += 1
    finally:
        capture.release()
    if decoded_count != frame_count:
        raise ValueError(f"video frame count mismatch: {decoded_count} != {frame_count}")
    return {"video_seed": seed, "video_frames": frame_count, "video_fps": fps,
            "video_seconds": frame_count / fps,
            "video_complete_visible_frames": visible_count,
            "video_sha256": sha256_file(output), "video_gt_sha256": sha256_file(gt_path)}


def _contact_sheet(path: Path, samples: list[tuple[Path, str]]) -> None:
    tile_w, tile_h, columns = 200, 170, 6
    rows = math.ceil(len(samples) / columns)
    canvas = np.full((rows * tile_h, columns * tile_w, 3), 240, dtype=np.uint8)
    for index, (source, caption) in enumerate(samples):
        image = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode contact-sheet source: {source}")
        tile = cv2.resize(image, (tile_w, 150), interpolation=cv2.INTER_AREA)
        x, y = index % columns * tile_w, index // columns * tile_h
        canvas[y:y + 150, x:x + tile_w] = tile
        cv2.putText(canvas, caption, (x + 3, y + 164), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (25, 25, 25), 1, cv2.LINE_AA)
    _jpeg(path, canvas)


def generate_bundle(output: str | Path, *, seed: int = 20261003,
                    images_per_session: int = 120, width: int = 640, height: int = 480,
                    video_seconds: float = 12.0, fps: int = 30,
                    progress: Callable[[str], None] | None = None) -> dict:
    """Write a fresh 12-session demo bundle; partial failures remain visible and are never deleted."""
    if (type(seed) is not int or not 0 <= seed < 2**32 or
            type(images_per_session) is not int or images_per_session < 2 or images_per_session % 2 or
            type(width) is not int or type(height) is not int or width < 320 or height < 240 or
            type(fps) is not int or not 1 <= fps <= 120 or
            not math.isfinite(video_seconds) or not 1 <= video_seconds <= 120):
        raise ValueError("require uint32 seed, even image count >=2, dimensions >=320x240, "
                         "fps 1..120 and finite video_seconds 1..120")
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"output must be new: {output}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "videos").mkdir()
    notify = progress or (lambda _: None)
    sessions = []
    annotations = []
    all_hashes = []
    contact_samples = []
    splits = {name: [] for name in SPLIT_SESSIONS}
    # SeedSequence children provide stable, distinct session seed streams.
    children = np.random.SeedSequence(seed).spawn(sum(SPLIT_SESSIONS.values()))
    session_index = 0
    for split, count in SPLIT_SESSIONS.items():
        for _ in range(count):
            child = children[session_index]
            session_seed = int(child.generate_state(1)[0])
            session_id = f"synthetic-cap-{session_index + 1:02d}-{session_seed:08x}"
            splits[split].append(session_id)
            rng = np.random.default_rng(session_seed)
            scene = Scene.create(width, height, session_seed)
            background = conveyor_background(scene)
            config = {"evidence_scope": "synthetic_only", "generator_version": GENERATOR_VERSION,
                      "orientation_semantics": "NOT_APPLICABLE", "seed": session_seed,
                      "scene": scene.description(), "images_per_session": images_per_session}
            capture = CaptureSession.create(output / "captures", part_id=PART_ID,
                                            source=f"synthetic://procedural-cap/{session_id}",
                                            backend="procedural_numpy_opencv", config=config,
                                            session_id=session_id)
            faces = ["front"] * (images_per_session // 2) + ["back"] * (images_per_session // 2)
            rng.shuffle(faces)
            records = []
            poses = []
            for image_index, face in enumerate(faces):
                pose = _sample_pose(scene, rng)
                corners = cap_corners(scene, pose)
                if not all(0 <= value <= 1 for point in corners for value in point):
                    raise ValueError("generated training object is outside image")
                image = render_cap(scene, background, face, pose, rng)
                relative = f"{face}/cap-{image_index:05d}.jpg"
                source = capture.path / relative
                _jpeg(source, image)
                jpeg_sha, pixel_sha, dhash = image_fingerprints(source)
                metrics = {"synthetic_geometry_complete": True,
                           "visible_face_cue": True, "blur_sigma": pose.blur_sigma,
                           "light_occlusion": pose.occlusion}
                capture.record_attempt(label=face, mode="auto", accepted=True,
                                       saved_path=relative, quality_metrics=metrics,
                                       warnings=("synthetic_only",), hash_value=jpeg_sha,
                                       timestamp=f"synthetic-seed-{session_seed}-sample-{image_index}")
                records.append({"schema_version": 1, "status": "annotated", "part_id": PART_ID,
                                "face": face, "class_name": canonical_class_name(PART_ID, face),
                                "source_path": relative, "source_image_sha256": jpeg_sha,
                                "image_width": width, "image_height": height, "corners": corners})
                all_hashes.append({"path": source.relative_to(output).as_posix(), "session_id": session_id,
                                   "split": split, "jpeg_sha256": jpeg_sha,
                                   "decoded_pixel_sha256": pixel_sha, "dhash": dhash})
                poses.append({"source_path": relative, "face": face, "pose": asdict(pose)})
                if image_index < 6:
                    contact_samples.append((source, f"{split} s{session_index + 1:02d} {face}"))
            # Freeze the source manifest before annotation hashes are created.
            store = AnnotationSession.create(output / "annotations" / session_id, capture.path)
            for record in records:
                store.write_record(record)
            annotations.append(store.path)
            _json(output / "annotations" / session_id / "synthetic_poses.json", poses)
            video_path = output / "videos" / f"{session_id}.mp4"
            video_info = _write_video(video_path, scene, background,
                                      int(child.generate_state(2)[1]), video_seconds, fps)
            sessions.append({"session_id": session_id, "split": split, "seed": session_seed,
                             "source_session": capture.path.relative_to(output).as_posix(),
                             "annotation_workspace": store.path.relative_to(output).as_posix(),
                             "video_path": video_path.relative_to(output).as_posix(),
                             "video_gt_path": video_path.with_suffix(".jsonl").relative_to(output).as_posix(),
                             "scene": scene.description(), "images": images_per_session,
                             "face_counts": {"front": images_per_session // 2, "back": images_per_session // 2},
                             **video_info})
            notify(f"generated {session_index + 1}/12: {split} {session_id}")
            session_index += 1
    split_path = output / "split_plan.json"
    _json(split_path, {"schema_version": 2, "splits": splits})
    dataset = build_project_dataset(annotations, split_path, output / "dataset")
    leakage = audit_leakage(all_hashes)
    _json(output / "leakage_report.json", leakage)
    if leakage["exact_cross_split_duplicates"]:
        raise ValueError("cross-split exact duplicate audit failed; partial bundle preserved")
    _contact_sheet(output / "contact_sheet.jpg", contact_samples)
    provenance = {"schema_version": 1, "evidence_scope": "synthetic_only", "part_id": PART_ID,
                  "orientation_semantics": "NOT_APPLICABLE", "angle_ground_truth": None,
                  "generator_version": GENERATOR_VERSION, "seed": seed,
                  "runtime_versions": {"numpy": np.__version__, "opencv": cv2.__version__},
                  "session_split_semantics": "whole independently seeded generated scene sessions",
                  "images_source": "procedural renders, independently sampled from video events",
                  "reference_image_used_as_training_texture": False,
                  "physical_experiment_protocol": "unchanged; not frozen by synthetic generation",
                  "formal_training_gate": "closed", "physical_event_evidence": False,
                  "dataset_manifest_sha256": sha256_file(output / "dataset" / "dataset_manifest.json"),
                  "leakage_report_sha256": sha256_file(output / "leakage_report.json"),
                  "sessions": sessions, "counts": dataset["counts"],
                  "limitations": ["Shared procedural templates across all splits.",
                                  "No real camera, physical object, or production performance evidence.",
                                  "Face classes are known generated surfaces, not inferred from OBB angles."]}
    _json(output / "synthetic_provenance.json", provenance)
    notify("bundle assembled; exact JPEG/pixel hash audit passed")
    return provenance
