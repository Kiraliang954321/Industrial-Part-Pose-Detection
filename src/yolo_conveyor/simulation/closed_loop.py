"""Paired GT/vision Panda trials with independent physical outcome evaluation."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from yolo_conveyor.orientation.common import new_directory, write_json, sha256, source_hashes, environment, utc_now, unlinked
from yolo_conveyor.orientation.evaluation import rate, distribution
from yolo_conveyor.orientation.geometry import directed_error
from .panda_scene import PandaScene, ADAPTATION
from .panda_control import PandaController
from .grasp_geometry import from_ground_truth, from_prediction
from .grasp_evaluation import PhysicalEvaluator, CRITERIA
from .m4_protocol import validate as validate_protocol

WEIGHT_SHA256 = "0a29b8c5660d62cb19b84bf2cc0805e1c3b3ed2fb2e7478dc80813ab2f1e155a"
LIGHT_CONDITIONS = ("nominal", "low_light", "reflection", "background_shift")
SCOPE = "simulation_closed_loop_development"


class VideoRecorder:
    def __init__(self, scene, path, branch, seed):
        self.scene, self.branch, self.seed = scene, branch, seed
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (1280, 704))
        if not self.writer.isOpened():
            raise RuntimeError("video writer unavailable")
        self.frames = 0

    def sample(self, stage, seconds):
        left, right = self.scene.capture("overview"), self.scene.capture("closeup")
        frame = np.vstack([np.full((64, 1280, 3), 242, dtype=np.uint8), np.hstack([left, right])])
        cv2.putText(frame, f"{self.branch} | seed {self.seed} | {stage} | sim {seconds:.2f}s", (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1)
        cv2.putText(frame, "PHYSICS SIMULATION | custom thin pads | no adhesion / no object pose reset during motion", (16, 51),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        self.writer.release()


def summarize_branch(results):
    attempted = [r for r in results if r["attempted"]]
    reasons = Counter(reason for r in results for reason in r["failure_reasons"])
    return {"episodes": len(results), "success": rate(sum(r["success"] for r in results), len(results)),
            "attempted": len(attempted), "rejected_before_execution": len(results) - len(attempted),
            "outcomes_all_episodes": {key: rate(sum(r.get("conditions", {}).get(key, False) for r in results), len(results))
                                      for key in ("bilateral_grasp", "lifted", "held", "transported", "released", "placement_xy", "placement_heading", "settled", "collision_free", "penetration_bounded")},
            "nonexclusive_failure_reasons": dict(reasons),
            "placement_xy_error_m_attempted": distribution([r["placement_xy_error_m"] for r in attempted]),
            "placement_heading_error_deg_attempted": distribution([r["placement_heading_error_deg"] for r in attempted]),
            "maximum_penetration_m": max((r["maximum_penetration_m"] for r in attempted), default=None)}


def run(args):
    if not 1 <= args.episodes <= 100 or not 0 <= args.record_first <= args.episodes or not 0 <= args.seed <= 2**32 - args.episodes:
        raise ValueError("invalid episode, recording, or seed budget")
    weights = unlinked(args.weights)
    if not weights.is_file() or sha256(weights) != WEIGHT_SHA256:
        raise ValueError("M4 uses the frozen M3 selected checkpoint")
    protocol = None
    if args.role == "paired_evaluation":
        if not args.protocol:
            raise ValueError("paired evaluation requires --protocol frozen after calibration")
        protocol = validate_protocol(args.protocol, args.seed, args.episodes, WEIGHT_SHA256,
                                     sha256(unlinked(args.assets) / "asset-manifest.json"), LIGHT_CONDITIONS)
    out = new_directory(args.output, [args.assets, weights])
    (out / "yolo-config").mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(out / "yolo-config")
    scene = None
    started = time.perf_counter()
    try:
        from yolo_conveyor.orientation.pose_adapter import PosePredictor
        from yolo_conveyor.orientation.evaluate import draw_predictions
        import torch
        torch.set_num_threads(4)
        cv2.setNumThreads(2)
        scene = PandaScene(args.assets)
        controller = PandaController(scene)
        predictor = PosePredictor(weights, device=args.device)
        predictor.predict(np.zeros((640, 640, 3), dtype=np.uint8)); predictor.synchronize()
        manifest = {"scope": SCOPE, "real_formal_gate": "closed", "run_role": args.role, "arguments": vars(args),
                    "protocol": protocol,
                    "created_at": utc_now(), "adaptation": ADAPTATION, "criteria": CRITERIA,
                    "weight_sha256": WEIGHT_SHA256, "asset_manifest_sha256": sha256(Path(args.assets) / "asset-manifest.json"),
                    "environment": environment(), "source_hashes": source_hashes(),
                    "scene_seeds": list(range(args.seed, args.seed + args.episodes)),
                    "light_conditions": list(LIGHT_CONDITIONS),
                    "planning_priors": {"head_height_m": 0.006, "tip_height_m": 0.003, "grasp_height_m": 0.0045,
                                        "release_height_m": 0.010, "maximum_joint_speed_rad_s": 1.2,
                                        "opening_half_width_m": 0.04, "closing_target_half_width_m": 0.0},
                    "boundaries": ["Fixed-camera, fixed-support 2.5D task; no full 6D prediction",
                                   "The controller receives only GraspObservation and robot joint feedback",
                                   "Object GT is used only for reference observations and outcome evaluation",
                                   "M3 weights frozen; no vision training or threshold tuning in this run",
                                   "Original finger coupling equality retained; no object attachment constraint"]}
        write_json(out / "run-manifest.json", manifest)
        (out / "scene.xml").write_text(scene.xml, encoding="utf-8")
        source_root = Path(__file__).resolve().parents[3]
        for relative, expected in manifest["source_hashes"].items():
            destination = out / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_root / relative, destination)
            if sha256(destination) != expected:
                raise ValueError("source changed while snapshotting")
        rows = []
        for index in range(args.episodes):
            seed = args.seed + index
            condition = LIGHT_CONDITIONS[index % len(LIGHT_CONDITIONS)]
            episode = out / f"episode-{seed}"
            episode.mkdir()
            image, _, _, _, camera, _, parameters = scene.render(np.random.default_rng(seed), condition, True)
            snapshot = scene.snapshot()
            initial_state_hash = hashlib.sha256(snapshot.tobytes()).hexdigest()
            reference = from_ground_truth(scene.object_anchors())
            t0 = time.perf_counter()
            predictions = predictor.predict(image); predictor.synchronize()
            inference_ms = (time.perf_counter() - t0) * 1000
            vision_error = None
            try:
                vision = from_prediction(predictions, camera)
            except ValueError as exc:
                vision, vision_error = None, str(exc)
            if not cv2.imwrite(str(episode / "sensor.png"), image):
                raise RuntimeError("sensor image write failed")
            if not cv2.imwrite(str(episode / "prediction.jpg"), draw_predictions(image, predictions, "M4 RGB prediction | SIM DEV")):
                raise RuntimeError("prediction image write failed")
            observations = {"gt": reference, "vision": vision}
            row = {"seed": seed, "condition": condition, "initial_state_sha256": initial_state_hash,
                   "sensor_sha256": sha256(episode / "sensor.png"), "camera": camera, "generation_parameters": parameters,
                   "predictions": predictions, "perception_ms": inference_ms,
                   "observations": {k: v.record() if v else None for k, v in observations.items()}, "branches": {}}
            if vision:
                row["perception_error"] = {"xy_m": float(np.linalg.norm(np.subtract(vision.center_xy, reference.center_xy))),
                                            "heading_deg": directed_error(math.degrees(vision.heading_rad), math.degrees(reference.heading_rad))}
            for branch, observation in observations.items():
                scene.restore(snapshot)
                actual_initial_hash = hashlib.sha256(scene.snapshot().tobytes()).hexdigest()
                # Compare restored states across both branches, including solver warm-start/control state.
                if branch == "gt":
                    restored_hash = actual_initial_hash
                elif actual_initial_hash != restored_hash:
                    raise ValueError("paired initial states differ")
                if observation is None:
                    result = {"attempted": False, "success": False, "failure_reasons": ["perception_rejected"], "detail": vision_error}
                else:
                    t0 = time.perf_counter()
                    try:
                        stages, planning = controller.plan(observation)
                    except ValueError as exc:
                        result = {"attempted": False, "success": False, "failure_reasons": ["planning_rejected"], "detail": str(exc)}
                    else:
                        planning_ms = (time.perf_counter() - t0) * 1000
                        video = VideoRecorder(scene, episode / f"{branch}.mp4", branch, seed) if index < args.record_first else None
                        try:
                            with (episode / f"{branch}-trace.jsonl").open("x", encoding="utf-8", newline="\n") as trace:
                                evaluator = PhysicalEvaluator(scene, trace, video.sample if video else None)
                                controller.execute(stages, evaluator.sample)
                                result = evaluator.result()
                            result.update(planning=planning, planning_ms=planning_ms,
                                          commanded_stage_seconds={stage.name: stage.duration for stage in stages},
                                          video_frames=video.frames if video else 0)
                            if not cv2.imwrite(str(episode / f"{branch}-final.png"), scene.capture("closeup")):
                                raise RuntimeError("final-state image write failed")
                        finally:
                            if video:
                                video.close()
                result["restored_initial_state_sha256"] = actual_initial_hash
                row["branches"][branch] = result
            write_json(episode / "result.json", row)
            rows.append(row)
            progress = {"status": "running", "completed_pairs": len(rows), "requested_pairs": args.episodes,
                        "gt_success": sum(r["branches"]["gt"]["success"] for r in rows),
                        "vision_success": sum(r["branches"]["vision"]["success"] for r in rows), "updated_at": utc_now()}
            write_json(out / "progress.json", progress, replace=True)
            print(json.dumps(progress), flush=True)
        report = {"status": "completed", "scope": SCOPE, "real_formal_gate": "closed", "run_role": args.role,
                  "manifest_sha256": sha256(out / "run-manifest.json"), "pairs": len(rows),
                  "overall": {b: summarize_branch([r["branches"][b] for r in rows]) for b in ("gt", "vision")},
                  "by_condition": {c: {b: summarize_branch([r["branches"][b] for r in rows if r["condition"] == c])
                                        for b in ("gt", "vision")} for c in sorted({r["condition"] for r in rows})},
                  "paired_outcomes": {"both_success": sum(r["branches"]["gt"]["success"] and r["branches"]["vision"]["success"] for r in rows),
                                      "gt_only": sum(r["branches"]["gt"]["success"] and not r["branches"]["vision"]["success"] for r in rows),
                                      "vision_only": sum(not r["branches"]["gt"]["success"] and r["branches"]["vision"]["success"] for r in rows),
                                      "both_failed": sum(not r["branches"]["gt"]["success"] and not r["branches"]["vision"]["success"] for r in rows)},
                  "perception_ms": distribution([r["perception_ms"] for r in rows]),
                  "wall_seconds_including_rendering": time.perf_counter() - started, "completed_at": utc_now(),
                  "claims": "Paired physics simulation development only; not a trained action policy or real robot validation"}
        write_json(out / "report.json", report)
        write_json(out / "progress.json", {"status": "completed", "pairs": len(rows), "updated_at": utc_now()}, replace=True)
        return report
    except BaseException as exc:
        write_json(out / "progress.json", {"status": "failed", "error": str(exc), "updated_at": utc_now()}, replace=True)
        raise
    finally:
        if scene is not None:
            scene.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", choices=("control_calibration", "paired_evaluation"), required=True)
    parser.add_argument("--seed", type=int, default=2026103104)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--record-first", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--protocol")
    report = run(parser.parse_args())
    print(json.dumps({"pairs": report["pairs"], "success": {b: r["success"] for b, r in report["overall"].items()}}, indent=2))
    return 0
