"""Freeze/check a development-only paired simulation protocol against calibration evidence."""

import argparse
import json
from pathlib import Path

from yolo_conveyor.orientation.common import sha256, unlinked, utc_now, write_json, source_hashes, dataset_file
from .panda_scene import ADAPTATION
from .grasp_evaluation import CRITERIA

CONTROL_FILES = [f"src/yolo_conveyor/simulation/{name}.py" for name in
                 ("screw_scene", "panda_scene", "panda_control", "grasp_geometry", "grasp_evaluation", "robot_assets")]
CONTROL_FILES += ["src/yolo_conveyor/orientation/pose_adapter.py", "src/yolo_conveyor/orientation/geometry.py"]


def freeze(calibration, output, seed, episodes):
    root = unlinked(calibration)
    report = json.loads(dataset_file(root, "report.json").read_text(encoding="utf-8"))
    manifest = json.loads(dataset_file(root, "run-manifest.json").read_text(encoding="utf-8"))
    if (report.get("status") != "completed" or report.get("run_role") != "control_calibration" or
            report["pairs"] < 4 or report["overall"]["gt"]["success"]["x"] != report["pairs"] or
            report["manifest_sha256"] != sha256(root / "run-manifest.json")):
        raise ValueError("completed, successful GT calibration evidence is required")
    if not 1 <= episodes <= 100 or not 0 <= seed <= 2**32 - episodes:
        raise ValueError("invalid seed/episode budget")
    current = source_hashes()
    if any(current[p] != manifest["source_hashes"].get(p) for p in CONTROL_FILES):
        raise ValueError("control/perception implementation changed since calibration")
    if manifest["adaptation"] != ADAPTATION or manifest["criteria"] != CRITERIA:
        raise ValueError("calibrated configuration no longer matches")
    seeds = list(range(seed, seed + episodes))
    if set(seeds) & set(manifest["scene_seeds"]):
        raise ValueError("evaluation seeds overlap calibration")
    protocol = {"schema_version": 1, "status": "frozen_development_only", "scope": manifest["scope"],
                "real_formal_gate": "closed", "created_at": utc_now(),
                "calibration_report": str(root / "report.json"), "calibration_report_sha256": sha256(root / "report.json"),
                "calibration_manifest_sha256": sha256(root / "run-manifest.json"),
                "calibration_seeds": manifest["scene_seeds"], "scene_seeds": seeds,
                "light_conditions": manifest["light_conditions"], "adaptation": ADAPTATION, "criteria": CRITERIA,
                "planning_priors": manifest["planning_priors"], "weight_sha256": manifest["weight_sha256"],
                "asset_manifest_sha256": manifest["asset_manifest_sha256"],
                "controller_source_hashes": {p: current[p] for p in CONTROL_FILES},
                "rule": "Do not tune on these outcomes; a controller change requires new calibration/protocol and new evaluation seeds"}
    write_json(unlinked(output), protocol)
    return protocol


def validate(protocol_path, seed, episodes, weight_hash, asset_hash, light_conditions):
    path = unlinked(protocol_path)
    p = json.loads(path.read_text(encoding="utf-8"))
    current = source_hashes()
    if (p.get("schema_version") != 1 or p.get("status") != "frozen_development_only" or
            p.get("scope") != "simulation_closed_loop_development" or p.get("real_formal_gate") != "closed" or
            p.get("scene_seeds") != list(range(seed, seed + episodes)) or
            p.get("weight_sha256") != weight_hash or p.get("asset_manifest_sha256") != asset_hash or
            p.get("adaptation") != ADAPTATION or p.get("criteria") != CRITERIA or
            p.get("light_conditions") != list(light_conditions) or
            set(p.get("controller_source_hashes", {})) != set(CONTROL_FILES) or
            any(p["controller_source_hashes"][f] != current[f] for f in CONTROL_FILES) or
            set(p["scene_seeds"]) & set(p["calibration_seeds"])):
        raise ValueError("M4 protocol, sources, assets, or seed assignment changed")
    return {"path": str(path), "sha256": sha256(path), "status": p["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026103104)
    parser.add_argument("--episodes", type=int, default=24)
    args = parser.parse_args()
    result = freeze(args.calibration, args.output, args.seed, args.episodes)
    print(json.dumps({"status": result["status"], "episodes": len(result["scene_seeds"])}))
    return 0
