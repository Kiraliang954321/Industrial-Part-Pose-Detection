"""Freeze L1 development configuration after calibration; real gates remain closed."""

import argparse
import json
from pathlib import Path
from yolo_conveyor.orientation.common import unlinked,sha256,write_json,utc_now
from yolo_conveyor.simulation.m4_protocol import CONTROL_FILES as M4_FILES
from yolo_conveyor.simulation.indexed_scene import LINE_SETUP
from yolo_conveyor.simulation.closed_loop import WEIGHT_SHA256
from .events import LineRecipe
from .provenance import line_sources
from .runner import CASES,LIGHTS,SCOPE

CALIBRATED_FILES = M4_FILES + ["src/yolo_conveyor/simulation/indexed_scene.py"]
CALIBRATED_FILES += [f"src/yolo_conveyor/line/{name}.py" for name in ("events","state","perception","control","metrics","recording","runner")]


def freeze(calibration,output,seed,episodes):
    root=unlinked(calibration)
    report=json.loads((root/"report.json").read_text(encoding="utf-8"))
    manifest=json.loads((root/"run-manifest.json").read_text(encoding="utf-8"))
    overall=report["overall"]
    if (report.get("status")!="completed" or report.get("role")!="calibration" or
        report.get("manifest_sha256")!=sha256(root/"run-manifest.json") or
        overall["multi_frame"]["events"]<len(CASES) or overall["missing_events"] or overall["duplicate_events"] or
        overall["automatic_action_success"]["x"]!=overall["automatic_action_success"]["n"] or
        overall["safe_halt"]["x"]!=overall["safe_halt"]["n"] or
        set(overall["multi_frame"]["outcome_counts"])!={"PASS","FAIL_ORIENTATION","UNCERTAIN"} or
        overall["multi_frame"]["wrong_accept"]["x"]!=0):
        raise ValueError("calibration must demonstrate event integrity, automatic actions and safe halt")
    current=line_sources()
    if any(current[p]!=manifest["sources"].get(p) for p in CALIBRATED_FILES):
        raise ValueError("validated L1 implementation changed since calibration")
    if manifest["recipe"]!=LineRecipe().record() or manifest["setup"]!=LINE_SETUP:
        raise ValueError("recipe or physical fixture changed since calibration")
    if not 6<=episodes<=100 or not 0<=seed<=2**32-episodes:raise ValueError("invalid evaluation assignment")
    seeds=list(range(seed,seed+episodes))
    if set(seeds)&set(manifest["seeds"]):raise ValueError("evaluation overlaps calibration")
    result={"schema_version":1,"status":"frozen_line_development_only","scope":SCOPE,"real_formal_gate":"closed",
            "created_at":utc_now(),"calibration_report":str(root/"report.json"),"calibration_report_sha256":sha256(root/"report.json"),
            "calibration_seeds":manifest["seeds"],"seeds":seeds,"recipe":LineRecipe().record(),"setup":LINE_SETUP,
            "cases":list(CASES),"lights":list(LIGHTS),"weight_sha256":WEIGHT_SHA256,
            "assets_sha256":manifest["assets_sha256"],"sources":current,
            "rules":"No tuning on these outcomes; a change needs new calibration/protocol/evaluation seeds. Independent indexing episodes, not continuous throughput."}
    write_json(unlinked(output),result);return result


def validate_protocol(path,seed,episodes,assets,recipe):
    if not path:raise ValueError("evaluation requires --protocol")
    path=unlinked(path);p=json.loads(path.read_text(encoding="utf-8"))
    if (p.get("schema_version")!=1 or p.get("status")!="frozen_line_development_only" or
        p.get("scope")!=SCOPE or p.get("real_formal_gate")!="closed" or
        p.get("seeds")!=list(range(seed,seed+episodes)) or p.get("recipe")!=recipe.record() or
        p.get("setup")!=LINE_SETUP or p.get("cases")!=list(CASES) or p.get("lights")!=list(LIGHTS) or
        p.get("weight_sha256")!=WEIGHT_SHA256 or p.get("assets_sha256")!=sha256(unlinked(assets)/"asset-manifest.json") or
        p.get("sources")!=line_sources() or set(p["seeds"])&set(p["calibration_seeds"])):
        raise ValueError("L1 protocol/configuration/source/asset assignment drift")
    return {"path":str(path),"sha256":sha256(path),"status":p["status"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,default=21000);parser.add_argument("--episodes",type=int,default=24)
    args=parser.parse_args();result=freeze(args.calibration,args.output,args.seed,args.episodes)
    print(json.dumps({"status":result["status"],"events":len(result["seeds"])}));return 0
