"""Physical single-pallet inspection episodes with model-driven event/action routing."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import time
import cv2
import numpy as np

from yolo_conveyor.orientation.common import new_directory,unlinked,sha256,write_json,environment,utc_now
from yolo_conveyor.orientation.geometry import directed_error
from yolo_conveyor.simulation.indexed_scene import IndexedScene,LINE_SETUP,GATE_Q,OUTFEED_Q
from yolo_conveyor.simulation.grasp_geometry import GraspObservation
from yolo_conveyor.simulation.grasp_evaluation import PhysicalEvaluator
from yolo_conveyor.simulation.closed_loop import WEIGHT_SHA256
from .events import LineRecipe,InspectionEvent,single_frame_decision
from .perception import frame_evidence
from .state import CellSequencer
from .control import IndexedPandaController
from .recording import LineWitness
from .metrics import summarize
from .provenance import line_sources

CASES = ("clear_pass","axis_fail","reversed_fail","startup_dropout","camera_blackout","boundary")
LIGHTS = ("nominal","low_light","reflection","background_shift")
SCOPE = "indexed_line_simulation_development"


def scenario(index, rng):
    case=CASES[index%len(CASES)]
    base={"clear_pass":0,"axis_fail":90,"reversed_fail":180,"startup_dropout":0,"camera_blackout":0,"boundary":9}[case]
    jitter=.25 if case=="boundary" else 2.
    return {"case":case,"light":LIGHTS[(index//len(CASES))%len(LIGHTS)],
            "spawn_heading_deg":float(base+rng.uniform(-jitter,jitter))}


def ground_truth(truth, recipe):
    x,y,z=truth["center_world_m"]
    valid=abs(x)<=.065 and abs(y)<=.065 and 0<=z<=.02 and truth["projected_axis_length_m"]>=.033
    error=directed_error(truth["heading_world_deg"],recipe.target_heading_deg)
    label=("CONFORMING" if error<=recipe.tolerance_deg else "NONCONFORMING") if valid else "INDETERMINATE"
    return {"label":label,"heading_error_deg":error,"truth":truth,
            "reference":"Independent simulator state at the end of the shared observation window"}


def run(args):
    if not 1<=args.episodes<=100 or not 0<=args.record_first<=args.episodes or not 0<=args.seed<=2**32-args.episodes:
        raise ValueError("invalid line run budget")
    weights=unlinked(args.weights)
    if not weights.is_file() or sha256(weights)!=WEIGHT_SHA256:raise ValueError("the frozen M3 checkpoint is required")
    recipe=LineRecipe()
    if args.role=="evaluation":
        from .protocol import validate_protocol
        protocol=validate_protocol(args.protocol,args.seed,args.episodes,args.assets,recipe)
    else:protocol=None
    out=new_directory(args.output,[args.assets,weights]);(out/"yolo-config").mkdir()
    os.environ["YOLO_CONFIG_DIR"]=str(out/"yolo-config")
    scene=None; started=time.perf_counter()
    try:
        from yolo_conveyor.orientation.pose_adapter import PosePredictor
        from yolo_conveyor.orientation.evaluate import draw_predictions
        import torch
        torch.set_num_threads(4);cv2.setNumThreads(2)
        scene=IndexedScene(args.assets);controller=IndexedPandaController(scene)
        predictor=PosePredictor(weights,device=args.device)
        predictor.predict(np.zeros((640,640,3),np.uint8));predictor.synchronize()
        manifest={"scope":SCOPE,"real_formal_gate":"closed","role":args.role,"created_at":utc_now(),
                  "arguments":vars(args),"recipe":recipe.record(),"setup":LINE_SETUP,"cases":list(CASES),"lights":list(LIGHTS),
                  "seeds":list(range(args.seed,args.seed+args.episodes)),"sources":line_sources(),"environment":environment(),
                  "protocol":protocol,"weight_sha256":WEIGHT_SHA256,
                  "assets_sha256":sha256(Path(args.assets)/"asset-manifest.json"),
                  "event_identity_source":"Simulated PLC feed job; single scheduled part per episode, not visual tracking",
                  "fault_injection":"Startup: first 2 input frames black. Blackout: all input frames black. Detector/evidence logic sees pixels only.",
                  "throughput_scope":"Independent episodes with no unloading/return/recovery; no continuous line throughput"}
        write_json(out/"run-manifest.json",manifest);(out/"scene.xml").write_text(scene.xml,encoding="utf-8")
        source_root=Path(__file__).resolve().parents[3]
        for relative,digest in manifest["sources"].items():
            dest=out/"source"/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source_root/relative,dest)
            if sha256(dest)!=digest:raise ValueError("source changed while snapshotting")
        rows=[]
        for index in range(args.episodes):
            seed=args.seed+index;rng=np.random.default_rng(seed);spec=scenario(index,rng)
            episode=out/f"job-{seed}";episode.mkdir();job_id=f"l1_{args.role}_{seed}"
            camera,params=scene.initialize(rng,spec["light"],math.radians(spec["spawn_heading_deg"]))
            cell=CellSequencer(job_id);cell.start(float(scene.data.time))
            with (episode/"cell-trace.jsonl").open("x",encoding="utf-8",newline="\n") as trace:
                witness=LineWitness(scene,cell,trace,episode/"cell.mp4" if index<args.record_first else None)
                try:
                    scene.index_to(GATE_Q,witness.sample,"index_in")
                    cell.arrived(float(scene.data.time));scene.advance(.2,witness.sample,"settling")
                    cell.inspect(float(scene.data.time));inspection_start=float(scene.data.time)
                    event=InspectionEvent(job_id,inspection_start,recipe);evidence_rows=[];first=None
                    wall_start=time.perf_counter()
                    for frame_id in range(recipe.frame_budget):
                        if frame_id:scene.advance(.1,witness.sample,"observing")
                        capture_time=float(scene.data.time);raw=scene.capture("rgb")
                        injected=spec["case"]=="camera_blackout" or (spec["case"]=="startup_dropout" and frame_id<2)
                        image=np.zeros_like(raw) if injected else raw
                        predictions=predictor.predict(image);predictor.synchronize()
                        evidence=frame_evidence(frame_id,capture_time,image,predictions,camera)
                        event.observe(evidence)
                        if first is None:first=single_frame_decision(job_id,evidence,recipe)
                        for name,frame in (("raw",raw),("input",image)):
                            if not cv2.imwrite(str(episode/f"frame-{frame_id:02d}-{name}.png"),frame):raise RuntimeError("frame write failed")
                        evidence_rows.append({"evidence":asdict(evidence),"predictions":predictions,"black_frame_injected":injected,
                                              "input_sha256":sha256(episode/f"frame-{frame_id:02d}-input.png"),
                                              "raw_sha256":sha256(episode/f"frame-{frame_id:02d}-raw.png")})
                        witness.inspection_image=draw_predictions(image,predictions,f"inspection frame {frame_id+1}/8")
                    final_event=event.finish(float(scene.data.time))
                    if final_event is None:raise RuntimeError("inspection window failed to finalize")
                    decision_wall_ms=(time.perf_counter()-wall_start)*1000
                    gt=ground_truth(witness.truth(),recipe)  # Scoring-only; never passed into decision or control.
                    write_json(episode/"event.json",final_event)
                    write_json(episode/"frame-evidence.json",evidence_rows)
                    cell.decide(final_event,float(scene.data.time))
                    if final_event["outcome"]=="PASS":
                        scene.index_to(OUTFEED_Q,witness.sample,"pass_to_downstream")
                        truth=witness.truth();x,y,z=truth["center_world_m"]
                        success=bool(abs(x)<.065 and .14<y<.23 and 0<z<.02 and "pallet_surface" in truth["support_contacts"] and witness.illegal_steps==0)
                        action={"kind":"downstream_handoff_on_pallet","success":success,"complete_cyclic_line":False,"truth":truth}
                        cell.complete(float(scene.data.time),success)
                    elif final_event["outcome"]=="FAIL_ORIENTATION":
                        observation=GraspObservation(tuple(final_event["center_xy_world_m"]),math.radians(final_event["heading_world_deg"]),"multi_frame_vision")
                        try:stages,plan=controller.plan(observation)
                        except ValueError as exc:action={"kind":"robot_reject","success":False,"planning_error":str(exc)}
                        else:
                            with (episode/"robot-trace.jsonl").open("x",encoding="utf-8",newline="\n") as robot_trace:
                                physical=PhysicalEvaluator(scene,robot_trace)
                                def observe(stage):
                                    witness.sample(f"robot_{stage}");physical.sample(stage)
                                controller.execute(stages,observe);physical_result=physical.result()
                            action={"kind":"robot_reject","success":physical_result["success"],"planning":plan,"physical_result":physical_result}
                        cell.complete(float(scene.data.time),action["success"])
                    else:
                        robot_before=scene.data.qpos[scene.arm_qpos].copy();scene.advance(1.0,witness.sample,"halted_waiting_for_review")
                        truth=witness.truth()
                        safe=bool(abs(float(scene.data.qpos[scene.pallet_q])-GATE_Q)<.001 and
                                  np.max(np.abs(scene.data.qpos[scene.arm_qpos]-robot_before))<.001 and
                                  "pallet_surface" in truth["support_contacts"] and witness.illegal_steps==0)
                        action={"kind":"manual_review_required","success":False,"safe_hold":safe,"automatic_recovery":False,"truth":truth}
                    witness.sample("episode_endpoint")
                    row={"job_id":job_id,"seed":seed,"scenario":spec,"camera":camera,"generation_parameters":params,
                         "ground_truth":gt,"single_frame":first,"event":final_event,"action":action,
                         "terminal_state":cell.state.value,"transitions":cell.transitions,
                         "elapsed_sim_s":float(scene.data.time-witness.start),"decision_wall_ms":decision_wall_ms,
                         "maximum_object_step_m":witness.max_step_m,"maximum_penetration_m":witness.max_penetration_m,
                         "illegal_contact_steps":witness.illegal_steps,"video_frames":witness.video_frames}
                    write_json(episode/"result.json",row);rows.append(row)
                    if not cv2.imwrite(str(episode/"final.png"),scene.capture("overview")):raise RuntimeError("final image write failed")
                finally:witness.close()
            progress={"status":"running","completed":len(rows),"requested":args.episodes,"last_outcome":row["event"]["outcome"],"last_state":row["terminal_state"]}
            write_json(out/"progress.json",progress,replace=True);print(json.dumps(progress),flush=True)
        report={"status":"completed","scope":SCOPE,"real_formal_gate":"closed","role":args.role,
                "manifest_sha256":sha256(out/"run-manifest.json"),"overall":summarize(rows),
                "by_scenario":{case:summarize([r for r in rows if r["scenario"]["case"]==case]) for case in CASES if any(r["scenario"]["case"]==case for r in rows)},
                "wall_seconds":time.perf_counter()-started,"completed_at":utc_now()}
        write_json(out/"report.json",report);write_json(out/"progress.json",{"status":"completed","events":len(rows)},replace=True)
        return report
    except BaseException as exc:
        write_json(out/"progress.json",{"status":"failed","error":str(exc)},replace=True);raise
    finally:
        if scene is not None:scene.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",required=True);parser.add_argument("--weights",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--role",choices=("calibration","evaluation"),required=True);parser.add_argument("--protocol")
    parser.add_argument("--seed",type=int,default=21000);parser.add_argument("--episodes",type=int,default=24)
    parser.add_argument("--record-first",type=int,default=6);parser.add_argument("--device",default="0")
    report=run(parser.parse_args());print(json.dumps(report["overall"],indent=2));return 0
