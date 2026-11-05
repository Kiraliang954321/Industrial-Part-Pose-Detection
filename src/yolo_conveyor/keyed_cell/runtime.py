"""Vision-driven keyed pre-insertion and sensor-only contact-stop recovery."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment
from yolo_conveyor.pose_cell.geometry import rz,error
from yolo_conveyor.pose_cell.predictor import Predictor
from yolo_conveyor.pose_cell.runtime import POLICY as EXECUTION,ANGLES as A3_ANGLES,TARGETS
from yolo_conveyor.active_pose.runtime import CASES as A4_CASES
from yolo_conveyor.active_pose.policy import POLICY as OBSERVATION_POLICY
from .scene import KeyedScene,SETUP
from .control import KeyedController
from .guard import Monitor,ContactStop,GUARD
from .evaluation import PhysicalScore,truth,summarize,LIMITS
from .inspection import inspect
from .recording import Recorder
from .provenance import snapshot,sources

WEIGHTS_SHA='ae474306bca4ec34f50ab9cdb437ae7dc1cf0a62a9bc9b4e5d9bfb3a3d8713b1'
CASES=(*A4_CASES,'fixture_misindex30','fixture_misindex90')
ANGLES={**A3_ANGLES,'empty':90,'fixture_misindex30':0,'fixture_misindex90':0}
OFFSETS={'fixture_misindex30':30.,'fixture_misindex90':90.}


def episode(s,c,predictor,folder,index,seed,case,recipe_angle,record_video):
    rng=np.random.default_rng(seed);target=rz(recipe_angle);spin=recipe_angle+float(rng.uniform(12,25))
    feature=case!='plain';empty=case=='empty';wall=time.perf_counter();s.initialize_pose(rng,spin,case=='reverse_grasp',empty)
    video=Recorder(s,folder/'physics.mp4',f'{s.kind} | {case} | recipe {recipe_angle:g} deg') if record_video else None
    score=PhysicalScore(s);monitor=Monitor(s,score,video)
    record=dict(observations=[],observation_attempts=[],observation_moves=0,corrections=0)
    attempted=False;blocked=False;failure=None;status='not_started';recipe_ok=False;tare=None;precontact_gt=None
    try:
        try:
            c.execute(c.pick(),monitor);monitor.fingers=s.data.qpos[s.finger_qpos].copy()
            if np.min(monitor.fingers)<c.close_width+.0004:raise ValueError('no_secure_grip')
            c.execute(c.present(ANGLES[case]),monitor)
            measured,reason=inspect(s,c,predictor,folder,target,feature and not empty,case=='blind',monitor,record)
            if measured is None:status='UNCERTAIN';failure=reason
            else:
                gt=record['observations'][-1]['gt'];e=gt['target_error']
                recipe_ok=bool(feature and not empty and gt['held'] and e['rotation_deg']<=EXECUTION['gt_rotation_deg'] and e['axis_deg']<=EXECUTION['gt_axis_deg'])
                geometry=c.assembly_geometry(measured,target);write_json(folder/'visual_grip_geometry.json',geometry)
                if video:video.message='Visual recipe confirmed; approach known keyed fixture'
                c.execute(c.approach(geometry),monitor);tare=monitor.tare()
                stages=c.preinsert(geometry)
                # Independent truth is logged after the visual plan is fixed, never
                # used to authorize or veto the descent (including fault fixtures).
                precontact_gt=truth(s,target);attempted=True;score.insertion_active=True
                if video:video.message='Guarded descent: physical key/pin must enter its matching opening'
                try:
                    c.execute(stages,monitor);status='EXECUTED'
                except ContactStop:
                    blocked=True;monitor.disarm()
                    if video:video.message='CONTACT LIMIT: stop descent and retract while holding the part'
                    c.execute(c.stop_hold(),monitor);c.execute(c.retreat(),monitor);status='CONTACT_STOP_RETRACTED'
        except ValueError as exc:status='ABORT';failure=str(exc)
        monitor.disarm();monitor.fingers=None;c.execute(c.stop_hold(.3),lambda _:score.observe())
        final=truth(s,target);physical=score.result(attempted,'contact_stop' if blocked else failure)
        aligned=bool(feature and not empty and final['held'] and final['target_error']['rotation_deg']<=EXECUTION['gt_rotation_deg'] and final['target_error']['axis_deg']<=EXECUTION['gt_axis_deg'])
        success=bool(attempted and not blocked and physical['keyed_success'] and aligned and failure is None)
        if status=='EXECUTED':status='KEYED_SUCCESS' if success else 'PHYSICAL_CHECK_FAILED'
        if status=='CONTACT_STOP_RETRACTED' and not physical['retracted_and_held']:status='CONTACT_STOP_NOT_CLEARED'
        fault=OFFSETS.get(case,0.)
        contained=bool(fault and blocked and physical['retracted_and_held'] and physical['index_contact_seen']
            and physical['deepest_engagement_m']<LIMITS['min_index_overlap_m'] and not physical['wrong_angle_passage']
            and physical['max_index_penetration_m']<=LIMITS['max_index_penetration_m'] and failure is None)
        row=dict(index=index,seed=seed,kind=s.kind,case=case,feature=feature,empty=empty,recipe_angle_deg=recipe_angle,fixture_offset_deg=fault,
            actual_fixture_angle_deg=s.fixture_angle,**record,insert_attempted=attempted,preinsert_gt_recipe_aligned=recipe_ok,
            precontact_gt=precontact_gt,tare_wrench=tare,blocked=blocked,fault_contained=contained,wrench=monitor.report(),
            final_gt=final,physical=physical,success=success,status=status,failure=failure,wall_s=time.perf_counter()-wall)
        write_json(folder/'result.json',row);write_json(folder/'wrench-trace.json',monitor.trace);cv2.imwrite(str(folder/'final.png'),s.capture('assembly_view'))
        if video:video.message=status+(' | '+failure if failure else '');video.next=0.;video.sample('final')
        return row
    finally:
        if video:video.close()


def run(args):
    protocol=None
    if args.mode=='evaluate':
        from .protocol import validate
        protocol=validate(args.protocol,args)
    elif not 271000<=args.seed<271867:raise ValueError('A5 calibration seed reservation')
    if sha256(Path(args.weights))!=WEIGHTS_SHA:raise ValueError('recorded A4 visual model required')
    out=new_directory(args.output,[args.assets,args.weights]);os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config');rows=[];scene=None
    count=len(CASES)*len(TARGETS);recorded={int(x) for x in args.record_indices.split(',') if x.strip()}
    try:
        predictor=Predictor(args.weights)
        manifest=dict(scope='A5_KEYED_CLEARANCE_PREINSERTION_SIMULATION',real_formal_gate='closed',mode=args.mode,arguments=vars(args),sources=snapshot(out),
            setup=SETUP,guard=GUARD,score_limits=LIMITS,execution=EXECUTION,observation_policy=OBSERVATION_POLICY,weights_sha256=WEIGHTS_SHA,
            asset_sha256=sha256(Path(args.assets)/'asset-manifest.json'),environment=environment(),cases=list(CASES),targets=TARGETS,
            protocol_sha256=sha256(Path(args.protocol)) if protocol else None)
        write_json(out/'manifest.json',manifest)
        for k,kind in enumerate(('bearing','gear')):
            for i in range(count):
                index=k*count+i;case=CASES[i%len(CASES)];target=TARGETS[i//len(CASES)];seed=args.seed+100*k+i
                folder=out/f'{index:03d}_{kind}_{case}_{seed}';folder.mkdir()
                scene=KeyedScene(args.assets,kind,case!='plain',target+OFFSETS.get(case,0.));(folder/'scene.xml').write_text(scene.xml,encoding='utf-8')
                row=episode(scene,KeyedController(scene),predictor,folder,index,seed,case,target,index in recorded);rows.append(row)
                scene.close();scene=None
                print(json.dumps(dict(index=index,kind=kind,case=case,status=row['status'],blocked=row['blocked'],success=row['success'],fault_contained=row['fault_contained'],failure=row['failure'])),flush=True)
                write_json(out/'progress.json',dict(status='running',completed=len(rows),requested=2*count),replace=True)
        if sources()!=manifest['sources'] or sha256(Path(args.weights))!=WEIGHTS_SHA:raise ValueError('runtime drift')
        if protocol:
            validate(args.protocol,args)
            if sha256(Path(args.protocol))!=manifest['protocol_sha256']:raise ValueError('protocol drift')
        result=dict(status='completed',mode=args.mode,scope=manifest['scope'],real_formal_gate='closed',overall=summarize(rows),
            by_part={kind:summarize([r for r in rows if r['kind']==kind]) for kind in ('bearing','gear')},by_case={case:summarize([r for r in rows if r['case']==case]) for case in CASES})
        write_json(out/'results.json',rows);write_json(out/'summary.json',result);write_json(out/'progress.json',dict(status='completed',episodes=len(rows)),replace=True)
        write_json(out/'artifact-hashes.json',{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()});return result
    except BaseException as exc:
        write_json(out/'progress.json',dict(status='failed',completed=len(rows),error=str(exc)),replace=True);raise
    finally:
        if scene is not None:scene.close()


def main():
    p=argparse.ArgumentParser()
    for name in ('assets','weights','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('calibration','evaluate'),default='calibration');p.add_argument('--protocol');p.add_argument('--seed',type=int,default=271000)
    p.add_argument('--record-indices',default='');print(json.dumps(run(p.parse_args())['overall'],indent=2));return 0
