"""A4 bounded active inspection; image decisions are isolated from GT scoring."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment
from yolo_conveyor.pose_cell.scene import CellScene,VIEWS,SETUP
from yolo_conveyor.pose_cell.faces import PARTS
from yolo_conveyor.pose_cell.geometry import rz,error
from yolo_conveyor.pose_cell.predictor import Predictor
from yolo_conveyor.pose_cell.metrology import measure,LIMITS
from yolo_conveyor.pose_cell.runtime import observe,hold,POLICY as EXECUTION,CASES as A3_CASES,ANGLES as A3_ANGLES,TARGETS
from .control import ActiveController
from .policy import POLICY,observation_gate,grip_transform,agreement
from .evaluation import PhysicalScore,truth,summarize
from .preprocessing import roi_inputs
from .recording import Recorder
from .provenance import snapshot,sources
from .training import BASELINE_SHA

CASES=(*A3_CASES,'empty');ANGLES={**A3_ANGLES,'empty':90}


def baseline_measure(folder,predictor,kind,gt):
    started=time.perf_counter()
    evidence=json.loads((folder/'evidence.json').read_text(encoding='utf-8'));images={v:cv2.imread(str(folder/f'{v}.png')) for v in VIEWS}
    pred=predictor.predict(images);images,pred,cameras=roi_inputs(images,pred,evidence['cameras'])
    try:
        m=measure(images,pred,cameras,kind,2.);return dict(measurement=m,error=error(m['rotation'],gt['rotation']),reason=None,wall_s=time.perf_counter()-started)
    except (ValueError,np.linalg.LinAlgError) as exc:return dict(measurement=None,error=None,reason=str(exc),wall_s=time.perf_counter()-started)


def episode(s,c,predictor,baseline,directory,index,seed,case,target_spin,record,record_observations=False):
    rng=np.random.default_rng(seed);target=rz(target_spin);spin=target_spin+float(rng.uniform(12,25));feature=case!='plain';empty=case=='empty';flipped=case=='reverse_grasp'
    wall=time.perf_counter();s.initialize_pose(rng,spin,flipped,empty);video=Recorder(s,directory/'physics.mp4',f'{s.kind} | {case}') if record else None
    score=PhysicalScore(s);trace=[];finger_reference=None;observations=[];attempts=[];corrections=0;moves=0;observation_sim=0.
    inserted=False;prealigned=False;status='not_started';failure=None;anchor=None;relative_reference=None;recovered=False;baseline_initial=None
    trigger='initial';observation_motion_wall=0.
    def observer(stage):
        score.observe()
        if video:video.sample(stage)
        fingers=s.data.qpos[s.finger_qpos]
        if not np.isfinite(fingers).all():raise ValueError('invalid_gripper_feedback')
        if s.illegal_contacts():raise ValueError('robot_fixed_environment_collision')
        if finger_reference is not None and np.max(np.abs(fingers-finger_reference))>EXECUTION['grip_width_change_m']:raise ValueError('grip_changed')
        if int(round(s.data.time*1000))%20==0:trace.append(dict(time=float(s.data.time),stage=stage,fingers=fingers.tolist()))
    try:
        try:
            c.execute(c.pick(),observer);finger_reference=s.data.qpos[s.finger_qpos].copy()
            if np.min(finger_reference)<c.close_width+.0004:raise ValueError('no_secure_grip')
            c.execute(c.present(ANGLES[case]),observer)
            while len(observations)<1+POLICY['max_observation_moves']+EXECUTION['max_corrections']:
                j=len(observations);folder=directory/f'observation_{j}'
                observed_at=time.perf_counter();obs,panel=observe(s,predictor,folder,j,target,feature and not empty,case=='blind')
                obs.update(trigger=trigger,wall_s=time.perf_counter()-observed_at)
                evidence=json.loads((folder/'evidence.json').read_text(encoding='utf-8'));images={v:cv2.imread(str(folder/f'{v}.png')) for v in VIEWS}
                gate=observation_gate(images,evidence['predictions'],s.kind);tool=s.tool_transform().copy();measured=obs['measurement'];check=None
                if measured is not None:
                    relative=grip_transform(measured,tool);check=agreement(relative_reference,relative)
                    if not check['consistent']:measured=None;obs['policy_reason']='inconsistent_grip_transform'
                    elif relative_reference is None:relative_reference=relative
                obs.update(tool_fk=tool.tolist(),observation_gate=gate,consistency=check,accepted=measured is not None,
                           observation_moves_before=moves,corrections_before=corrections)
                observations.append(obs);write_json(folder/'decision.json',{k:obs.get(k) for k in ('accepted','policy_reason','observation_gate','consistency','tool_fk')})
                if j==0 and baseline is not None:baseline_initial=baseline_measure(folder,baseline,s.kind,obs['gt'])
                if video:video.snapshot=panel
                if measured is None:
                    status='UNCERTAIN';failure=obs.get('policy_reason') or obs['reason']
                    if not gate['allowed']:failure=gate['reason'];break
                    if video is None and record_observations:
                        video=Recorder(s,directory/'physics.mp4',f'{s.kind} | {case} | observation recovery')
                        video.snapshot=panel
                    if anchor is None:anchor=tool.copy()
                    moved=False
                    while len(attempts)<POLICY['max_observation_moves']:
                        slot=len(attempts);attempt=dict(slot=slot,executed=False,angle_deg=POLICY['observation_angles_deg'][slot]);attempts.append(attempt)
                        try:stages=c.observe_pose(anchor,slot)
                        except ValueError as exc:attempt['reason']=str(exc);continue
                        # Observation motion has its own budget and event type. No GT or
                        # construction feature/empty flag authorizes this branch.
                        start=float(s.data.time);motion_start=time.perf_counter();attempt['executed']=True;moves+=1
                        if video:video.message=f'Observe again: pose {slot+1}/{POLICY["max_observation_moves"]}; no insertion permission'
                        try:c.execute(stages,observer)
                        finally:
                            observation_sim+=float(s.data.time)-start;observation_motion_wall+=time.perf_counter()-motion_start
                        moved=True;break
                    if moved:trigger='observation';continue
                    if attempts and not any(a['executed'] for a in attempts):failure='observation_unreachable'
                    break
                if j and not observations[0]['measurement'] and corrections==0:recovered=True
                anchor=None;target_error=error(measured['rotation'],target)
                if video:video.message=f'Observed target error: axis {target_error["axis_deg"]:.1f} deg | rotation {target_error["rotation_deg"]:.1f} deg'
                if target_error['rotation_deg']<=EXECUTION['rotation_gate_deg'] and target_error['axis_deg']<=EXECUTION['axis_gate_deg']:
                    if float(s.data.time)-obs['captured_sim_s']>EXECUTION['frame_lifetime_s']:raise ValueError('stale_verification')
                    g=obs['gt']['target_error'];prealigned=bool(feature and not empty and obs['gt']['held'] and g['rotation_deg']<=EXECUTION['gt_rotation_deg'] and g['axis_deg']<=EXECUTION['gt_axis_deg'])
                    stages=c.insert(measured,target);inserted=True;c.execute(stages,observer);status='EXECUTED';failure=None;break
                if corrections>=EXECUTION['max_corrections']:status='CORRECTION_LIMIT';failure='correction_budget';break
                stages=c.correction(measured,target);corrections+=1;c.execute(stages,observer);trigger='correction'
            else:status='UNCERTAIN';failure='observation_budget'
        except ValueError as exc:status='ABORT';failure=str(exc)
        finger_reference=None;hold(s,c,.5,lambda _:score.observe())
        final=truth(s,target);physical=score.result(inserted,failure if inserted or status=='ABORT' else None)
        aligned=bool(feature and not empty and final['held'] and final['target_error']['rotation_deg']<=EXECUTION['gt_rotation_deg'] and final['target_error']['axis_deg']<=EXECUTION['gt_axis_deg'])
        success=bool(feature and not empty and inserted and physical['success'] and aligned)
        if status=='EXECUTED':status='SUCCESS' if success else 'PHYSICAL_CHECK_FAILED'
        row=dict(index=index,seed=seed,kind=s.kind,case=case,feature=feature,empty=empty,initial_flip=flipped,presentation_deg=ANGLES[case],target_spin_deg=target_spin,
            observations=observations,baseline_initial=baseline_initial,observation_attempts=attempts,observation_moves=moves,observation_sim_s=observation_sim,
            observation_motion_wall_s=observation_motion_wall,observation_extra_wall_s=observation_motion_wall+sum(o['wall_s'] for o in observations if o['trigger']=='observation'),
            recovered_initial=recovered,recovered_verification=any(o['accepted'] and o['corrections_before']>0 and any(not prev['accepted'] and prev['corrections_before']==o['corrections_before'] for prev in observations[:j]) for j,o in enumerate(observations)),
            corrections=corrections,insert_attempted=inserted,preinsert_gt_aligned=prealigned,final_gt=final,physical=physical,aligned_gt=aligned,
            success=success,status=status,failure=failure,wall_s=time.perf_counter()-wall)
        write_json(directory/'result.json',row);write_json(directory/'trace.json',trace);cv2.imwrite(str(directory/'final.png'),s.capture('assembly_view'))
        if video:video.message=status+(' | '+failure if failure else '');video.next=0.;video.sample('final')
        return row
    finally:
        if video:video.close()


def run(args):
    protocol=None
    if args.mode=='evaluate':
        from .protocol import validate
        protocol=validate(args.protocol,args)
    elif not 241000<=args.seed<241873:raise ValueError('A4 calibration seed reservation')
    if sha256(Path(args.baseline_weights))!=BASELINE_SHA:raise ValueError('A3 baseline hash mismatch')
    out=new_directory(args.output,[args.assets,args.weights,args.baseline_weights]);os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config');rows=[];scene=None
    count=len(CASES)*len(TARGETS)
    try:
        predictor=Predictor(args.weights);baseline=Predictor(args.baseline_weights)
        for p in (predictor,baseline):p.predict({v:np.zeros((640,640,3),np.uint8) for v in VIEWS})
        manifest=dict(scope='A4_ACTIVE_INSPECTION_SIMULATION',mode=args.mode,real_formal_gate='closed',arguments=vars(args),sources=snapshot(out),setup=SETUP,
            limits=LIMITS,policy=POLICY,execution=EXECUTION,weights_sha256=sha256(Path(args.weights)),baseline_sha256=BASELINE_SHA,
            asset_sha256=sha256(Path(args.assets)/'asset-manifest.json'),environment=environment(),cases=list(CASES),targets=TARGETS,
            protocol_sha256=sha256(Path(args.protocol)) if protocol else None)
        write_json(out/'manifest.json',manifest);recorded={int(x) for x in args.record_indices.split(',') if x.strip()}
        for k,kind in enumerate(PARTS):
            for feature in (True,False):
                scene=CellScene(args.assets,kind,feature);controller=ActiveController(scene);(out/f'{kind}-{feature}-scene.xml').write_text(scene.xml,encoding='utf-8')
                for i in range(count):
                    case=CASES[i%len(CASES)]
                    if feature!=(case!='plain'):continue
                    index=k*count+i;seed=args.seed+k*100+i;folder=out/f'{index:03d}_{kind}_{case}_{seed}';folder.mkdir()
                    r=episode(scene,controller,predictor,baseline,folder,index,seed,case,TARGETS[i//len(CASES)],index in recorded,args.record_observations);rows.append(r)
                    print(json.dumps(dict(index=index,kind=kind,case=case,status=r['status'],looks=r['observation_moves'],corrections=r['corrections'],failure=r['failure'])),flush=True)
                    write_json(out/'progress.json',dict(status='running',completed=len(rows),requested=2*count),replace=True)
                scene.close();scene=None
        if sources()!=manifest['sources'] or sha256(Path(args.weights))!=manifest['weights_sha256']:raise ValueError('runtime source/weight drift')
        if protocol:
            validate(args.protocol,args)
            if sha256(Path(args.protocol))!=manifest['protocol_sha256']:raise ValueError('protocol drift')
        rows.sort(key=lambda r:r['index']);report=dict(status='completed',mode=args.mode,scope=manifest['scope'],real_formal_gate='closed',overall=summarize(rows),
            by_part={k:summarize([r for r in rows if r['kind']==k]) for k in PARTS},by_case={c:summarize([r for r in rows if r['case']==c]) for c in CASES})
        write_json(out/'results.json',rows);write_json(out/'summary.json',report);write_json(out/'progress.json',dict(status='completed',episodes=len(rows)),replace=True)
        write_json(out/'artifact-hashes.json',{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()});return report
    except BaseException as exc:
        write_json(out/'progress.json',dict(status='failed',completed=len(rows),error=str(exc)),replace=True);raise
    finally:
        if scene is not None:scene.close()


def main():
    p=argparse.ArgumentParser()
    for n in ('assets','weights','baseline-weights','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--mode',choices=('calibration','evaluate'),default='calibration');p.add_argument('--protocol');p.add_argument('--seed',type=int,default=241000)
    p.add_argument('--record-indices',default='');p.add_argument('--record-observations',action='store_true')
    print(json.dumps(run(p.parse_args())['overall'],indent=2));return 0
