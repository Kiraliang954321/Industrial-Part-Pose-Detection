"""Initial visual transfer gate, then independent fresh-image insertion permit."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment,utc_now
from yolo_conveyor.orientation.pose_adapter import PosePredictor
from yolo_conveyor.held.control import HeldController
from yolo_conveyor.held.data import fixture_case
from yolo_conveyor.held.gate import inspect as h1_gate
from yolo_conveyor.held.evaluation import PhysicalScore
from yolo_conveyor.held.video import Recorder
from yolo_conveyor.simulation.panda_control import Stage
from .scene import RecheckScene,RECHECK_VIEWS,RECHECK_CENTER,SETUP
from .calibration import Calibrator,WEIGHT_SHA
from .inspection import POLICY,first_decision,recheck,MotionPermit
from .perception import capture,save
from .development import illuminate
from .evaluation import capture_truth,audit_recheck,summarize
from .faults import FAULTS,disturb
from .provenance import snapshot,sources

CASES=('normal','long','yaw_jitter','reversed','short','empty','initial_blackout',
       'axial_slip','strong_disturbance','recheck_blackout','stale_recheck','expired_insert')
SPEC_INDEX={'long':3,'yaw_jitter':4,'reversed':1,'short':2,'empty':5}


class Monitor:
    def __init__(self,scene,video):
        self.scene=scene;self.video=video;self.score=PhysicalScore(scene)
        self.fingers=None;self.wrench=None;self.trace=[]

    def __call__(self,stage):
        s=self.scene;self.score.observe()
        if self.video:self.video.sample(stage)
        fingers=s.data.qpos[s.finger_qpos];wrench=s.wrist_wrench()
        if not np.isfinite(fingers).all() or not np.isfinite(wrench).all():raise ValueError('invalid_sensor_feedback')
        if int(round(s.data.time*1000))%10==0:
            self.trace.append(dict(time=float(s.data.time),stage=stage,fingers=fingers.tolist(),wrench=wrench.tolist()))
        if self.fingers is not None and np.max(np.abs(fingers-self.fingers))>POLICY['finger_change_m']:
            raise ValueError('finger_width_changed')
        if self.wrench is not None:
            delta=wrench-self.wrench
            if np.linalg.norm(delta[:3])>POLICY['max_force_delta_n'] or np.linalg.norm(delta[3:])>POLICY['max_torque_delta_nm']:
                raise ValueError('wrist_delta_limit')


def hold(scene,controller,seconds,monitor):
    controller.execute([Stage('hold',scene.data.qpos[scene.arm_qpos].copy(),0,seconds)],monitor)


def consume(permit,purpose,scene,grasp_id,evidence_id):
    return permit.consume(purpose,grasp_id,1,'panda_custom_thin_pad_v1',float(scene.data.time),
                          scene.tool_transform(),scene.data.qpos[scene.finger_qpos],
                          float(scene.data.ctrl[scene.gripper_actuator]),evidence_id)


def episode(scene,controller,predictor,calibrator,directory,index,seed,record):
    case=CASES[index%len(CASES)];rng=np.random.default_rng(seed)
    spec=fixture_case(SPEC_INDEX.get(case,0),rng);wall=time.perf_counter()
    scene.initialize(rng,spec);start=float(scene.data.time)
    video=Recorder(scene,directory/'physics.mp4',f'H2 | {case} | {seed}') if record else None
    monitor=Monitor(scene,video);post_gt=None;post=None;permit_denied=None;abort=None;attempted=False
    second=None;recheck_ms=None;grasp_id=f'h2-{seed}';disturbance=None;force_baseline=None
    try:
        controller.execute(controller.plan_pick_and_inspect(),monitor)
        appearance=illuminate(scene,rng,[-.1,0,.25])
        first=capture(scene,predictor,calibrator,('inspect_a','inspect_b'),appearance['gain'],
                      'inspect_b' if case=='initial_blackout' else None)
        raw=h1_gate(first['images'],first['raw'],first['cameras'],first['tool'],first['fingers'])
        initial=first_decision(first)
        # GT is measured after the decision and never returned to the motion branch.
        gt=capture_truth(scene);save(directory/'initial',first,initial)
        if video:video.evidence=dict(images=first['images'],predictions=first['corrected'])
        if initial['decision']=='PASS':
            permit=MotionPermit.issue('transfer',grasp_id,1,'panda_custom_thin_pad_v1',initial,first,float(scene.data.time))
            estimate=consume(permit,'transfer',scene,grasp_id,first['evidence_id'])
            monitor.fingers=np.array(first['fingers'])
            try:
                controller.execute(controller.plan_prealign(estimate,first['tool']),monitor)
                disturbance=FAULTS.get(case)
                disturb(scene,controller,case,monitor)
                appearance=illuminate(scene,rng,RECHECK_CENTER)
                second=capture(scene,predictor,calibrator,RECHECK_VIEWS,appearance['gain'],
                               'recheck_b' if case=='recheck_blackout' else None)
                if case=='stale_recheck':hold(scene,controller,1.,monitor)
                post=recheck(second,estimate,first['tool'],float(scene.data.time))
                post_gt=audit_recheck(gt,capture_truth(scene));recheck_ms=second['inference_ms']
                save(directory/'recheck',second,post)
                if video:
                    video.evidence=dict(images={f'inspect_{s}':second['images'][f'recheck_{s}'] for s in ('a','b')},
                                        predictions={f'inspect_{s}':second['corrected'][f'recheck_{s}'] for s in ('a','b')})
                if post['decision']=='PASS':
                    insert=MotionPermit.issue('insert',grasp_id,1,'panda_custom_thin_pad_v1',post,second,float(scene.data.time))
                    if case=='expired_insert':hold(scene,controller,POLICY['insert_permit_lifetime_s']+.1,monitor)
                    try:new_estimate=consume(insert,'insert',scene,grasp_id,second['evidence_id'])
                    except ValueError as exc:permit_denied=str(exc)
                    else:
                        trajectory=controller.plan_insert(new_estimate,second['tool'])
                        force_baseline=scene.wrist_wrench().copy();monitor.wrench=force_baseline
                        attempted=True;controller.execute(trajectory,monitor)
            except ValueError as exc:
                abort=str(exc);monitor.wrench=None;monitor.fingers=None
                hold(scene,controller,.6,monitor)
                if post_gt is None:post_gt=audit_recheck(gt,capture_truth(scene))
        monitor.fingers=None;monitor.wrench=None
        if not attempted:hold(scene,controller,.7,monitor)
        result=monitor.score.result(attempted,abort)
        row=dict(index=index,seed=seed,case=case,fixture=spec,initial=initial,raw_h1=raw,initial_gt=gt,
                 recheck=post,post_gt=post_gt,result=result,permit_denied=permit_denied,abort=abort,
                 disturbance=disturbance,first_fingers=first['fingers'],
                 post_fingers=second['fingers'] if second else scene.data.qpos[scene.finger_qpos].tolist(),
                 force_baseline=force_baseline.tolist() if force_baseline is not None else None,
                 initial_inference_ms=first['inference_ms'],recheck_inference_ms=recheck_ms,
                 wall_s=time.perf_counter()-wall,simulated_s=float(scene.data.time)-start)
        write_json(directory/'result.json',row);write_json(directory/'trace.json',monitor.trace)
        cv2.imwrite(str(directory/'final_overview.png'),scene.capture('overview'))
        if video:
            cv2.imwrite(str(directory/'final_detail.png'),video.assembly_view());video.next_time=0
            video.sample('PREINSERT_OK' if result['preinsert_success'] else 'HOLD_FOR_REVIEW')
        return row
    finally:
        scene.data.xfrc_applied[scene.body_id,:]=0
        if video:video.close()


def run(args):
    if not 1<=args.episodes<=96:raise ValueError('episode budget')
    protocol=None
    if args.mode=='evaluate':
        from .protocol import validate
        protocol=validate(args.protocol,args)
    elif not 83000<=args.seed<84000-args.episodes:raise ValueError('calibration seed reservation')
    if sha256(Path(args.weights))!=WEIGHT_SHA:raise ValueError('wrong H1 model')
    out=new_directory(args.output,[args.assets,args.weights,args.calibrator]);scene=None;rows=[]
    os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    try:
        cv2.setNumThreads(2);predictor=PosePredictor(args.weights);calibrator=Calibrator.load(args.calibrator,args.weights)
        predictor.predict(np.zeros((640,640,3),np.uint8));predictor.synchronize()
        scene=RecheckScene(args.assets);controller=HeldController(scene)
        (out/'scene.xml').write_text(scene.xml,encoding='utf-8')
        manifest=dict(scope='H2_SIMULATION_DEVELOPMENT',real_formal_gate='closed',mode=args.mode,arguments=vars(args),
                      sources=snapshot(out),weights_sha256=WEIGHT_SHA,calibrator_sha256=sha256(Path(args.calibrator)),
                      asset_sha256=sha256(Path(args.assets)/'asset-manifest.json'),scene_sha256=sha256(out/'scene.xml'),
                      environment=environment(),setup=SETUP,policy=POLICY,faults=FAULTS,cases=list(CASES),created_at=utc_now())
        if protocol and protocol['scene_sha256']!=manifest['scene_sha256']:raise ValueError('frozen scene changed')
        write_json(out/'manifest.json',manifest)
        recorded={int(x) for x in args.record_indices.split(',') if x.strip()}
        if any(x<0 or x>=args.episodes for x in recorded):raise ValueError('record index outside run')
        for index in range(args.episodes):
            seed=args.seed+index;directory=out/f'{index:03d}_{CASES[index%len(CASES)]}_{seed}';directory.mkdir()
            row=episode(scene,controller,predictor,calibrator,directory,index,seed,index in recorded);rows.append(row)
            write_json(out/'progress.json',dict(status='running',completed=len(rows),requested=args.episodes),replace=True)
            print(json.dumps(dict(index=index,case=row['case'],initial=row['initial']['decision'],
                                  recheck=row['recheck']['reason'] if row['recheck'] else None,abort=row['abort'],
                                  permit_denied=row['permit_denied'],success=row['result']['preinsert_success'])),flush=True)
        if (sources()!=manifest['sources'] or sha256(Path(args.calibrator))!=manifest['calibrator_sha256']
                or sha256(Path(args.weights))!=WEIGHT_SHA):
            raise ValueError('source/calibrator changed during run')
        report=dict(status='completed',scope='H2_SIMULATION_DEVELOPMENT',mode=args.mode,real_formal_gate='closed',
                    overall=summarize(rows),by_case={c:summarize([r for r in rows if r['case']==c]) for c in CASES})
        write_json(out/'results.json',rows);write_json(out/'summary.json',report)
        write_json(out/'progress.json',dict(status='completed',episodes=len(rows)),replace=True)
        write_json(out/'artifact-hashes.json',{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
        return report
    except BaseException as exc:
        write_json(out/'progress.json',dict(status='failed',completed=len(rows),error=str(exc)),replace=True);raise
    finally:
        if scene is not None:scene.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','weights','calibrator','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('calibration','evaluate'),default='calibration');p.add_argument('--protocol')
    p.add_argument('--seed',type=int,default=83000);p.add_argument('--episodes',type=int,default=36)
    p.add_argument('--record-indices',default='')
    print(json.dumps(run(p.parse_args())['overall'],indent=2));return 0
