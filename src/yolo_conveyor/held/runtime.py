"""Held RGB inspection to encoder-controlled pre-insertion, with separate truth scoring."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment,utc_now
from yolo_conveyor.orientation.pose_adapter import PosePredictor
from yolo_conveyor.orientation.evaluate import draw_predictions
from yolo_conveyor.simulation.panda_control import Stage
from .scene import HeldScene,SETUP
from .control import HeldController
from .data import fixture_case,VIEWS
from .gate import inspect,Permit,POLICY
from .evaluation import truth,PhysicalScore,summarize
from .provenance import snapshot_sources,held_sources
from .video import Recorder

CASES=('normal','reversed','short','long','tilted','empty','blackout','expired_permit')


def run(args):
    if not 1<=args.episodes<=64 or not 0<=args.record_first<=args.episodes:raise ValueError('invalid episode budget')
    protocol=None
    if args.mode=='evaluate':
        from .protocol import validate
        protocol=validate(args.protocol,args)
    elif args.seed<61000 or args.seed+args.episodes>62000:
        raise ValueError('calibration seeds must be in reserved 61000:62000 range')
    out=new_directory(args.output,[args.assets,args.weights]);scene=None;results=[]
    os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    try:
        cv2.setNumThreads(2)
        predictor=PosePredictor(args.weights,confidence=.25,point_confidence=.5)
        predictor.predict(np.zeros((640,640,3),np.uint8));predictor.synchronize()
        scene=HeldScene(args.assets);controller=HeldController(scene)
        (out/'scene.xml').write_text(scene.xml,encoding='utf-8')
        if protocol and sha256(out/'scene.xml')!=protocol['scene_sha256']:raise ValueError('frozen scene drift')
        manifest=dict(mode=args.mode,scope='held_simulation_development',real_formal_gate='closed',arguments=vars(args),
                      created_at=utc_now(),environment=environment(),sources=snapshot_sources(out),setup=SETUP,policy=POLICY,
                      weights_sha256=sha256(Path(args.weights)),scene_sha256=sha256(out/'scene.xml'),
                      asset_manifest_sha256=sha256(Path(args.assets)/'asset-manifest.json'),
                      protocol_sha256=sha256(Path(args.protocol)) if args.protocol else None,
                      cases=list(CASES),control_inputs='RGB, fixed calibration/fixture, robot FK and finger/wrist feedback only')
        write_json(out/'run-manifest.json',manifest)
        for index in range(args.episodes):
            seed=args.seed+index;case=CASES[index%len(CASES)];rng=np.random.default_rng(seed)
            spec=fixture_case(index%8 if index%8<6 else 0,rng);spec['case']=case
            ep=out/f'{index:03d}_{case}_{seed}';ep.mkdir();wall=time.perf_counter()
            scene.initialize(rng,spec);start_sim=float(scene.data.time)
            score=PhysicalScore(scene);video=Recorder(scene,ep/'physics.mp4',f'VISION | {case} | seed {seed}') if index<args.record_first else None
            trace=[];baseline=None;recorded_baseline=None;finger_baseline=None;abort=None;permit_denied=None;attempted=False
            def observer(stage):
                nonlocal baseline,finger_baseline
                score.observe()
                if video:video.sample(stage)
                wrench=scene.wrist_wrench();fingers=scene.data.qpos[scene.finger_qpos]
                if not np.isfinite(wrench).all() or not np.isfinite(fingers).all():raise ValueError('invalid_sensor_feedback')
                if int(round(scene.data.time*1000))%10==0:
                    trace.append(dict(time=float(scene.data.time),stage=stage,wrench=wrench.tolist(),fingers=fingers.tolist()))
                if finger_baseline is not None and np.max(np.abs(fingers-finger_baseline))>POLICY['finger_change_m']:
                    raise ValueError('finger_width_changed')
                if baseline is not None:
                    delta=wrench-baseline
                    if np.linalg.norm(delta[:3])>POLICY['max_force_delta_n'] or np.linalg.norm(delta[3:])>POLICY['max_torque_delta_nm']:
                        raise ValueError('wrist_delta_limit')
            try:
                controller.execute(controller.plan_pick_and_inspect(),observer)
                # Per-episode light/exposure sampling, independent of the scored part pose.
                light=np.array([rng.uniform(-.25,.15),rng.uniform(-.2,.2),rng.uniform(.4,.6)])
                scene.model.light_pos[0]=light;direction=np.array([-.1,0,.25])-light
                scene.model.light_dir[0]=direction/np.linalg.norm(direction)
                intensity=float(rng.uniform(.4,1.1));scene.model.light_diffuse[0]=intensity
                gain=float(rng.uniform(.85,1.15))
                images={n:np.clip(scene.capture(n).astype(float)*gain,0,255).astype(np.uint8) for n in VIEWS}
                if case=='blackout':images['inspect_b'][:]=0
                cameras={n:scene.camera_info(n) for n in VIEWS};tool=scene.tool_transform().copy()
                fingers=scene.data.qpos[scene.finger_qpos].copy();capture_time=float(scene.data.time)
                predictor.synchronize();tick=time.perf_counter()
                predictions={n:predictor.predict(images[n]) for n in VIEWS};predictor.synchronize()
                inference_ms=(time.perf_counter()-tick)*1000
                decision=inspect(images,predictions,cameras,tool,fingers)
                # Independent ground truth is recorded only after the visual decision.
                gt=truth(scene);error=None
                if decision['estimate']:
                    estimate=decision['estimate'];actual=gt['estimate']
                    error=dict(tip_mm=float(1000*np.linalg.norm(np.subtract(estimate['tip_world_m'],actual['tip_world_m']))),
                               axis_deg=float(np.degrees(np.arccos(np.clip(np.dot(estimate['axis_world'],actual['axis_world']),-1,1)))))
                for name in VIEWS:
                    cv2.imwrite(str(ep/f'{name}.png'),images[name])
                    cv2.imwrite(str(ep/f'{name}_prediction.jpg'),draw_predictions(images[name],predictions[name],decision['decision']+' | '+decision['reason']))
                write_json(ep/'inspection.json',dict(predictions=predictions,cameras=cameras,tool=tool.tolist(),fingers=fingers.tolist(),
                           captured_sim_s=capture_time,decision=decision,illumination=dict(gain=gain,intensity=intensity,light_position=light.tolist()),
                           input_hashes={n:sha256(ep/f'{n}.png') for n in VIEWS}))
                if video:video.evidence=dict(images=images,predictions=predictions)
                if decision['decision']=='PASS':
                    grasp_id=f'h1-{seed}';epoch=1;tool_id='panda_custom_thin_pad_v1'
                    permit=Permit.issue(decision,grasp_id,epoch,tool_id,capture_time,tool,fingers)
                    if case=='expired_permit':
                        controller.execute([Stage('hold',scene.data.qpos[scene.arm_qpos].copy(),0,POLICY['permit_lifetime_s']+.1)],observer)
                    try:
                        estimate=permit.consume(grasp_id,epoch,tool_id,float(scene.data.time),scene.tool_transform(),
                                                scene.data.qpos[scene.finger_qpos],float(scene.data.ctrl[scene.gripper_actuator]))
                    except ValueError as exc:permit_denied=str(exc)
                    else:
                        finger_baseline=fingers
                        try:
                            controller.execute(controller.plan_prealign(estimate,tool),observer)
                            baseline=scene.wrist_wrench().copy()
                            recorded_baseline=baseline.copy()
                            insertion=controller.plan_insert(estimate,tool);attempted=True
                            controller.execute(insertion,observer)
                        except ValueError as exc:
                            abort=str(exc);permit.invalidate(abort)
                            # Hold current robot joints, keep grasp closed; no object-pose reset.
                            baseline=None;finger_baseline=None
                            controller.execute([Stage('hold',scene.data.qpos[scene.arm_qpos].copy(),0,.5)],observer)
                if not attempted:
                    controller.execute([Stage('hold',scene.data.qpos[scene.arm_qpos].copy(),0,1.)],observer)
                result=score.result(attempted,abort)
                row=dict(index=index,seed=seed,case=case,fixture=spec,decision=decision,inspection_gt=gt,prediction_error=error,
                         permit_denied=permit_denied,result=result,inference_ms=inference_ms,
                         wall_s=time.perf_counter()-wall,simulated_s=float(scene.data.time)-start_sim,
                         force_baseline=recorded_baseline.tolist() if recorded_baseline is not None else None)
                write_json(ep/'result.json',row);write_json(ep/'wrench-trace.json',trace)
                cv2.imwrite(str(ep/'final_overview.png'),scene.capture('overview'))
                if video:
                    cv2.imwrite(str(ep/'final_detail.png'),video.assembly_view())
                    video.sample('PREINSERT_SUCCESS' if result['preinsert_success'] else decision['decision'])
                results.append(row)
                write_json(out/'progress.json',dict(status='running',completed=len(results),requested=args.episodes),replace=True)
                print(json.dumps(dict(index=index,case=case,decision=decision['decision'],reason=decision['reason'],
                                     permit_denied=permit_denied,success=result['preinsert_success'],abort=abort)),flush=True)
            finally:
                if video:video.close()
        if held_sources()!=manifest['sources']:raise ValueError('sources changed during run')
        if sha256(Path(args.weights))!=manifest['weights_sha256']:raise ValueError('weights changed during run')
        report=dict(status='completed',mode=args.mode,scope='held_simulation_development',real_formal_gate='closed',
                    overall=summarize(results),by_case={c:summarize([r for r in results if r['case']==c]) for c in CASES},
                    limitations=['Independent biased scenario episodes; not production prevalence or throughput',
                                 'No real CAD, real sensor validation, thread engagement or tightening',
                                 'Finger encoder does not exclude axial slip with unchanged width; no continuous visual servo'])
        write_json(out/'results.json',results);write_json(out/'summary.json',report)
        write_json(out/'progress.json',dict(status='completed',episodes=len(results)),replace=True)
        write_json(out/'artifact-hashes.json',{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
        return report
    except BaseException as exc:
        write_json(out/'progress.json',dict(status='failed',completed=len(results),error=str(exc)),replace=True);raise
    finally:
        if scene:scene.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('assets','weights','output'):p.add_argument('--'+flag,required=True)
    p.add_argument('--mode',choices=('calibration','evaluate'),default='calibration');p.add_argument('--protocol')
    p.add_argument('--seed',type=int,default=61000);p.add_argument('--episodes',type=int,default=24)
    p.add_argument('--record-first',type=int,default=0)
    print(json.dumps(run(p.parse_args())['overall'],indent=2));return 0
