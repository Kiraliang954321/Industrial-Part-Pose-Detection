"""Simple single-camera held orientation gate followed by a known recipe motion."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment
from yolo_conveyor.simulation.panda_control import Stage
from .assets import PARTS
from .scene import AssemblyScene,SETUP
from .control import AssemblyController
from .data import physical_label
from .perception import Detector,decide,POLICY
from .evaluation import PhysicalScore,summarize
from .video import Recorder,draw
from .provenance import snapshot,sources

CASES=('front','back','tilted','empty','blackout')


def episode(scene,controller,detector,directory,index,seed,case,record):
    rng=np.random.default_rng(seed);face='back' if case=='back' or (case=='tilted' and rng.random()<.5) else 'front'
    tilt=float(rng.choice([-1,1])*rng.uniform(30,45)) if case=='tilted' else 0.
    wall=time.perf_counter();scene.initialize(rng,face,case=='empty');start=float(scene.data.time)
    score=PhysicalScore(scene);video=Recorder(scene,directory/'physics.mp4',scene.kind+' | '+case) if record else None
    baseline=None;trace=[];abort=None;attempted=False
    def observer(stage):
        score.observe()
        if video:video.sample(stage)
        fingers=scene.data.qpos[scene.finger_qpos]
        if not np.isfinite(fingers).all():raise ValueError('invalid_gripper_feedback')
        if int(round(scene.data.time*1000))%20==0:trace.append(dict(time=float(scene.data.time),stage=stage,fingers=fingers.tolist()))
        if baseline is not None and np.max(np.abs(fingers-baseline))>POLICY['finger_change_m']:raise ValueError('grip_changed')
    try:
        controller.execute(controller.pick(tilt),observer)
        image=np.clip(scene.capture('inspect_rgb').astype(float)*rng.uniform(.85,1.15),0,255).astype(np.uint8)
        if case=='blackout':image[:]=0
        fingers=scene.data.qpos[scene.finger_qpos].copy();tool=scene.tool_transform().copy();captured=float(scene.data.time)
        detector.sync();tick=time.perf_counter();predictions=detector.predict(image);detector.sync();latency=(time.perf_counter()-tick)*1000
        decision=decide(image,predictions,scene.kind,fingers)
        gt=physical_label(scene)  # Independent score only; not passed to the gate/controller.
        cv2.imwrite(str(directory/'input.png'),image);preview=draw(image,predictions,decision['decision']+' | '+decision['reason'])
        cv2.imwrite(str(directory/'prediction.jpg'),preview)
        write_json(directory/'inspection.json',dict(predictions=predictions,decision=decision,tool=tool.tolist(),fingers=fingers.tolist(),
                   captured_sim_s=captured,input_sha256=sha256(directory/'input.png')))
        if video:video.evidence=preview
        if decision['decision']=='PASS':
            try:
                trajectory=controller.preposition()
                if float(scene.data.time)-captured>POLICY['evidence_age_s'] or np.linalg.norm(scene.tool_transform()[:3,3]-tool[:3,3])>.002:
                    raise ValueError('stale_inspection')
                if np.max(np.abs(scene.data.qpos[scene.finger_qpos]-fingers))>POLICY['finger_change_m']:raise ValueError('grip_changed')
                baseline=fingers;attempted=True;controller.execute(trajectory,observer)
            except ValueError as exc:
                abort=str(exc);baseline=None
                controller.execute([Stage('hold',scene.data.qpos[scene.arm_qpos].copy(),0,.5)],observer)
        else:controller.execute([Stage('hold',scene.data.qpos[scene.arm_qpos].copy(),0,.8)],observer)
        result=score.result(attempted,abort)
        row=dict(index=index,seed=seed,kind=scene.kind,case=case,decision=decision,initial_gt=gt,result=result,
                 inference_ms=latency,wall_s=time.perf_counter()-wall,simulated_s=float(scene.data.time)-start)
        write_json(directory/'result.json',row);write_json(directory/'trace.json',trace)
        cv2.imwrite(str(directory/'final.png'),scene.capture('assembly_view'))
        if video:video.next=0;video.sample('PREPOSITION_OK' if result['success'] else 'HOLD_FOR_REVIEW')
        return row
    finally:
        if video:video.close()


def run(args):
    if not 10<=args.episodes_per_part<=30 or args.episodes_per_part%5:raise ValueError('episode budget')
    protocol=None
    if args.mode=='evaluate':
        from .protocol import validate
        protocol=validate(args.protocol,args)
    elif not 121000<=args.seed<122000-args.episodes_per_part:raise ValueError('calibration seed reservation')
    out=new_directory(args.output,[args.assets,args.weights]);os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    rows=[];scene=None
    try:
        cv2.setNumThreads(2);detector=Detector(args.weights);detector.predict(np.zeros((640,640,3),np.uint8))
        manifest=dict(scope='A1_SIMULATION_DEVELOPMENT',real_formal_gate='closed',mode=args.mode,arguments=vars(args),
                      sources=snapshot(out),weights_sha256=sha256(Path(args.weights)),asset_sha256=sha256(Path(args.assets)/'asset-manifest.json'),
                      environment=environment(),setup=SETUP,policy=POLICY,parts=PARTS,cases=list(CASES))
        write_json(out/'manifest.json',manifest)
        record={int(v) for v in args.record_indices.split(',') if v.strip()}
        for k,kind in enumerate(PARTS):
            scene=AssemblyScene(args.assets,kind);controller=AssemblyController(scene);(out/f'{kind}-scene.xml').write_text(scene.xml,encoding='utf-8')
            if protocol and sha256(out/f'{kind}-scene.xml')!=protocol['scene_hashes'][kind]:raise ValueError('scene drift')
            for i in range(args.episodes_per_part):
                index=k*args.episodes_per_part+i;seed=args.seed+k*100+i;case=CASES[i%5]
                directory=out/f'{index:03d}_{kind}_{case}_{seed}';directory.mkdir()
                row=episode(scene,controller,detector,directory,index,seed,case,index in record);rows.append(row)
                print(json.dumps(dict(index=index,kind=kind,case=case,decision=row['decision'],success=row['result']['success'],abort=row['result']['abort'])),flush=True)
                write_json(out/'progress.json',dict(status='running',completed=len(rows),requested=2*args.episodes_per_part),replace=True)
            scene.close();scene=None
        if sources()!=manifest['sources'] or sha256(Path(args.weights))!=manifest['weights_sha256']:raise ValueError('runtime inputs changed')
        report=dict(status='completed',mode=args.mode,scope=manifest['scope'],real_formal_gate='closed',overall=summarize(rows),
                    by_part={kind:summarize([r for r in rows if r['kind']==kind]) for kind in PARTS},
                    by_condition={kind+'/'+case:summarize([r for r in rows if r['kind']==kind and r['case']==case]) for kind in PARTS for case in CASES})
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
    for flag in ('assets','weights','output'):p.add_argument('--'+flag,required=True)
    p.add_argument('--mode',choices=('calibration','evaluate'),default='calibration');p.add_argument('--protocol')
    p.add_argument('--seed',type=int,default=121000);p.add_argument('--episodes-per-part',type=int,default=20);p.add_argument('--record-indices',default='')
    print(json.dumps(run(p.parse_args())['overall'],indent=2));return 0
