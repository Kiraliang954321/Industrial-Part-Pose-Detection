"""Capture arbitrary held directions, measure from RGB, compare with configured targets."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment
from yolo_conveyor.annular.data import physical_label
from yolo_conveyor.simulation.panda_control import Stage
from .scene import DatumScene,PoseController,VIEWS,SETUP
from .assets import PARTS
from .targets import load,matrix
from .perception import Predictor,measure,compare,POLICY
from .geometry import errors,describe
from .evaluation import summarize
from .provenance import snapshot,sources
from .recording import Recorder,draw

CASES=('aligned','axis_error','spin_error','boundary','one_camera_dark','plain')


def run(args):
    config=load(args.targets);count=len(config['targets'])*len(CASES);protocol=None
    if args.mode=='evaluate':
        from .protocol import validate
        protocol=validate(args.protocol,args)
    elif not 161000<=args.seed<162000-count-100:raise ValueError('calibration seed range')
    out=new_directory(args.output,[args.assets,args.weights,args.targets]);os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    rows=[];scene=None
    try:
        predictor=Predictor(args.weights);predictor.predict({v:np.zeros((640,640,3),np.uint8) for v in VIEWS})
        manifest=dict(scope='A2_CUSTOM_PROXY_SIMULATION',real_formal_gate='closed',mode=args.mode,arguments=vars(args),
                      sources=snapshot(out),weights_sha256=sha256(Path(args.weights)),targets_sha256=sha256(Path(args.targets)),
                      targets=config,policy=POLICY,setup=SETUP,environment=environment(),asset_sha256=sha256(Path(args.assets)/'asset-manifest.json'))
        write_json(out/'manifest.json',manifest);recorded={int(x) for x in args.record_indices.split(',') if x.strip()}
        for k,kind in enumerate(PARTS):
            for feature in (True,False):
                scene=DatumScene(args.assets,kind,feature);controller=PoseController(scene)
                (out/f'{kind}-{feature}-scene.xml').write_text(scene.xml,encoding='utf-8')
                for i in range(count):
                    case=CASES[i%len(CASES)]
                    if feature!=(case!='plain'):continue
                    index=k*count+i;seed=args.seed+k*100+i;rng=np.random.default_rng(seed);target=config['targets'][i//len(CASES)]
                    tilt=target['inclination_deg'];az=target['azimuth_deg'];spin=target['spin_deg']
                    if case=='axis_error':az+=180;tilt=max(tilt,15)
                    if case=='spin_error':spin+=45
                    if case=='boundary':spin+=12
                    directory=out/f'{index:03d}_{kind}_{case}_{seed}';directory.mkdir();start=time.perf_counter()
                    scene.initialize_pose(rng,spin);video=Recorder(scene,directory/'physics.mp4',f'{kind} | {case}') if index in recorded else None
                    illegal=set()
                    def observer(stage):
                        illegal.update(scene.illegal_contacts())
                        if video:video.sample(stage)
                    try:
                        controller.execute(controller.pick(),observer);controller.execute(controller.present(tilt,az),observer)
                        images={v:np.clip(scene.capture(v).astype(float)*rng.uniform(.9,1.1),0,255).astype(np.uint8) for v in VIEWS}
                        if case=='one_camera_dark':images[VIEWS[0]][:]=0
                        cameras={v:scene.camera_info(v) for v in VIEWS};predictor.sync();tick=time.perf_counter();predictions=predictor.predict(images);predictor.sync();latency=1000*(time.perf_counter()-tick)
                        target_r=matrix(target);measured=None
                        try:measured=measure(images,predictions,cameras,kind);decision=compare(measured,target_r,config)
                        except ValueError as exc:decision=dict(decision='UNCERTAIN',reason=str(exc),error=None)
                        # These reads are downstream scoring only, never estimator or decision inputs.
                        actual=scene.data.xmat[scene.body_id].reshape(3,3).copy();truth=physical_label(scene);actual_error=errors(actual,target_r)
                        gt=('INDETERMINATE' if not feature else 'CONFORMING' if truth['held'] and actual_error['axis_deg']<=config['axis_tolerance_deg'] and actual_error['spin_deg']<=config['spin_tolerance_deg'] else 'NONCONFORMING')
                        row=dict(index=index,seed=seed,kind=kind,case=case,target=target,gt=gt,actual_angles=describe(actual),actual_target_error=actual_error,
                                 measurement=measured,decision=decision,measurement_error=errors(measured['rotation'],actual) if measured and feature else None,
                                 inference_ms=latency,wall_s=time.perf_counter()-start,illegal_contacts=sorted(illegal),held=truth['held'])
                        for v in VIEWS:
                            cv2.imwrite(str(directory/f'{v}.png'),images[v]);cv2.imwrite(str(directory/f'{v}_pose.jpg'),draw(images[v],predictions[v],cameras[v],measured))
                        write_json(directory/'observation.json',dict(predictions=predictions,cameras=cameras,input_hashes={v:sha256(directory/f'{v}.png') for v in VIEWS}))
                        write_json(directory/'result.json',row);rows.append(row)
                        if video:
                            video.panel=draw(images[VIEWS[0]],predictions[VIEWS[0]],cameras[VIEWS[0]],measured)
                            video.lines=[f"Target tilt/azimuth/spin: {tilt if case=='aligned' else target['inclination_deg']:.1f} / {target['azimuth_deg']:.1f} / {target['spin_deg']:.1f} deg | {decision['decision']}"]
                            if measured:video.lines.append('Measured: '+str({n:round(v,1) if v is not None else None for n,v in measured['angles'].items()}))
                            controller.execute([Stage('hold',scene.data.qpos[scene.arm_qpos].copy(),0,1.)],observer)
                        print(json.dumps(dict(index=index,kind=kind,case=case,decision=decision['decision'],reason=decision['reason'],error=row['measurement_error'])),flush=True)
                        write_json(out/'progress.json',dict(status='running',completed=len(rows),requested=2*count),replace=True)
                    finally:
                        if video:video.close()
                scene.close();scene=None
        if sources()!=manifest['sources'] or sha256(Path(args.weights))!=manifest['weights_sha256']:raise ValueError('source/checkpoint drift')
        rows.sort(key=lambda r:r['index']);report=dict(status='completed',mode=args.mode,scope=manifest['scope'],real_formal_gate='closed',overall=summarize(rows),
                    by_part={p:summarize([r for r in rows if r['kind']==p]) for p in PARTS},
                    by_case={p+'/'+c:summarize([r for r in rows if r['kind']==p and r['case']==c]) for p in PARTS for c in CASES})
        write_json(out/'results.json',rows);write_json(out/'summary.json',report);write_json(out/'progress.json',dict(status='completed',episodes=len(rows)),replace=True)
        write_json(out/'artifact-hashes.json',{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()});return report
    except BaseException as exc:
        write_json(out/'progress.json',dict(status='failed',completed=len(rows),error=str(exc)),replace=True);raise
    finally:
        if scene is not None:scene.close()


def main():
    p=argparse.ArgumentParser()
    for name in ('assets','weights','targets','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('calibration','evaluate'),default='calibration');p.add_argument('--protocol');p.add_argument('--seed',type=int,default=161000)
    p.add_argument('--record-indices',default='');print(json.dumps(run(p.parse_args())['overall'],indent=2));return 0
