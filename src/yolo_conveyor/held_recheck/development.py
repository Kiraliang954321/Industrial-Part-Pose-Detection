"""Fit on Train only, validate separately, and collect real rendered prealign views."""
import argparse
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,dataset_file,environment
from yolo_conveyor.orientation.pose_adapter import PosePredictor
from yolo_conveyor.orientation.evaluation import distribution
from yolo_conveyor.held.data import load_dataset,audit_dataset,fixture_case,annotated_view
from yolo_conveyor.held.control import HeldController
from yolo_conveyor.held.geometry import stereo_estimate
from .scene import RecheckScene,RECHECK_VIEWS,RECHECK_CENTER,SETUP
from .calibration import Calibrator,fit,WEIGHT_SHA
from .provenance import snapshot


def illuminate(scene,rng,center):
    light=np.array([rng.uniform(-.25,.15),rng.uniform(-.2,.2),rng.uniform(.4,.6)])
    scene.model.light_pos[0]=light
    direction=np.asarray(center)-light
    scene.model.light_dir[0]=direction/np.linalg.norm(direction)
    intensity=float(rng.uniform(.4,1.1));scene.model.light_diffuse[0]=intensity
    return dict(gain=float(rng.uniform(.85,1.15)),intensity=intensity,light_position=light.tolist())


def score(records,parameters):
    calibrator=Calibrator(parameters);raw_errors=[];corrected_errors=[];missing=0
    for row in records:
        if not row['objects']:continue
        if len(row['raw'])!=1:missing+=1;continue
        try:corrected=calibrator.apply(row['view'],row['raw'])[0]
        except ValueError:missing+=1;continue
        expected=np.array(row['objects'][0]['keypoints_xy'])
        raw_errors.extend(np.linalg.norm(np.array(row['raw'][0]['keypoints_xy'])-expected,axis=1).tolist())
        corrected_errors.extend(np.linalg.norm(np.array(corrected['keypoints_xy'])-expected,axis=1).tolist())
    return dict(images=len(records),missing_positive_images=missing,raw_endpoint_px=distribution(raw_errors),
                corrected_endpoint_px=distribution(corrected_errors))


def develop(args):
    if sha256(Path(args.weights))!=WEIGHT_SHA:raise ValueError('H1 selected checkpoint required')
    if not 24<=args.train_episodes<=60 or not 12<=args.val_episodes<=24:raise ValueError('development budget')
    root=Path(args.dataset).resolve();audit=audit_dataset(root)
    if audit['errors']:raise ValueError('H1 dataset audit failed')
    out=new_directory(args.output,[root,args.weights,args.assets]);start=time.perf_counter();scene=None
    os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    try:
        manifest=dict(sources=snapshot(out),weights_sha256=WEIGHT_SHA,source_dataset_sha256=sha256(root/'dataset.json'),
                      arguments=vars(args),environment=environment(),setup=SETUP)
        write_json(out/'manifest.json',manifest)
        predictor=PosePredictor(args.weights);_,rows=load_dataset(root);initial={split:[] for split in ('train','val')}
        for row in rows:
            if row['variant']!=0:continue
            image=cv2.imread(str(dataset_file(root,row['image'])))
            initial[row['split']].append(dict(episode_id=row['episode_id'],view=row['view'],objects=row['objects'],
                                              image=str(root/row['image']),image_sha256=row['image_sha256'],raw=predictor.predict(image)))
        parameters=fit(initial['train']);write_json(out/'initial-parameters.json',parameters)
        write_json(out/'initial-predictions.json',initial)
        print(json.dumps(dict(stage='initial',train=score(initial['train'],parameters),val=score(initial['val'],parameters))),flush=True)
        calibrator=Calibrator(parameters);scene=RecheckScene(args.assets);controller=HeldController(scene)
        (out/'scene.xml').write_text(scene.xml,encoding='utf-8');post={split:[] for split in ('train','val')}
        for split,count,seed_start in (('train',args.train_episodes,81000),('val',args.val_episodes,81500)):
            for index in range(count):
                seed=seed_start+index;rng=np.random.default_rng(seed);spec=fixture_case((0,3,4)[index%3],rng)
                scene.initialize(rng,spec);controller.execute(controller.plan_pick_and_inspect(),lambda _:None)
                appearance=illuminate(scene,rng,[-.1,0,.25])
                images={name:np.clip(scene.capture(name).astype(float)*appearance['gain'],0,255).astype(np.uint8) for name in ('inspect_a','inspect_b')}
                predictions={name:calibrator.apply(name,predictor.predict(images[name])) for name in images}
                estimate=stereo_estimate(predictions,{name:scene.camera_info(name) for name in images})
                # Development collection uses image-derived targets, never object truth.
                controller.execute(controller.plan_prealign(estimate,scene.tool_transform()),lambda _:None)
                episode=f'h2_{split}_{seed}';directory=out/'images'/episode;directory.mkdir(parents=True)
                appearance=illuminate(scene,rng,RECHECK_CENTER)
                for view in RECHECK_VIEWS:
                    image,objects,camera=annotated_view(scene,view,appearance['gain'])
                    path=directory/f'{view}.png'
                    if not cv2.imwrite(str(path),image):raise RuntimeError('image write')
                    post[split].append(dict(episode_id=episode,seed=seed,view=view,objects=objects,camera=camera,
                                            image=str(path),image_sha256=sha256(path),raw=predictor.predict(image),
                                            tool=scene.tool_transform().tolist(),appearance=appearance))
                print(json.dumps(dict(stage='prealign',split=split,completed=index+1,requested=count)),flush=True)
        parameters.update(fit(post['train']))
        result=dict(scope='H2_DEVELOPMENT_RESIDUAL_CALIBRATION',weights_sha256=WEIGHT_SHA,parameters=parameters,
                    initial=dict(train=score(initial['train'],parameters),val=score(initial['val'],parameters)),
                    prealign=dict(train=score(post['train'],parameters),val=score(post['val'],parameters)),
                    rule='Offsets fit on Train only; Val never enters median fitting; runtime receives no GT',
                    scene_sha256=sha256(out/'scene.xml'),seconds=time.perf_counter()-start)
        write_json(out/'prealign-records.json',post);write_json(out/'calibrator.json',result)
        write_json(out/'artifact-hashes.json',{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
        return result
    finally:
        if scene is not None:scene.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','dataset','weights','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--train-episodes',type=int,default=30);p.add_argument('--val-episodes',type=int,default=12)
    result=develop(p.parse_args());print(json.dumps({k:v for k,v in result.items() if k!='parameters'},indent=2));return 0
