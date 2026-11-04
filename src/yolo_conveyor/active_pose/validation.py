"""Paired development-Val pose evaluation, with physical-grab grouping retained."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256
from yolo_conveyor.orientation.evaluation import rate,distribution
from yolo_conveyor.pose_cell.predictor import Predictor
from yolo_conveyor.pose_cell.metrology import measure
from yolo_conveyor.pose_cell.geometry import error
from yolo_conveyor.pose_cell.provenance import sources as baseline_sources
from .data import audit_dataset,read
from .provenance import snapshot,sources
from .training import BASELINE_SHA


def summarize(rows):
    positive=[r for r in rows if r['eligible']];usable=[r for r in positive if r['measurement']];negative=[r for r in rows if not r['eligible']]
    groups=defaultdict(list)
    for r in positive:groups[r['grab_id']].append(r)
    return dict(capture_coverage=rate(len(usable),len(positive)),
        grabs_with_any_measured_capture=rate(sum(any(r['measurement'] for r in group) for group in groups.values()),len(groups)),
        negative_with_pose=rate(sum(r['measurement'] is not None for r in negative),len(negative)),
        over_10deg=rate(sum(r['rotation_error']['rotation_deg']>10 for r in usable),len(usable)),
        rotation_deg=distribution([r['rotation_error']['rotation_deg'] for r in usable]),
        position_mm=distribution([r['position_error_mm'] for r in usable]),
        note='Multiple captures of one physical grab are correlated; capture counts are not independent grab counts.')


def run(args):
    root=Path(args.dataset).resolve()
    if audit_dataset(root)['errors']:raise ValueError('A4 dataset audit')
    if sha256(Path(args.baseline_weights))!=BASELINE_SHA:raise ValueError('A3 baseline mismatch')
    out=new_directory(args.output,[root,args.weights,args.baseline_weights]);os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    cache_root=Path(args.baseline_validation).resolve();cache_manifest=read(cache_root/'manifest.json')
    if (cache_manifest['sources']!=baseline_sources() or cache_manifest['weights_sha256']!=BASELINE_SHA
            or cache_manifest['dataset_sha256']!=read(root/'manifest.json')['source_dataset_sha256']):raise ValueError('baseline cache provenance mismatch')
    cached={r['episode']:r for r in read(cache_root/'results.json')}
    parent=Path(read(root/'manifest.json')['arguments']['source']).resolve()
    if sha256(parent/'dataset.json')!=cache_manifest['dataset_sha256']:raise ValueError('parent dataset drift')
    parent_rows={r['image']:r for r in read(parent/'records.json')}
    cache_hashes={str(cache_root/n):sha256(cache_root/n) for n in ('manifest.json','results.json','summary.json')}
    manifest=dict(scope='A4_DEVELOPMENT_VAL',weights_sha256=sha256(Path(args.weights)),baseline_sha256=BASELINE_SHA,
                  dataset_sha256=sha256(root/'dataset.json'),sources=snapshot(out),baseline_cache=cache_hashes)
    write_json(out/'manifest.json',manifest);models={'a3':Predictor(args.baseline_weights),'a4':Predictor(args.weights)}
    records=defaultdict(list)
    for r in read(root/'records.json'):records[r['episode']].append(r)
    results=[]
    for e in read(root/'episodes.json'):
        if e['split']!='val':continue
        views=records[e['id']];images={v['view']:cv2.imread(str(root/v['image'])) for v in views};cameras={v['view']:v['camera'] for v in views}
        for model_name,predictor in models.items():
            if model_name=='a3' and e['regime']=='a3_replay':
                if any(v['image_sha256']!=parent_rows[v['image']]['image_sha256'] or v['camera']!=parent_rows[v['image']]['camera'] for v in views):raise ValueError('cached baseline inputs differ')
                old=cached[e['id']]
                results.append({**old,'model':'a3','grab_id':e['grab_id'],'regime':e['regime'],'observation_pose':e['observation_pose'],'baseline_cached':True})
                continue
            pred=predictor.predict(images,preprocessed=True)
            row=dict(model=model_name,episode=e['id'],grab_id=e['grab_id'],kind=e['kind'],regime=e['regime'],observation_pose=e['observation_pose'],eligible=e['feature'] and not e['empty'],measurement=None,predictions=pred)
            try:
                m=measure(images,pred,cameras,e['kind'],2.);row.update(measurement=m,rotation_error=error(m['rotation'],e['rotation']),position_error_mm=float(np.linalg.norm(np.subtract(m['translation_m'],e['center']))*1000))
            except (ValueError,np.linalg.LinAlgError) as exc:row['reason']=str(exc)
            results.append(row)
        if len(results)%16==0:print(json.dumps(dict(captures=len(results)//2,kind=e['kind'],regime=e['regime'])),flush=True)
    if sources()!=manifest['sources'] or sha256(Path(args.weights))!=manifest['weights_sha256']:raise ValueError('validation input drift')
    if any(sha256(Path(p))!=h for p,h in cache_hashes.items()):raise ValueError('baseline cache drift')
    summary={model:{kind:summarize([r for r in results if r['model']==model and r['kind']==kind]) for kind in ('bearing','gear')} for model in models}
    by_regime={model:{regime:summarize([r for r in results if r['model']==model and r['regime']==regime]) for regime in ('a3_replay','a4_active')} for model in models}
    write_json(out/'results.json',results);write_json(out/'summary.json',dict(by_model=summary,by_regime=by_regime));return summary


def main():
    p=argparse.ArgumentParser()
    for n in ('dataset','weights','baseline-weights','baseline-validation','output'):p.add_argument('--'+n,required=True)
    print(json.dumps(run(p.parse_args()),indent=2));return 0
