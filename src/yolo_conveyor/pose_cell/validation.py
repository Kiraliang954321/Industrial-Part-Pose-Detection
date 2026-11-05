"""Direct held-out development pose validation, with misses kept in coverage."""
import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256
from yolo_conveyor.orientation.evaluation import rate,distribution
from .data import audit_dataset
from .predictor import Predictor
from .metrology import measure
from .geometry import error
from .provenance import snapshot,sources


def run(args):
    root=Path(args.dataset).resolve()
    if audit_dataset(root)['errors']:raise ValueError('data audit')
    out=new_directory(args.output,[root,args.weights]);os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    manifest=dict(scope='A3_VAL_ONLY',weights_sha256=sha256(Path(args.weights)),dataset_sha256=sha256(root/'dataset.json'),sources=snapshot(out))
    write_json(out/'manifest.json',manifest)
    predictor=Predictor(args.weights);rows=json.loads((root/'records.json').read_text(encoding='utf-8'));eps=json.loads((root/'episodes.json').read_text(encoding='utf-8'));results=[]
    for e in eps:
        if e['split']!='val':continue
        views=[r for r in rows if r['episode']==e['id']];images={r['view']:cv2.imread(str(root/r['image'])) for r in views};pred=predictor.predict(images,preprocessed=True)
        r=dict(episode=e['id'],kind=e['kind'],eligible=e['feature'] and not e['empty'],flipped=e['flipped'],command=e['command'],predictions=pred,measurement=None)
        try:
            m=measure(images,pred,{v['view']:v['camera'] for v in views},e['kind'],pixel_scale=2.);r.update(measurement=m,rotation_error=error(m['rotation'],e['rotation']),position_error_mm=float(np.linalg.norm(np.array(m['translation_m'])-e['center'])*1000))
        except (ValueError,np.linalg.LinAlgError) as exc:r['reason']=str(exc)
        results.append(r)
        if len(results)%8==0:print(json.dumps(dict(completed=len(results),kind=e['kind'],measured=r['measurement'] is not None)),flush=True)
    summary={}
    for kind in ('bearing','gear'):
        positive=[r for r in results if r['kind']==kind and r['eligible']];usable=[r for r in positive if r['measurement']];negative=[r for r in results if r['kind']==kind and not r['eligible']]
        summary[kind]=dict(coverage=rate(len(usable),len(positive)),negative_with_pose=rate(sum(r['measurement'] is not None for r in negative),len(negative)),
                          rotation_deg=distribution([r['rotation_error']['rotation_deg'] for r in usable]),axis_deg=distribution([r['rotation_error']['axis_deg'] for r in usable]),position_mm=distribution([r['position_error_mm'] for r in usable]))
    if sources()!=manifest['sources'] or sha256(Path(args.weights))!=manifest['weights_sha256']:raise ValueError('validation input drift')
    write_json(out/'results.json',results);write_json(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser()
    for n in ('dataset','weights','output'):p.add_argument('--'+n,required=True)
    print(json.dumps(run(p.parse_args()),indent=2));return 0
