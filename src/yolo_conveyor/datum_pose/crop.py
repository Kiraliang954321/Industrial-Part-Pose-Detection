"""Derived fixed-station ROI dataset; preserves physical episode splits and calibration."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256
from .data import audit_dataset,yaml,label_text
from .provenance import snapshot

ROI=(160,160,480,480)


def crop_image(image):return cv2.resize(image[160:480,160:480],(640,640),interpolation=cv2.INTER_LINEAR)


def derive(args):
    root=Path(args.dataset).resolve()
    if audit_dataset(root)['errors']:raise ValueError('parent audit failed')
    out=new_directory(args.output,[root]);rows=json.loads((root/'records.json').read_text());episodes=json.loads((root/'episodes.json').read_text())
    manifest=dict(task='datum_pose_a2_roi',parent=str(root),parent_dataset_sha256=sha256(root/'dataset.json'),roi=list(ROI),
                  scope='derived views, not additional independent grasps',sources=snapshot(out))
    sigmas=[args.oks_sigma]*5 if args.oks_sigma is not None else None
    if sigmas and not 0<args.oks_sigma<=.2:raise ValueError('invalid OKS scale')
    manifest['oks_sigmas']=sigmas
    write_json(out/'manifest.json',manifest);write_json(out/'data.yaml',yaml(out,sigmas));write_json(out/'episodes.json',episodes)
    transform=np.array([[2,0,-320],[0,2,-320],[0,0,1.]])
    for row in rows:
        image=cv2.imread(str(root/row['image']));output=crop_image(image)
        for o in row['objects']:
            points=(np.array(o['points'])-160)*2;box=(np.array(o['bbox'])-160)*2
            if np.any(points<0) or np.any(points>=640) or np.any(box<0) or np.any(box>640):raise ValueError('ROI clips a required feature')
            o['points']=points.tolist();o['bbox']=box.tolist()
        row['camera']['K']=(transform@np.array(row['camera']['K'])).tolist()
        ip=out/row['image'];lp=out/row['label'];ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(ip),output):raise RuntimeError('image write')
        with lp.open('x',encoding='utf-8') as f:f.write(label_text(row['objects']))
        row['parent_image_sha256']=row['image_sha256'];row['image_sha256']=sha256(ip);row['label_sha256']=sha256(lp)
    write_json(out/'records.json',rows);meta=deepcopy(json.loads((root/'dataset.json').read_text()));meta['input_space']='fixed_roi'
    if sigmas:meta['oks_sigmas']=sigmas
    meta['files']={n:sha256(out/n) for n in ('manifest.json','data.yaml','records.json','episodes.json')}
    write_json(out/'dataset.json',meta);audit=audit_dataset(out);write_json(out/'audit.json',audit)
    if audit['errors']:raise ValueError(str(audit['errors'][:5]))
    return audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--oks-sigma',type=float)
    print(json.dumps(derive(p.parse_args()),indent=2));return 0
