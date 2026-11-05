import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256
from yolo_conveyor.pose_cell.geometry import error
from .diagnostics import diagnose,oracle_predictions
from .provenance import snapshot
from .preprocessing import roi_inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    source=Path(args.source).resolve();out=new_directory(args.output,[source]);rows=json.loads((source/'results.json').read_text(encoding='utf-8'));reports=[]
    write_json(out/'manifest.json',dict(scope='A4_DEVELOPMENT_DIAGNOSTICS_OF_SEEN_A3',source=str(source),source_results_sha256=sha256(source/'results.json'),sources=snapshot(out)))
    for row in rows:
        if row['case'] in ('blind','plain') or row['status']!='UNCERTAIN':continue
        j=len(row['observations'])-1;folder=source/f"{row['index']:03d}_{row['kind']}_{row['case']}_{row['seed']}"/f'observation_{j}'
        evidence=json.loads((folder/'evidence.json').read_text(encoding='utf-8'));obs=row['observations'][j]
        images={v:cv2.imread(str(folder/f'{v}.png')) for v in evidence['cameras']}
        images,pred,cameras=roi_inputs(images,evidence['predictions'],evidence['cameras'])
        actual=diagnose(images,pred,cameras,row['kind']);oracle=diagnose(images,oracle_predictions(pred,cameras,row['kind'],obs['gt']),cameras,row['kind'])
        r=dict(index=row['index'],kind=row['kind'],case=row['case'],observation=j,actual=actual,oracle_seed_only=oracle)
        if oracle['measurement']:r['oracle_rotation_error']=error(oracle['measurement']['rotation'],obs['gt']['rotation'])
        reports.append(r);write_json(out/f"{row['index']:03d}.json",r)
        tiles=[]
        for name,image in images.items():
            im=image.copy()
            for d in pred[name]:
                for x,y,conf in d['keypoints'][:2]:
                    if conf>=.6:cv2.circle(im,(round(x),round(y)),4,(0,200,255),-1)
            cv2.putText(im,name,(8,25),0,.6,(0,200,255),1);tiles.append(cv2.resize(im,(320,320)))
        cv2.imwrite(str(out/f"{row['index']:03d}.jpg"),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
        print(json.dumps(dict(index=row['index'],case=row['case'],solver=actual['solver'],oracle=oracle['reason'],oracle_error=r.get('oracle_rotation_error'))),flush=True)
    write_json(out/'results.json',reports);return 0
