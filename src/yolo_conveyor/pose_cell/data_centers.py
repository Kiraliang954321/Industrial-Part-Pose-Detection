"""Versioned center supervision plus physical terminal-pose development captures."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment,unlinked
from yolo_conveyor.held.geometry import project
from yolo_conveyor.annular.data import physical_label
from yolo_conveyor.datum_pose.data import label_text
from .data import annotate,audit_dataset,yaml
from .faces import PARTS,landmarks,CENTER_SCHEMA
from .scene import CellScene,VIEWS,SETUP
from .control import CellController
from .provenance import snapshot


def center_objects(objects,episode,camera):
    result=copy.deepcopy(objects)
    for o in result:
        face=1 if o['class_id']%2 else -1;local=landmarks(episode['kind'],face)[1:].mean(0)
        center=project(camera,np.array(episode['rotation'])@local+episode['center'])
        visible=any(min(o['visibility'][a],o['visibility'][b])==2 for a,b in ((1,3),(2,4)))
        o['points']=[o['points'][0],center.tolist(),[0.,0.],[0.,0.],[0.,0.]]
        o['visibility']=[o['visibility'][0],2 if visible else 0,0,0,0]
    return result


def prepare(args):
    source=unlinked(args.source)
    if audit_dataset(source)['errors']:raise ValueError('source audit')
    if not 8<=args.terminal_train<=24 or not 2<=args.terminal_val<=8:raise ValueError('terminal data budget')
    out=new_directory(args.output,[source,args.assets]);rows=[];scene=None
    episodes=json.loads((source/'episodes.json').read_text(encoding='utf-8'));groups={e['id']:e for e in episodes}
    write_json(out/'manifest.json',dict(task='pose_cell_a3',scope='simulation_development',real_formal_gate='closed',arguments=vars(args),
        point_schema=CENTER_SCHEMA,source_dataset_sha256=sha256(source/'dataset.json'),setup=SETUP,sources=snapshot(out),environment=environment()))
    write_json(out/'data.yaml',yaml(out,CENTER_SCHEMA))
    for split in ('train','val'):
        (out/'images'/split).mkdir(parents=True);(out/'labels'/split).mkdir(parents=True)
    def save(row,image=None):
        if image is None:shutil.copyfile(source/row['image'],out/row['image'])
        elif not cv2.imwrite(str(out/row['image']),image):raise RuntimeError('image write')
        (out/row['label']).write_text(label_text(row['objects']),encoding='utf-8')
        row.update(image_sha256=sha256(out/row['image']),label_sha256=sha256(out/row['label']));rows.append(row)
    for r in json.loads((source/'records.json').read_text(encoding='utf-8')):
        r['objects']=center_objects(r['objects'],groups[r['episode']],r['camera']);save(r)
    try:
        for k,kind in enumerate(PARTS):
            scene=CellScene(args.assets,kind,True);controller=CellController(scene)
            (out/f'{kind}-terminal-scene.xml').write_text(scene.xml,encoding='utf-8')
            for split,total,base in (('train',args.terminal_train,183000+k*1000),('val',args.terminal_val,193000+k*1000)):
                for i in range(total):
                    seed=base+i;rng=np.random.default_rng(seed);flipped=bool(i%2);angle=180. if flipped else 0.
                    yaw=float(rng.uniform(-35,0));spin=float(rng.uniform(-10,70))-yaw
                    scene.initialize_pose(rng,spin,flipped);controller.execute(controller.pick(),lambda _:None)
                    controller.execute(controller.present(angle,yaw),lambda _:None)
                    if not physical_label(scene)['held']:raise ValueError('terminal grip failed')
                    group=f'{kind}_{split}_terminal_{seed}'
                    e=dict(id=group,kind=kind,split=split,seed=seed,feature=True,empty=False,flipped=flipped,regime='terminal',
                           command=dict(spin=spin,angle=angle,yaw=yaw),rotation=scene.data.xmat[scene.body_id].reshape(3,3).tolist(),center=scene.data.xpos[scene.body_id].tolist())
                    episodes.append(e)
                    for view in VIEWS:
                        im,objects,camera=annotate(scene,view,float(rng.uniform(.9,1.1)));stem=group+'_'+view
                        save(dict(episode=group,split=split,view=view,image=f'images/{split}/{stem}.png',label=f'labels/{split}/{stem}.txt',
                                  objects=center_objects(objects,e,camera),camera=camera),im)
                    print(json.dumps(dict(kind=kind,split=split,terminal=i+1,total=total)),flush=True)
            scene.close();scene=None
        write_json(out/'records.json',rows);write_json(out/'episodes.json',episodes)
        meta=dict(task='pose_cell_a3',status='completed',real_formal_gate='closed',input_space='fixed_roi',point_schema=CENTER_SCHEMA,
                  counts=dict(Counter(r['split'] for r in rows)),episode_counts=dict(Counter(e['split'] for e in episodes)),
                  files={n:sha256(out/n) for n in ('manifest.json','data.yaml','records.json','episodes.json')})
        write_json(out/'dataset.json',meta);audit=audit_dataset(out);write_json(out/'audit.json',audit)
        if audit['errors']:raise ValueError(str(audit['errors'][:5]))
        return meta
    finally:
        if scene is not None:scene.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','assets','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--terminal-train',type=int,default=20);p.add_argument('--terminal-val',type=int,default=6)
    print(json.dumps(prepare(p.parse_args()),indent=2));return 0
