"""Grab-grouped A4 data; every active view stays with its physical source grab."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment,unlinked,dataset_file
from yolo_conveyor.annular.data import physical_label
from yolo_conveyor.datum_pose.data import label_text
from yolo_conveyor.pose_cell.data import annotate,yaml,audit_dataset as audit_a3
from yolo_conveyor.pose_cell.data_centers import center_objects
from yolo_conveyor.pose_cell.faces import PARTS,CENTER_SCHEMA,NAMES
from yolo_conveyor.pose_cell.scene import CellScene,VIEWS,SETUP
from .control import ActiveController
from .provenance import snapshot


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def audit_dataset(path):
    root=unlinked(path);meta=read(root/'dataset.json');rows=read(root/'records.json');eps=read(root/'episodes.json');errors=[]
    if meta['task']!='active_pose_a4' or meta['status']!='completed' or meta['point_schema']!=CENTER_SCHEMA or meta['real_formal_gate']!='closed':errors.append('contract')
    if read(root/'data.yaml')!=json.loads(json.dumps(yaml(root,CENTER_SCHEMA))):errors.append('yaml drift')
    for rel,h in meta['files'].items():
        if sha256(dataset_file(root,rel))!=h:errors.append('metadata drift')
    by_id={e['id']:e for e in eps};grabs={};seeds={};seen=set();hashes={};visibility=Counter()
    if len(by_id)!=len(eps):errors.append('duplicate capture')
    for e in eps:
        g=e['grab_id'];key=(e['split'],e['seed'])
        if g in grabs and grabs[g]!=key:errors.append('grab split leakage')
        if e['seed'] in seeds and seeds[e['seed']]!=g:errors.append('seed reused across grabs')
        grabs[g]=key;seeds[e['seed']]=g
    for row in rows:
        e=by_id.get(row['episode'])
        if e is None or e['split']!=row['split']:errors.append('capture split mismatch')
        key=(row['episode'],row['view'])
        if key in seen:errors.append('duplicate view')
        seen.add(key)
        for f in ('image','label'):
            if sha256(dataset_file(root,row[f]))!=row[f+'_sha256']:errors.append('data drift')
        if row['image_sha256'] in hashes and hashes[row['image_sha256']]!=row['split']:errors.append('cross split duplicate')
        hashes[row['image_sha256']]=row['split']
        if (root/row['label']).read_text(encoding='utf-8')!=label_text(row['objects']):errors.append('label mismatch')
        for obj in row['objects']:
            points=np.asarray(obj['points']);box=np.asarray(obj['bbox'])
            if (points.shape!=(5,2) or not np.isfinite(points).all() or np.any(points<0) or np.any(points>=640)
                    or obj['visibility'][2:]!=[0,0,0] or np.any(points[2:]!=0) or obj['class_id'] not in range(4)):errors.append('bad keypoints')
            if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box<0) or np.any(box>640):errors.append('bad box')
            visibility.update(obj['visibility'])
    if seen!={(e['id'],v) for e in eps for v in VIEWS}:errors.append('missing views')
    for split in ('train','val'):
        for directory,field in (('images','image'),('labels','label')):
            if {p for p in (root/directory/split).rglob('*') if p.is_file()}!={root/r[field] for r in rows if r['split']==split}:errors.append('unlisted data')
    return dict(errors=errors,counts=meta['counts'],capture_counts=meta['capture_counts'],grab_counts=dict(Counter(v[0] for v in grabs.values())),
                visibility=dict(visibility),classes=dict(Counter(NAMES[o['class_id']] for r in rows for o in r['objects'])))


def generate(args):
    source=unlinked(args.source)
    if audit_a3(source)['errors'] or read(source/'dataset.json').get('point_schema')!=CENTER_SCHEMA:raise ValueError('A3 source dataset required')
    out=new_directory(args.output,[source,args.assets]);rows=[];episodes=[];grabs=[];scene=None
    write_json(out/'manifest.json',dict(task='active_pose_a4',scope='simulation_development',real_formal_gate='closed',arguments=vars(args),
        source_dataset_sha256=sha256(source/'dataset.json'),sources=snapshot(out),setup=SETUP,environment=environment()))
    write_json(out/'data.yaml',yaml(out,CENTER_SCHEMA))
    for split in ('train','val'):
        (out/'images'/split).mkdir(parents=True);(out/'labels'/split).mkdir(parents=True)
    for e in read(source/'episodes.json'):episodes.append({**e,'grab_id':e['id'],'observation_pose':0,'regime':'a3_replay'})
    for r in read(source/'records.json'):
        shutil.copyfile(source/r['image'],out/r['image']);shutil.copyfile(source/r['label'],out/r['label']);rows.append(r)
    try:
        for k,kind in enumerate(PARTS):
            for feature in (True,False):
                scene=CellScene(args.assets,kind,feature);c=ActiveController(scene);(out/f'{kind}-{feature}-scene.xml').write_text(scene.xml,encoding='utf-8')
                for split,n,base in (('train',args.train_grabs,221000+k*1000),('val',args.val_grabs,231000+k*1000)):
                    total=n+(2 if split=='train' else 1) if feature else (4 if split=='train' else 2)
                    for i in range(total):
                        seed=base+i+(500 if not feature else 0);rng=np.random.default_rng(seed);empty=feature and i>=n
                        spin=float(rng.uniform(-10,110)) if i%3 else float(rng.uniform(0,360));flipped=bool(i%2)
                        angle=float(rng.uniform(75,130)) if i%4 else (180. if flipped else 0.)
                        yaw=float(rng.uniform(-12,12));grab=f'{kind}_{split}_{seed}';entry=dict(id=grab,seed=seed,split=split,kind=kind,feature=feature,empty=empty,command=dict(spin=spin,angle=angle,yaw=yaw),failures=[])
                        scene.initialize_pose(rng,spin,flipped,empty)
                        c.execute(c.pick(),lambda _:None);c.execute(c.present(angle,yaw),lambda _:None)
                        if physical_label(scene)['held']!=(not empty):raise ValueError('initial data grip failed')
                        anchor=scene.tool_transform().copy()
                        for pose in range(3):
                            if pose:
                                try:c.execute(c.observe_pose(anchor,pose-1),lambda _:None)
                                except ValueError as exc:entry['failures'].append(dict(pose=pose,reason=str(exc)));continue
                            if physical_label(scene)['held']!=(not empty):raise ValueError('active data grip failed')
                            ident=grab+f'_pose{pose}';e=dict(id=ident,grab_id=grab,observation_pose=pose,kind=kind,split=split,seed=seed,feature=feature,empty=empty,flipped=flipped,
                                command=entry['command'],regime='a4_active',rotation=scene.data.xmat[scene.body_id].reshape(3,3).tolist(),center=scene.data.xpos[scene.body_id].tolist())
                            episodes.append(e)
                            for view in VIEWS:
                                im,objects,camera=annotate(scene,view,float(rng.uniform(.9,1.1)));objects=center_objects(objects,e,camera);stem=ident+'_'+view
                                ip=f'images/{split}/{stem}.png';lp=f'labels/{split}/{stem}.txt'
                                if not cv2.imwrite(str(out/ip),im):raise ValueError('capture write failed')
                                (out/lp).write_text(label_text(objects),encoding='utf-8')
                                rows.append(dict(episode=ident,split=split,view=view,image=ip,label=lp,objects=objects,camera=camera,image_sha256=sha256(out/ip),label_sha256=sha256(out/lp)))
                        grabs.append(entry);write_json(out/'progress.json',dict(kind=kind,split=split,grab=i+1,total=total),replace=True)
                        print(json.dumps(dict(kind=kind,feature=feature,split=split,grab=i+1,total=total,unreachable=len(entry['failures']))),flush=True)
                scene.close();scene=None
        write_json(out/'records.json',rows);write_json(out/'episodes.json',episodes);write_json(out/'grabs.json',grabs)
        meta=dict(task='active_pose_a4',status='completed',real_formal_gate='closed',input_space='fixed_roi',point_schema=CENTER_SCHEMA,
                  counts=dict(Counter(r['split'] for r in rows)),capture_counts=dict(Counter(e['split'] for e in episodes)),
                  files={n:sha256(out/n) for n in ('manifest.json','data.yaml','records.json','episodes.json','grabs.json')})
        write_json(out/'dataset.json',meta);audit=audit_dataset(out);write_json(out/'audit.json',audit)
        if audit['errors']:raise ValueError(str(audit['errors'][:5]))
        return meta
    finally:
        if scene is not None:scene.close()


def main():
    p=argparse.ArgumentParser()
    for n in ('source','assets','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--train-grabs',type=int,choices=range(12,33),default=20);p.add_argument('--val-grabs',type=int,choices=range(4,13),default=6)
    print(json.dumps(generate(p.parse_args()),indent=2));return 0
