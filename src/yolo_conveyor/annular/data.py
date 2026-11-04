"""Physical held-image development data with per-grasp splits and six semantic classes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment,dataset_file,unlinked
from .assets import PARTS,NAMES
from .scene import AssemblyScene,SETUP
from .control import AssemblyController
from .provenance import snapshot

CASES=('front','back','tilted','empty')


def physical_label(scene):
    center=scene.data.xpos[scene.body_id].copy();axis=scene.data.xmat[scene.body_id].reshape(3,3)[:,2]
    contacts=set()
    for c in scene.data.contact:
        a,b=(int(scene.model.geom_bodyid[i]) for i in (c.geom1,c.geom2))
        if a==scene.body_id and b in scene.finger_bodies:contacts.add(b)
        if b==scene.body_id and a in scene.finger_bodies:contacts.add(a)
    held=len(contacts)==2 and np.linalg.norm(center-scene.tool_transform()[:3,3])<.03
    tilt=float(np.degrees(np.arccos(np.clip(abs(axis[2]),0,1))))
    face='tilted' if tilt>15 else ('front' if axis[2]>0 else 'back')
    return dict(held=bool(held),tilt_deg=tilt,face=face,center_world_m=center.tolist(),axis_world=axis.tolist(),
                gt='CONFORMING' if held and face=='front' else 'NONCONFORMING')


def annotated(scene,gain):
    option=scene.mj.MjvOption();option.sitegroup[:]=0
    scene.renderer.update_scene(scene.data,camera='inspect_rgb',scene_option=option)
    image=cv2.cvtColor(scene.renderer.render().copy(),cv2.COLOR_RGB2BGR)
    image=np.clip(image.astype(float)*gain,0,255).astype(np.uint8)
    scene.renderer.enable_segmentation_rendering();seg=scene.renderer.render().copy();scene.renderer.disable_segmentation_rendering()
    mask=(seg[:,:,1]==int(scene.mj.mjtObj.mjOBJ_GEOM))&np.isin(seg[:,:,0],scene.screw_geom_ids)
    gt=physical_label(scene);objects=[]
    if mask.sum()>40:
        y,x=np.nonzero(mask)
        if min(x)<1 or min(y)<1 or max(x)>=639 or max(y)>=639:raise ValueError('part clipped by camera')
        objects=[dict(class_id=NAMES.index(scene.kind+'_'+gt['face']),bbox_xyxy=[int(min(x)),int(min(y)),int(max(x))+1,int(max(y))+1])]
    elif gt['held']:raise ValueError('held part not visible')
    return image,objects,gt


def labels(objects):
    output=[]
    for obj in objects:
        x0,y0,x1,y1=obj['bbox_xyxy']
        output.append(f"{obj['class_id']} {(x0+x1)/1280:.8f} {(y0+y1)/1280:.8f} {(x1-x0)/640:.8f} {(y1-y0)/640:.8f}")
    return '\n'.join(output)+('\n' if output else '')


def yaml(root):
    return dict(path=str(root),train='images/train',val='images/val',names={i:name for i,name in enumerate(NAMES)})


def generate(args):
    if not (24<=args.train_episodes<=120 and 8<=args.val_episodes<=40 and 1<=args.variants<=6):raise ValueError('dataset budget')
    out=new_directory(args.output,[args.assets]);start=time.perf_counter();rows=[];episodes=[];seen=set();scene=None
    sources=snapshot(out)
    write_json(out/'data.yaml',yaml(out));write_json(out/'parts.json',PARTS);write_json(out/'setup.json',SETUP)
    for split in ('train','val'):
        (out/'images'/split).mkdir(parents=True);(out/'labels'/split).mkdir(parents=True)
    try:
        for k,kind in enumerate(PARTS):
            scene=AssemblyScene(args.assets,kind);controller=AssemblyController(scene)
            (out/f'{kind}-scene.xml').write_text(scene.xml,encoding='utf-8')
            for split,count,base in (('train',args.train_episodes,101000+k*1000),('val',args.val_episodes,111000+k*1000)):
                for index in range(count):
                    seed=base+index;rng=np.random.default_rng(seed);case=CASES[index%4]
                    face='back' if case=='back' or (case=='tilted' and index%8==2) else 'front'
                    tilt=float(rng.choice([-1,1])*rng.uniform(30,45)) if case=='tilted' else 0.
                    scene.initialize(rng,face,case=='empty');controller.execute(controller.pick(tilt),lambda _:None)
                    truth=physical_label(scene)
                    if truth['held']!=(case!='empty'):raise ValueError('physical grasp failed in requested generation envelope')
                    group=f'{kind}_{split}_{seed}';ep=dict(id=group,kind=kind,split=split,seed=seed,case=case,tilt_command_deg=tilt,gt=truth,variants=[])
                    episodes.append(ep)
                    count_variants=args.variants if split=='train' else min(2,args.variants)
                    for variant in range(count_variants):
                        scene.appearance(rng);gain=float(rng.uniform(.85,1.15));image,objects,gt=annotated(scene,gain)
                        ok,png=cv2.imencode('.png',image)
                        if not ok:raise RuntimeError('PNG encoding')
                        digest=hashlib.sha256(png).hexdigest()
                        if digest in seen:continue
                        seen.add(digest);ep['variants'].append(variant);stem=f'{group}_v{variant}'
                        ip=f'images/{split}/{stem}.png';lp=f'labels/{split}/{stem}.txt'
                        with (out/ip).open('xb') as f:f.write(png.tobytes())
                        with (out/lp).open('x',encoding='utf-8') as f:f.write(labels(objects))
                        rows.append(dict(episode=group,split=split,kind=kind,variant=variant,image=ip,label=lp,image_sha256=digest,
                                         label_sha256=sha256(out/lp),objects=objects,gt=gt,gain=gain))
                    if not ep['variants']:raise ValueError('all variants duplicated')
                    if index%12==0 or index==count-1:print(json.dumps(dict(kind=kind,split=split,completed=index+1,requested=count,images=len(rows))),flush=True)
            scene.close();scene=None
        write_json(out/'records.json',rows);write_json(out/'episodes.json',episodes)
        meta=dict(task='annular_face_a1',status='completed',scope='simulation_development',real_formal_gate='closed',arguments=vars(args),
                  classes=NAMES,sources=sources,environment=environment(),counts=dict(Counter(r['split'] for r in rows)),
                  episode_counts=dict(Counter(e['split'] for e in episodes)),seconds=time.perf_counter()-start,
                  files={name:sha256(out/name) for name in ('records.json','episodes.json','data.yaml','parts.json','setup.json','bearing-scene.xml','gear-scene.xml')})
        write_json(out/'dataset.json',meta);audit=audit_dataset(out);write_json(out/'audit.json',audit)
        if audit['errors']:raise ValueError(str(audit['errors'][:5]))
        return meta
    finally:
        if scene is not None:scene.close()


def audit_dataset(path):
    root=unlinked(path);meta=json.loads((root/'dataset.json').read_text(encoding='utf-8'));errors=[];seen=set();groups={}
    if (meta.get('task')!='annular_face_a1' or meta.get('status')!='completed' or meta.get('classes')!=NAMES
            or meta.get('real_formal_gate')!='closed'):errors.append('dataset contract')
    for relative,digest in meta['files'].items():
        if sha256(dataset_file(root,relative))!=digest:errors.append('file drift: '+relative)
    rows=json.loads((root/'records.json').read_text(encoding='utf-8'));episodes=json.loads((root/'episodes.json').read_text(encoding='utf-8'))
    known={e['id']:e for e in episodes}
    if len(known)!=len(episodes) or len({e['seed'] for e in episodes})!=len(episodes):errors.append('duplicate episodes/seeds')
    if json.loads((root/'data.yaml').read_text(encoding='utf-8'))!=json.loads(json.dumps(yaml(root))):errors.append('training YAML mismatch')
    variants=set()
    for row in rows:
        split=row['split'];group=row['episode']
        if split not in ('train','val') or group not in known or known[group]['split']!=split:errors.append('split/group')
        if group in groups and groups[group]!=split:errors.append('episode leakage')
        groups[group]=split
        marker=(group,row['variant'])
        if marker in variants:errors.append('duplicate variant')
        variants.add(marker)
        if not row['image'].startswith('images/'+split+'/') or not row['label'].startswith('labels/'+split+'/'):errors.append('split path mismatch')
        for field in ('image','label'):
            if sha256(dataset_file(root,row[field]))!=row[field+'_sha256']:errors.append('image/label drift')
        if row['image_sha256'] in seen:errors.append('duplicate images')
        seen.add(row['image_sha256'])
        if (root/row['label']).read_text(encoding='utf-8')!=labels(row['objects']):errors.append('label mismatch')
        for obj in row['objects']:
            box=np.asarray(obj['bbox_xyxy'])
            if (obj['class_id'] not in range(len(NAMES)) or box.shape!=(4,) or not np.isfinite(box).all()
                    or np.any(box<0) or np.any(box>640) or box[2]<=box[0] or box[3]<=box[1]):errors.append('invalid object label')
        image=cv2.imread(str(root/row['image']))
        if image is None or image.shape!=(640,640,3):errors.append('image shape')
    if dict(Counter(r['split'] for r in rows))!=meta['counts'] or dict(Counter(groups.values()))!=meta['episode_counts']:errors.append('counts')
    if variants!={(e['id'],v) for e in episodes for v in e['variants']}:errors.append('variant inventory mismatch')
    for split in ('train','val'):
        for folder,field in (('images','image'),('labels','label')):
            actual={p for p in (root/folder/split).rglob('*') if p.is_file()}
            if actual!={root/r[field] for r in rows if r['split']==split}:errors.append('unlisted '+folder)
    return dict(errors=errors,counts=meta['counts'],episode_counts=meta['episode_counts'],classes=dict(Counter(NAMES[o['class_id']] for r in rows for o in r['objects'])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-episodes',type=int,default=72);p.add_argument('--val-episodes',type=int,default=24);p.add_argument('--variants',type=int,default=4)
    m=generate(p.parse_args());print(json.dumps({k:m[k] for k in ('counts','episode_counts','seconds')}));return 0
