"""Physically held continuous rotations with grouped stereo landmark supervision."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment,dataset_file,unlinked
from yolo_conveyor.annular.data import physical_label
from .assets import PARTS,NAMES,KEYPOINTS,anchors
from .scene import DatumScene,PoseController,VIEWS,SETUP
from .provenance import snapshot


def yaml(root,sigmas=None):
    result=dict(path=str(root),train='images/train',val='images/val',names={i:n for i,n in enumerate(NAMES)},
                kpt_shape=[5,3],kpt_names={i:KEYPOINTS for i in range(2)},flip_idx=list(range(5)))
    if sigmas is not None:result['kpt_oks_sigmas']=sigmas
    return result


def label_text(objects):
    result=[]
    for o in objects:
        x0,y0,x1,y1=o['bbox'];parts=[str(o['class_id']),f'{(x0+x1)/1280:.8f}',f'{(y0+y1)/1280:.8f}',f'{(x1-x0)/640:.8f}',f'{(y1-y0)/640:.8f}']
        for (x,y),v in zip(o['points'],o['visibility']):parts.extend([f'{x/640:.8f}',f'{y/640:.8f}',str(v)])
        result.append(' '.join(parts))
    return '\n'.join(result)+('\n' if result else '')


def annotate(scene,name,gain):
    option=scene.mj.MjvOption();option.sitegroup[:]=0;scene.renderer.update_scene(scene.data,camera=name,scene_option=option)
    image=np.clip(cv2.cvtColor(scene.renderer.render().copy(),cv2.COLOR_RGB2BGR).astype(float)*gain,0,255).astype(np.uint8)
    scene.renderer.enable_segmentation_rendering();seg=scene.renderer.render().copy();scene.renderer.disable_segmentation_rendering()
    scene.renderer.enable_depth_rendering();depth=scene.renderer.render().copy();scene.renderer.disable_depth_rendering()
    mask=(seg[:,:,1]==int(scene.mj.mjtObj.mjOBJ_GEOM))&np.isin(seg[:,:,0],scene.screw_geom_ids)
    objects=[];camera=scene.camera_info(name)
    if scene.feature and mask.sum()>80:
        y,x=np.nonzero(mask);world=scene.datum_world();points=scene.project(name,world)
        if min(x)<1 or min(y)<1 or max(x)>638 or max(y)>638 or np.any(points<1) or np.any(points>638):raise ValueError('datum cropped')
        t=np.array(camera['T_world_camera_cv']);cam=(world-t[:3,3])@t[:3,:3];visibility=[2]
        for pixel,z in zip(points[1:],cam[1:,2]):
            px,py=np.round(pixel).astype(int);patch=depth[py-2:py+3,px-2:px+3];part=mask[py-2:py+3,px-2:px+3]
            visibility.append(2 if np.any(part&(np.abs(patch-z)<.0008)) else 0)
        objects=[dict(class_id=list(PARTS).index(scene.kind),bbox=[int(min(x)),int(min(y)),int(max(x))+1,int(max(y))+1],
                      points=points.tolist(),visibility=visibility)]
    return image,objects,camera


def generate(args):
    if not 40<=args.train_poses<=160 or not 12<=args.val_poses<=48:raise ValueError('development budget')
    out=new_directory(args.output,[args.assets]);rows=[];episodes=[];scene=None
    manifest=dict(task='datum_pose_a2',scope='custom_proxy_simulation',real_formal_gate='closed',arguments=vars(args),sources=snapshot(out),environment=environment(),setup=SETUP)
    write_json(out/'manifest.json',manifest);write_json(out/'data.yaml',yaml(out));(out/'previews').mkdir()
    for split in ('train','val'):(out/'images'/split).mkdir(parents=True);(out/'labels'/split).mkdir(parents=True)
    try:
        for k,kind in enumerate(PARTS):
            for feature in (True,False):
                scene=DatumScene(args.assets,kind,feature);controller=PoseController(scene)
                (out/f'{kind}-{feature}-scene.xml').write_text(scene.xml,encoding='utf-8')
                for split,count,base in (('train',args.train_poses,141000+k*1000),('val',args.val_poses,151000+k*1000)):
                    numbers=count+(8 if split=='train' else 4) if feature else (12 if split=='train' else 4)
                    for i in range(numbers):
                        seed=base+i+(500 if not feature else 0);rng=np.random.default_rng(seed);empty=feature and i>=count
                        command=dict(tilt=float(rng.uniform(0,30)),azimuth=float(rng.uniform(0,360)),spin=float(rng.uniform(0,360)))
                        scene.initialize_pose(rng,command['spin'],empty);controller.execute(controller.pick(),lambda _:None)
                        controller.execute(controller.present(command['tilt'],command['azimuth']),lambda _:None)
                        gt=physical_label(scene)
                        if gt['held']!=bool(not empty):raise ValueError('physical grasp outside development envelope')
                        group=f'{kind}_{split}_{seed}';R=scene.data.xmat[scene.body_id].reshape(3,3).copy()
                        ep=dict(id=group,kind=kind,split=split,seed=seed,feature=feature,empty=empty,command=command,rotation=R.tolist(),center=scene.data.xpos[scene.body_id].tolist())
                        episodes.append(ep);scene.appearance(rng);gain=float(rng.uniform(.85,1.15))
                        for view in VIEWS:
                            image,objects,camera=annotate(scene,view,gain);stem=f'{group}_{view}'
                            ip=f'images/{split}/{stem}.png';lp=f'labels/{split}/{stem}.txt'
                            if not cv2.imwrite(str(out/ip),image):raise RuntimeError('image write')
                            with (out/lp).open('x',encoding='utf-8') as f:f.write(label_text(objects))
                            rows.append(dict(episode=group,split=split,view=view,image=ip,label=lp,objects=objects,camera=camera,
                                             image_sha256=sha256(out/ip),label_sha256=sha256(out/lp)))
                            if i<2 and feature and split=='val':
                                for o in objects:
                                    for j,(p,v) in enumerate(zip(o['points'],o['visibility'])):
                                        cv2.circle(image,tuple(np.round(p).astype(int)),3,(0,200,0) if v else (0,0,230),-1)
                                cv2.imwrite(str(out/'previews'/f'{stem}.jpg'),image)
                        if i%16==0 or i==numbers-1:print(json.dumps(dict(kind=kind,feature=feature,split=split,completed=i+1,total=numbers)),flush=True)
                scene.close();scene=None
        write_json(out/'records.json',rows);write_json(out/'episodes.json',episodes)
        meta=dict(status='completed',task='datum_pose_a2',real_formal_gate='closed',counts=dict(Counter(r['split'] for r in rows)),
                  episode_counts=dict(Counter(e['split'] for e in episodes)),files={name:sha256(out/name) for name in ('manifest.json','data.yaml','records.json','episodes.json')})
        write_json(out/'dataset.json',meta);audit=audit_dataset(out);write_json(out/'audit.json',audit)
        if audit['errors']:raise ValueError(str(audit['errors'][:5]))
        return meta
    finally:
        if scene is not None:scene.close()


def audit_dataset(path):
    root=unlinked(path);meta=json.loads((root/'dataset.json').read_text());rows=json.loads((root/'records.json').read_text());eps=json.loads((root/'episodes.json').read_text());errors=[]
    if meta['status']!='completed' or meta['task']!='datum_pose_a2' or meta['real_formal_gate']!='closed':errors.append('contract')
    for p,h in meta['files'].items():
        if sha256(dataset_file(root,p))!=h:errors.append('metadata drift')
    if json.loads((root/'data.yaml').read_text())!=json.loads(json.dumps(yaml(root,meta.get('oks_sigmas')))):errors.append('yaml drift')
    groups={e['id']:e for e in eps};seen=set();views=set();visibility=Counter()
    if len(groups)!=len(eps) or len({e['seed'] for e in eps})!=len(eps):errors.append('duplicate episode/seed')
    for r in rows:
        if r['episode'] not in groups or groups[r['episode']]['split']!=r['split']:errors.append('split leakage')
        marker=(r['episode'],r['view'])
        if marker in views:errors.append('duplicate view')
        views.add(marker)
        for field in ('image','label'):
            if sha256(dataset_file(root,r[field]))!=r[field+'_sha256']:errors.append('data drift')
        # Empty views can quantize identically within a split; cross-split bytes must remain disjoint.
        marker=(r['image_sha256'],r['split']);other=(r['image_sha256'],'val' if r['split']=='train' else 'train')
        if other in seen:errors.append('cross-split identical image')
        seen.add(marker)
        if (root/r['label']).read_text()!=label_text(r['objects']):errors.append('label mismatch')
        for o in r['objects']:
            pts=np.asarray(o['points']);box=np.asarray(o['bbox'])
            if pts.shape!=(5,2) or not np.isfinite(pts).all() or np.any(pts<0) or np.any(pts>=640):errors.append('bad keypoints')
            if box.shape!=(4,) or np.any(box<0) or np.any(box>640):errors.append('bad box')
            visibility.update(o['visibility'])
    if views!={(e['id'],v) for e in eps for v in VIEWS}:errors.append('missing view')
    for split in ('train','val'):
        for folder,field in (('images','image'),('labels','label')):
            if {p for p in (root/folder/split).rglob('*') if p.is_file()}!={root/r[field] for r in rows if r['split']==split}:errors.append('unlisted files')
    return dict(errors=errors,counts=meta['counts'],episode_counts=meta['episode_counts'],keypoint_visibility=dict(visibility))


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-poses',type=int,default=80);p.add_argument('--val-poses',type=int,default=24)
    print(json.dumps(generate(p.parse_args()),indent=2));return 0
