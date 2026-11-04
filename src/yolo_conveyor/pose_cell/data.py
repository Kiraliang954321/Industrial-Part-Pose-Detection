"""Physical wide-angle/flip data with visible-end labels and hidden datum visibility."""
import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment,unlinked,dataset_file
from yolo_conveyor.annular.data import physical_label
from yolo_conveyor.datum_pose.data import label_text
from yolo_conveyor.datum_pose.crop import crop_image
from .faces import PARTS,NAMES,KEYPOINTS,ROI,landmarks,class_id,CORNER_SCHEMA,CENTER_SCHEMA,keypoint_names
from .scene import CellScene,VIEWS,SETUP
from .control import CellController
from .provenance import snapshot


def yaml(root,schema=CORNER_SCHEMA):
    return dict(path=str(root),train='images/train',val='images/val',names={i:n for i,n in enumerate(NAMES)},
                kpt_shape=[5,3],kpt_names={i:keypoint_names(schema) for i in range(4)},
                kpt_oks_sigmas=([.015,.015,.025,.025,.025] if schema==CENTER_SCHEMA else [.025]*5),flip_idx=list(range(5)))


def annotate(scene,name,gain):
    option=scene.mj.MjvOption();option.sitegroup[:]=0;scene.renderer.update_scene(scene.data,camera=name,scene_option=option)
    im=np.clip(cv2.cvtColor(scene.renderer.render().copy(),cv2.COLOR_RGB2BGR).astype(float)*gain,0,255).astype(np.uint8)
    scene.renderer.enable_segmentation_rendering();seg=scene.renderer.render().copy();scene.renderer.disable_segmentation_rendering()
    scene.renderer.enable_depth_rendering();depth=scene.renderer.render().copy();scene.renderer.disable_depth_rendering()
    mask=(seg[:,:,1]==int(scene.mj.mjtObj.mjOBJ_GEOM))&np.isin(seg[:,:,0],scene.screw_geom_ids)
    # MuJoCo segmentation can contain isolated ID pixels at anti-aliased edges of
    # other objects. Keep actual connected visible regions, not these speckles.
    count,components,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    keep=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=30]
    mask=np.isin(components,keep)
    camera=scene.camera_info(name);R=scene.data.xmat[scene.body_id].reshape(3,3);center=scene.data.xpos[scene.body_id];t=np.array(camera['T_world_camera_cv'])
    direction=t[:3,3]-center;cosine=float(R[:,2]@direction/np.linalg.norm(direction));face=1 if cosine>0 else -1;objects=[]
    if mask.sum()>80 and abs(cosine)>.28:
        y,x=np.nonzero(mask);points=landmarks(scene.kind,face);world=points@R.T+center;uv=scene.project(name,world);cp=(world-t[:3,3])@t[:3,:3]
        vis=[2]
        for pixel,z in zip(uv[1:],cp[1:,2]):
            px,py=np.round(pixel).astype(int)
            valid=2<=px<638 and 2<=py<638 and scene.feature
            if valid:valid=bool(np.any(mask[py-2:py+3,px-2:px+3]&(np.abs(depth[py-2:py+3,px-2:px+3]-z)<.0005)))
            vis.append(2 if valid else 0)
        box=np.array([min(x),min(y),max(x)+1,max(y)+1]);box=(box-160)*2;uv=(uv-160)*2
        if np.any(box<0) or np.any(box>640) or np.any(uv<0) or np.any(uv>=640):raise ValueError('ROI clipping')
        objects=[dict(class_id=class_id(scene.kind,face),bbox=box.tolist(),points=uv.tolist(),visibility=vis)]
    camera['K']=(np.array([[2,0,-320],[0,2,-320],[0,0,1.]])@np.array(camera['K'])).tolist()
    return crop_image(im),objects,camera


def generate(args):
    if not 40<=args.train_poses<=120 or not 12<=args.val_poses<=32:raise ValueError('dataset budget')
    out=new_directory(args.output,[args.assets]);rows=[];episodes=[];scene=None
    write_json(out/'manifest.json',dict(task='pose_cell_a3',scope='simulation_development',real_formal_gate='closed',arguments=vars(args),setup=SETUP,sources=snapshot(out),environment=environment()))
    write_json(out/'data.yaml',yaml(out))
    for split in ('train','val'):(out/'images'/split).mkdir(parents=True);(out/'labels'/split).mkdir(parents=True)
    try:
        for k,kind in enumerate(PARTS):
            for feature in (True,False):
                scene=CellScene(args.assets,kind,feature);c=CellController(scene);(out/f'{kind}-{feature}-scene.xml').write_text(scene.xml,encoding='utf-8')
                for split,n,base in (('train',args.train_poses,181000+k*1000),('val',args.val_poses,191000+k*1000)):
                    total=n+(8 if split=='train' else 4) if feature else (8 if split=='train' else 4)
                    for i in range(total):
                        seed=base+i+(500 if not feature else 0);rng=np.random.default_rng(seed);empty=feature and i>=n
                        spin=float(rng.uniform(0,360));flipped=bool(i%2);angle=float(rng.uniform(0,180));yaw=float(rng.uniform(-15,15))
                        scene.initialize_pose(rng,spin,flipped,empty);c.execute(c.pick(),lambda _:None);c.execute(c.present(angle,yaw),lambda _:None)
                        gt=physical_label(scene)
                        if gt['held']!=(not empty):raise ValueError('physical grip failed')
                        group=f'{kind}_{split}_{seed}';episodes.append(dict(id=group,kind=kind,split=split,seed=seed,feature=feature,empty=empty,
                            flipped=flipped,command=dict(spin=spin,angle=angle,yaw=yaw),rotation=scene.data.xmat[scene.body_id].reshape(3,3).tolist(),center=scene.data.xpos[scene.body_id].tolist()))
                        for view in VIEWS:
                            im,objects,cam=annotate(scene,view,float(rng.uniform(.9,1.1)));stem=group+'_'+view;ip=f'images/{split}/{stem}.png';lp=f'labels/{split}/{stem}.txt'
                            if not cv2.imwrite(str(out/ip),im):raise RuntimeError('image write')
                            with (out/lp).open('x',encoding='utf-8') as f:f.write(label_text(objects))
                            rows.append(dict(episode=group,split=split,view=view,image=ip,label=lp,objects=objects,camera=cam,image_sha256=sha256(out/ip),label_sha256=sha256(out/lp)))
                        if i%12==0 or i==total-1:print(json.dumps(dict(kind=kind,feature=feature,split=split,completed=i+1,total=total)),flush=True)
                scene.close();scene=None
        write_json(out/'records.json',rows);write_json(out/'episodes.json',episodes)
        meta=dict(task='pose_cell_a3',status='completed',real_formal_gate='closed',input_space='fixed_roi',counts=dict(Counter(r['split'] for r in rows)),episode_counts=dict(Counter(e['split'] for e in episodes)),files={n:sha256(out/n) for n in ('manifest.json','data.yaml','records.json','episodes.json')})
        write_json(out/'dataset.json',meta);audit=audit_dataset(out);write_json(out/'audit.json',audit)
        if audit['errors']:raise ValueError(str(audit['errors'][:5]))
        return meta
    finally:
        if scene is not None:scene.close()


def audit_dataset(path):
    root=unlinked(path);m=json.loads((root/'dataset.json').read_text(encoding='utf-8'));rows=json.loads((root/'records.json').read_text(encoding='utf-8'));eps=json.loads((root/'episodes.json').read_text(encoding='utf-8'));errors=[]
    if m['task']!='pose_cell_a3' or m['status']!='completed' or m['real_formal_gate']!='closed':errors.append('contract')
    schema=m.get('point_schema',CORNER_SCHEMA)
    if json.loads((root/'data.yaml').read_text(encoding='utf-8'))!=json.loads(json.dumps(yaml(root,schema))):errors.append('yaml')
    for p,h in m['files'].items():
        if sha256(dataset_file(root,p))!=h:errors.append('metadata drift')
    groups={e['id']:e for e in eps};views=set();hashes={};vis=Counter()
    if len(groups)!=len(eps) or len({e['seed'] for e in eps})!=len(eps):errors.append('duplicate episodes')
    for r in rows:
        if r['episode'] not in groups or groups[r['episode']]['split']!=r['split']:errors.append('split leakage')
        key=(r['episode'],r['view'])
        if key in views:errors.append('duplicate view')
        views.add(key)
        for f in ('image','label'):
            if sha256(dataset_file(root,r[f]))!=r[f+'_sha256']:errors.append('data drift')
        if r['image_sha256'] in hashes and hashes[r['image_sha256']]!=r['split']:errors.append('cross-split duplicate image')
        hashes[r['image_sha256']]=r['split']
        if (root/r['label']).read_text(encoding='utf-8')!=label_text(r['objects']):errors.append('label drift')
        for o in r['objects']:
            p=np.array(o['points']);b=np.array(o['bbox'])
            if p.shape!=(5,2) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>=640) or o['class_id'] not in range(4):errors.append('bad label')
            if b.shape!=(4,) or np.any(b<0) or np.any(b>640):errors.append('bad box')
            if schema==CENTER_SCHEMA and (o['visibility'][2:]!=[0,0,0] or np.any(p[2:]!=0)):errors.append('inactive center slots')
            vis.update(o['visibility'])
    if views!={(e['id'],v) for e in eps for v in VIEWS}:errors.append('missing views')
    for split in ('train','val'):
        for folder,field in (('images','image'),('labels','label')):
            if {p for p in (root/folder/split).rglob('*') if p.is_file()}!={root/r[field] for r in rows if r['split']==split}:errors.append('unlisted files')
    return dict(errors=errors,counts=m['counts'],episode_counts=m['episode_counts'],visibility=dict(vis),classes=dict(Counter(NAMES[o['class_id']] for r in rows for o in r['objects'])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True);p.add_argument('--train-poses',type=int,default=80);p.add_argument('--val-poses',type=int,default=24)
    print(json.dumps(generate(p.parse_args()),indent=2));return 0
