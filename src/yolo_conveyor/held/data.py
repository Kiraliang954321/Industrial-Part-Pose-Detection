"""Physics-derived held-view supervision and episode-isolated development audit."""

import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import time
import cv2
import numpy as np

from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment,dataset_file,unlinked,KEYPOINTS,utc_now
from yolo_conveyor.datasets.screw_orientation import data_yaml,label_text
from yolo_conveyor.orientation.evaluate import draw_predictions
from .scene import HeldScene,SETUP
from .control import HeldController
from .provenance import snapshot_sources
from .geometry import project

TASK="held_stereo_axis_v1"
VIEWS=("inspect_a","inspect_b")
CASES=("normal","reversed","short","long","tilted","empty")


def fixture_case(index,rng):
    case=CASES[index%len(CASES)]
    offset={"normal":0.,"reversed":0.,"short":.006,"long":-.004,"tilted":0.,"empty":0.}[case]
    yaw=(np.pi if case=="reversed" else 0)+rng.uniform(-.025,.025)
    if case=="tilted":yaw=rng.choice([-1,1])*rng.uniform(.12,.20)
    return {"case":case,"offset_x":float(offset+rng.uniform(-.001,.001)),
            "offset_y":float(rng.uniform(-.0004,.0004)),"yaw":float(yaw),"empty":case=="empty"}


def annotated_view(scene,name,gain):
    mj=scene.mj;option=mj.MjvOption();option.sitegroup[:]=0
    scene.renderer.update_scene(scene.data,camera=name,scene_option=option)
    image=cv2.cvtColor(scene.renderer.render().copy(),cv2.COLOR_RGB2BGR)
    image=np.clip(image.astype(float)*gain,0,255).astype(np.uint8)
    scene.renderer.enable_segmentation_rendering();seg=scene.renderer.render().copy();scene.renderer.disable_segmentation_rendering()
    scene.renderer.enable_depth_rendering();depth=scene.renderer.render().copy();scene.renderer.disable_depth_rendering()
    mask=(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM))&np.isin(seg[:,:,0],scene.screw_geom_ids)
    world=scene.object_anchors();points=scene.project(name,world);camera=scene.camera_info(name)
    objects=[]
    if int(mask.sum())>=50:
        y,x=np.nonzero(mask)
        if np.any(points<0) or np.any(points>=scene.size) or min(x)<1 or min(y)<1 or max(x)>=scene.size-1 or max(y)>=scene.size-1:
            raise ValueError("visible part clipped; camera/envelope must be revised before dataset release")
        transform=np.array(camera["T_world_camera_cv"])
        cp=(world-transform[:3,3])@transform[:3,:3]
        vis=[]
        for pixel,z in zip(points,cp[:,2]):
            px,py=np.round(pixel).astype(int);px=min(px,scene.size-1);py=min(py,scene.size-1)
            vis.append(2 if mask[py,px] and abs(depth[py,px]-z)<.001 else 1)
        objects=[{"class_id":0,"bbox_xyxy":[int(min(x)),int(min(y)),int(max(x))+1,int(max(y))+1],
                  "keypoints_xy":points.tolist(),"visibility":vis,"world_points_m":world.tolist(),"visible_pixels":int(mask.sum())}]
    elif np.linalg.norm(world.mean(axis=0)-scene.tool_transform()[:3,3])<.06:
        raise ValueError("held object not sufficiently visible; do not silently label it as empty")
    return image,objects,camera


def generate(args):
    if not 6<=args.train_episodes<=500 or not 6<=args.val_episodes<=100 or not 1<=args.variants<=12:
        raise ValueError("invalid bounded development dataset budget")
    out=new_directory(args.output,[args.assets]);scene=None;start=time.perf_counter()
    try:
        sources=snapshot_sources(out)
        scene=HeldScene(args.assets);controller=HeldController(scene)
        (out/"scene.xml").write_text(scene.xml,encoding="utf-8")
        write_json(out/"data.yaml",data_yaml(out));write_json(out/"setup.json",SETUP);(out/"previews").mkdir()
        rows=[];episodes=[];image_hashes=set()
        for split,number,salt in (("train",args.train_episodes,0),("val",args.val_episodes,10000)):
            (out/"images"/split).mkdir(parents=True);(out/"labels"/split).mkdir(parents=True)
            for index in range(number):
                seed=args.seed+salt+index;rng=np.random.default_rng(seed);spec=fixture_case(index,rng)
                group=f"h1_{split}_{seed}"
                scene.initialize(rng,spec);controller.execute(controller.plan_pick_and_inspect(),lambda stage:None)
                tool=scene.tool_transform();world=scene.object_anchors()
                ep={"episode_id":group,"split":split,"seed":seed,"fixture":spec,
                    "tool_transform":tool.tolist(),"world_points_m":world.tolist(),"finger_q":scene.data.qpos[scene.finger_qpos].tolist(),
                    "retained_variants":[],"duplicate_variants_omitted":[]}
                episodes.append(ep)
                for variant in range(args.variants):
                    light_seed=[seed,variant,991];appearance=np.random.default_rng(np.random.SeedSequence(light_seed))
                    shade=float(appearance.uniform(.10,.40));scene.model.mat_rgba[scene.model.material("steel").id,:3]=[shade*.97,shade,shade*1.03]
                    light_position=np.array([appearance.uniform(-.25,.15),appearance.uniform(-.20,.20),appearance.uniform(.4,.6)])
                    scene.model.light_pos[0]=light_position
                    direction=np.array([-.1,0,.25])-light_position;scene.model.light_dir[0]=direction/np.linalg.norm(direction)
                    intensity=float(appearance.uniform(.4,1.1));scene.model.light_diffuse[0]=intensity
                    gain=float(appearance.uniform(.85,1.15))
                    paired=[]
                    for view in VIEWS:
                        image,objects,camera=annotated_view(scene,view,gain)
                        ok,encoded=cv2.imencode('.png',image)
                        if not ok:raise RuntimeError('PNG encoding failed')
                        paired.append((view,image,objects,camera,encoded.tobytes(),hashlib.sha256(encoded).hexdigest()))
                    if any(item[-1] in image_hashes for item in paired) or paired[0][-1]==paired[1][-1]:
                        ep['duplicate_variants_omitted'].append(variant)
                        continue
                    ep['retained_variants'].append(variant)
                    for view,image,objects,camera,encoded,image_hash in paired:
                        stem=f"{group}_v{variant:02d}_{view}"
                        image_rel=f"images/{split}/{stem}.png";label_rel=f"labels/{split}/{stem}.txt"
                        with (out/image_rel).open('xb') as stream:stream.write(encoded)
                        image_hashes.add(image_hash)
                        with (out/label_rel).open("x",encoding="utf-8",newline="\n") as stream:stream.write(label_text(objects,scene.size))
                        row={"episode_id":group,"split":split,"view":view,"variant":variant,"image":image_rel,"label":label_rel,
                             "image_sha256":image_hash,"label_sha256":sha256(out/label_rel),"objects":objects,"camera":camera,
                             "tool_transform":tool.tolist(),"appearance":{"seed":light_seed,"shade":shade,"gain":gain,"intensity":intensity,"light_position":light_position.tolist()}}
                        rows.append(row)
                        if index<6 and variant==0:
                            shown=image.copy()
                            for obj in objects:
                                head,tip=[tuple(np.round(p).astype(int)) for p in obj["keypoints_xy"]]
                                cv2.arrowedLine(shown,tip,head,(30,230,40),2,tipLength=.1)
                            cv2.putText(shown,f"GT ONLY | {spec['case']} | {view}",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,230),1)
                            cv2.imwrite(str(out/"previews"/f"{stem}.jpg"),shown)
                if not ep['retained_variants']:raise ValueError('entire episode duplicated')
                if index%10==0 or index==number-1:
                    state={"status":"generating","split":split,"episodes":index+1,"requested":number,"images":len(rows),"seconds":time.perf_counter()-start}
                    write_json(out/"progress.json",state,replace=True);print(json.dumps(state),flush=True)
        with (out/"records.jsonl").open("x",encoding="utf-8",newline="\n") as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+"\n")
        write_json(out/"episodes.json",episodes)
        meta={"schema_version":1,"task":TASK,"scope":"held_simulation_development","status":"completed","real_formal_gate":"closed",
              "arguments":vars(args),"image_size":640,"keypoint_names":KEYPOINTS,"views":list(VIEWS),
              "counts":dict(Counter(r["split"] for r in rows)),"episode_counts":dict(Counter(e["split"] for e in episodes)),
              "records_sha256":sha256(out/"records.jsonl"),"episodes_sha256":sha256(out/"episodes.json"),"yaml_sha256":sha256(out/"data.yaml"),
              "scene_sha256":sha256(out/"scene.xml"),"sources":sources,"environment":environment(),"created_at":utc_now(),
              "seconds":time.perf_counter()-start,"notes":["Physical grasp episodes; post-grasp appearance variants are correlated, not independent trials",
              "Both camera views and every variant of an episode stay in the same split","No real part, thread engagement or six-DoF claim"]}
        write_json(out/"dataset.json",meta)
        audit=audit_dataset(out);write_json(out/"audit.json",audit)
        if audit["errors"]:raise ValueError(str(audit["errors"][:5]))
        write_json(out/"progress.json",{"status":"completed","counts":meta["counts"],"episode_counts":meta["episode_counts"]},replace=True)
        return meta
    except BaseException as exc:
        write_json(out/"progress.json",{"status":"failed","error":str(exc)},replace=True);raise
    finally:
        if scene is not None:scene.close()


def load_dataset(path):
    root=unlinked(path);meta=json.loads(dataset_file(root,"dataset.json").read_text(encoding="utf-8"))
    if meta.get("task")!=TASK or meta.get("status")!="completed" or meta.get("real_formal_gate")!="closed":raise ValueError("invalid held dataset contract")
    for name,key in (("records.jsonl","records_sha256"),("episodes.json","episodes_sha256"),("data.yaml","yaml_sha256"),("scene.xml","scene_sha256")):
        if sha256(dataset_file(root,name))!=meta[key]:raise ValueError(f"dataset hash drift: {name}")
    if json.loads((root/"data.yaml").read_text(encoding="utf-8"))!=data_yaml(root):raise ValueError("unexpected training YAML")
    rows=[json.loads(x) for x in (root/"records.jsonl").read_text(encoding="utf-8").splitlines()]
    return meta,rows


def audit_dataset(path):
    root=unlinked(path);errors=[];counts=Counter();positives=Counter();groups={};seen=set();view_groups=set()
    try:meta,rows=load_dataset(root)
    except (ValueError,OSError,KeyError,TypeError) as exc:return {"errors":[str(exc)]}
    episodes=json.loads((root/"episodes.json").read_text(encoding="utf-8"));episode_map={e["episode_id"]:e for e in episodes}
    if len(episode_map)!=len(episodes):errors.append("duplicate episode IDs")
    for row in rows:
        try:
            split=row["split"];group=row["episode_id"]
            if split not in ("train","val") or group not in episode_map or episode_map[group]["split"]!=split:raise ValueError("bad split/episode")
            if group in groups and groups[group]!=split:raise ValueError("episode leakage")
            groups[group]=split
            marker=(group,row["variant"],row["view"])
            if marker in view_groups or row["view"] not in VIEWS:raise ValueError("duplicate/invalid view")
            view_groups.add(marker)
            ip=dataset_file(root,row["image"]);lp=dataset_file(root,row["label"])
            if not row["image"].startswith(f"images/{split}/") or not row["label"].startswith(f"labels/{split}/"):raise ValueError("split path mismatch")
            if sha256(ip)!=row["image_sha256"] or sha256(lp)!=row["label_sha256"]:raise ValueError("image/label hash drift")
            if row["image_sha256"] in seen:raise ValueError("duplicate image bytes")
            seen.add(row["image_sha256"])
            image=cv2.imread(str(ip))
            if image is None or image.shape!=(640,640,3):raise ValueError("bad image shape")
            if lp.read_text(encoding="utf-8")!=label_text(row["objects"],640):raise ValueError("label/metadata mismatch")
            for obj in row["objects"]:
                points=np.array(obj["keypoints_xy"]);box=np.array(obj["bbox_xyxy"])
                if points.shape!=(2,2) or not np.isfinite(points).all() or np.any(points<0) or np.any(points>=640):raise ValueError("bad points")
                if box.shape!=(4,) or np.any(box<0) or np.any(box>640) or box[2]<=box[0] or box[3]<=box[1]:raise ValueError("bad box")
                if not np.allclose([project(row["camera"],p) for p in obj["world_points_m"]],points,atol=1e-5):raise ValueError("reprojection mismatch")
                positives[split]+=1
            counts[split]+=1
        except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:errors.append(f"{row.get('image','?')}: {exc}")
    if dict(counts)!=meta["counts"] or dict(Counter(groups.values()))!=meta["episode_counts"]:errors.append("count mismatch")
    expected_views={(e['episode_id'],variant,view) for e in episodes for variant in e['retained_variants'] for view in VIEWS}
    if view_groups!=expected_views:errors.append("missing paired variants")
    if any(not e['retained_variants'] for e in episodes):errors.append('empty episode')
    for split in ("train","val"):
        if set((root/"images"/split).glob('*.png'))!={root/r["image"] for r in rows if r["split"]==split}:errors.append("unlisted images")
        if set((root/"labels"/split).glob('*.txt'))!={root/r["label"] for r in rows if r["split"]==split}:errors.append("unlisted labels")
    return {"errors":errors,"counts":dict(counts),"positive_images":dict(positives),"episode_counts":dict(Counter(groups.values())),
            "independence":"episode-level split; views/appearance variants are not independent samples"}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-episodes',type=int,default=100);p.add_argument('--val-episodes',type=int,default=20)
    p.add_argument('--variants',type=int,default=8);p.add_argument('--seed',type=int,default=41000)
    m=generate(p.parse_args());print(json.dumps({'counts':m['counts'],'episodes':m['episode_counts'],'seconds':m['seconds']}));return 0
