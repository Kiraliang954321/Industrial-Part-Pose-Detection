"""Image-driven correction, re-observation and guarded physical pre-positioning."""
import argparse
import json
import os
from pathlib import Path
from copy import deepcopy
import time
import cv2
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json,sha256,environment
from yolo_conveyor.simulation.panda_control import Stage
from yolo_conveyor.datum_pose.crop import crop_image
from .scene import CellScene,VIEWS,SETUP,CENTER
from .control import CellController
from .faces import PARTS
from .geometry import rz,error
from .predictor import Predictor
from .metrology import measure,LIMITS
from .evaluation import PhysicalScore,truth,summarize
from .recording import Recorder,draw
from .provenance import snapshot,sources

CASES=('upright','tilt60','side90','tilt120','inverted180','reverse_grasp','blind','plain')
ANGLES={'upright':0,'tilt60':60,'side90':90,'tilt120':120,'inverted180':180,'reverse_grasp':0,'blind':90,'plain':120}
TARGETS=[0.,30.,60.]
POLICY=dict(max_corrections=3,rotation_gate_deg=7.,axis_gate_deg=4.,gt_rotation_deg=10.,gt_axis_deg=5.,
            grip_width_change_m=.0008,inspection_center_radius_m=.025,frame_lifetime_s=3.)


def hold(s,c,seconds,observer):c.execute([Stage('hold',s.data.qpos[s.arm_qpos].copy(),c.close_width,seconds)],observer)


def observe(s,predictor,directory,index,target,feature,blind=False):
    directory.mkdir();images={v:s.capture(v) for v in VIEWS};cameras={v:s.camera_info(v) for v in VIEWS};captured=float(s.data.time)
    if blind:
        for image in images.values():image[:]=0
    predictor.sync();start=time.perf_counter();predictions=predictor.predict(images);predictor.sync();network_ms=1000*(time.perf_counter()-start)
    measured=None;reason=None;estimate_start=time.perf_counter()
    try:
        # Use exactly the same pixels, scale and edge extraction as development
        # validation. Network outputs are retained in raw coordinates for overlays.
        geometry_images={v:crop_image(im) for v,im in images.items()}
        geometry_predictions=deepcopy(predictions);geometry_cameras=deepcopy(cameras)
        for camera in geometry_cameras.values():camera['K']=(np.array([[2,0,-320],[0,2,-320],[0,0,1.]])@np.array(camera['K'])).tolist()
        for items in geometry_predictions.values():
            for p in items:
                p['bbox']=(np.array(p['bbox'])*2-320).tolist();points=np.array(p['keypoints']);points[:,:2]=points[:,:2]*2-320;p['keypoints']=points.tolist()
        measured=measure(geometry_images,geometry_predictions,geometry_cameras,s.kind,pixel_scale=2.)
        if np.linalg.norm(np.asarray(measured['translation_m'])-CENTER)>POLICY['inspection_center_radius_m']:raise ValueError('outside_inspection_volume')
    except (ValueError,np.linalg.LinAlgError) as exc:measured=None;reason=str(exc)
    # Independent object-state reads are deliberately after inference and never passed back.
    gt=truth(s,target);e=error(measured['rotation'],gt['rotation']) if measured and feature else None
    record=dict(index=index,captured_sim_s=captured,measurement=measured,reason=reason,gt=gt,error=e,
                position_error_mm=float(np.linalg.norm(np.asarray(measured['translation_m'])-gt['center'])*1000) if measured and feature else None,
                neural_ms=network_ms,geometry_ms=1000*(time.perf_counter()-estimate_start))
    for v in VIEWS:
        cv2.imwrite(str(directory/f'{v}.png'),images[v]);cv2.imwrite(str(directory/f'{v}_pose.jpg'),draw(images[v],predictions[v],cameras[v],measured))
    write_json(directory/'measurement.json',record)
    write_json(directory/'evidence.json',dict(predictions=predictions,cameras=cameras,geometry_preprocessing='fixed_roi_2x',input_hashes={v:sha256(directory/f'{v}.png') for v in VIEWS}))
    return record,draw(images[VIEWS[0]],predictions[VIEWS[0]],cameras[VIEWS[0]],measured)


def episode(s,c,predictor,directory,index,seed,case,target_spin,record):
    rng=np.random.default_rng(seed);target=rz(target_spin);spin=target_spin+float(rng.uniform(12,25));feature=case!='plain';flipped=case=='reverse_grasp'
    wall=time.perf_counter();s.initialize_pose(rng,spin,flipped);video=Recorder(s,directory/'physics.mp4',f'{s.kind} | {case}') if record else None
    score=PhysicalScore(s);trace=[];finger_reference=None;observations=[];corrections=0;inserted=False;prealigned=False;status='not_started';failure=None
    def observer(stage):
        score.observe()
        if video:video.sample(stage)
        fingers=s.data.qpos[s.finger_qpos]
        if not np.isfinite(fingers).all():raise ValueError('invalid_gripper_feedback')
        if s.illegal_contacts():raise ValueError('robot_fixed_environment_collision')
        if finger_reference is not None and np.max(np.abs(fingers-finger_reference))>POLICY['grip_width_change_m']:raise ValueError('grip_changed')
        if int(round(s.data.time*1000))%20==0:trace.append(dict(time=float(s.data.time),stage=stage,fingers=fingers.tolist()))
    try:
        try:
            c.execute(c.pick(),observer);finger_reference=s.data.qpos[s.finger_qpos].copy()
            if np.min(finger_reference)<c.close_width+.0004:raise ValueError('no_secure_grip')
            c.execute(c.present(ANGLES[case]),observer)
            for iteration in range(POLICY['max_corrections']+1):
                obs,panel=observe(s,predictor,directory/f'observation_{iteration}',iteration,target,feature,case=='blind');observations.append(obs)
                if video:video.snapshot=panel
                measured=obs['measurement']
                if measured is None:status='UNCERTAIN';failure=obs['reason'];break
                target_error=error(measured['rotation'],target)
                if video:video.message=f"RGB target error: axis {target_error['axis_deg']:.1f} deg | full rotation {target_error['rotation_deg']:.1f} deg"
                if target_error['rotation_deg']<=POLICY['rotation_gate_deg'] and target_error['axis_deg']<=POLICY['axis_gate_deg']:
                    if float(s.data.time)-obs['captured_sim_s']>POLICY['frame_lifetime_s']:raise ValueError('stale_verification')
                    # GT only records whether permission was correct; it cannot veto or authorize it.
                    g=obs['gt']['target_error'];prealigned=bool(feature and obs['gt']['held'] and g['rotation_deg']<=POLICY['gt_rotation_deg'] and g['axis_deg']<=POLICY['gt_axis_deg'])
                    trajectory=c.insert(measured,target);inserted=True;c.execute(trajectory,observer);status='EXECUTED';break
                if iteration>=POLICY['max_corrections']:status='CORRECTION_LIMIT';break
                trajectory=c.correction(measured,target);corrections+=1;c.execute(trajectory,observer)
        except ValueError as exc:status='ABORT';failure=str(exc)
        finger_reference=None;hold(s,c,.5,lambda stage:score.observe())
        final=truth(s,target);physical=score.result(inserted,failure);aligned=bool(feature and final['held'] and final['target_error']['rotation_deg']<=POLICY['gt_rotation_deg'] and final['target_error']['axis_deg']<=POLICY['gt_axis_deg'])
        success=bool(feature and inserted and physical['success'] and aligned)
        if status=='EXECUTED':status='SUCCESS' if success else 'PHYSICAL_CHECK_FAILED'
        row=dict(index=index,seed=seed,kind=s.kind,case=case,feature=feature,initial_flip=flipped,presentation_deg=ANGLES[case],target_spin_deg=target_spin,
                 observations=observations,corrections=corrections,insert_attempted=inserted,preinsert_gt_aligned=prealigned,
                 final_gt=final,physical=physical,aligned_gt=aligned,success=success,status=status,failure=failure,wall_s=time.perf_counter()-wall)
        write_json(directory/'result.json',row);write_json(directory/'trace.json',trace);cv2.imwrite(str(directory/'final.png'),s.capture('assembly_view'))
        if video:video.message=status+(' | '+failure if failure else '');video.next=0;video.sample('final')
        return row
    finally:
        if video:video.close()


def run(args):
    protocol=None
    if args.mode=='evaluate':
        from .protocol import validate
        protocol=validate(args.protocol,args)
    elif not 201000<=args.seed<202000-124:raise ValueError('calibration seed reservation')
    out=new_directory(args.output,[args.assets,args.weights]);os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config');rows=[];scene=None
    try:
        predictor=Predictor(args.weights);predictor.predict({v:np.zeros((640,640,3),np.uint8) for v in VIEWS})
        manifest=dict(scope='A3_SIMULATION_CLOSED_LOOP',mode=args.mode,real_formal_gate='closed',arguments=vars(args),sources=snapshot(out),setup=SETUP,limits=LIMITS,policy=POLICY,
                      weights_sha256=sha256(Path(args.weights)),asset_sha256=sha256(Path(args.assets)/'asset-manifest.json'),environment=environment(),cases=list(CASES),targets=TARGETS,
                      protocol_sha256=sha256(Path(args.protocol)) if protocol is not None else None)
        write_json(out/'manifest.json',manifest);recorded={int(x) for x in args.record_indices.split(',') if x.strip()}
        for k,kind in enumerate(PARTS):
            for feature in (True,False):
                scene=CellScene(args.assets,kind,feature);controller=CellController(scene);(out/f'{kind}-{feature}-scene.xml').write_text(scene.xml,encoding='utf-8')
                for i in range(len(CASES)*len(TARGETS)):
                    case=CASES[i%len(CASES)]
                    if feature!=(case!='plain'):continue
                    index=k*24+i;seed=args.seed+k*100+i;directory=out/f'{index:03d}_{kind}_{case}_{seed}';directory.mkdir()
                    row=episode(scene,controller,predictor,directory,index,seed,case,TARGETS[i//len(CASES)],index in recorded);rows.append(row)
                    print(json.dumps(dict(index=index,kind=kind,case=case,status=row['status'],corrections=row['corrections'],success=row['success'],failure=row['failure'])),flush=True)
                    write_json(out/'progress.json',dict(status='running',completed=len(rows),requested=48),replace=True)
                scene.close();scene=None
        if sources()!=manifest['sources'] or sha256(Path(args.weights))!=manifest['weights_sha256']:raise ValueError('source or weight drift')
        if protocol is not None:
            validate(args.protocol,args)
            if sha256(Path(args.protocol))!=manifest['protocol_sha256']:raise ValueError('protocol drift')
        rows.sort(key=lambda r:r['index']);report=dict(status='completed',mode=args.mode,scope=manifest['scope'],real_formal_gate='closed',overall=summarize(rows),
                    by_part={k:summarize([r for r in rows if r['kind']==k]) for k in PARTS},by_case={c:summarize([r for r in rows if r['case']==c]) for c in CASES})
        write_json(out/'results.json',rows);write_json(out/'summary.json',report);write_json(out/'progress.json',dict(status='completed',episodes=len(rows)),replace=True)
        write_json(out/'artifact-hashes.json',{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()});return report
    except BaseException as exc:
        write_json(out/'progress.json',dict(status='failed',completed=len(rows),error=str(exc)),replace=True);raise
    finally:
        if scene is not None:scene.close()


def main():
    p=argparse.ArgumentParser()
    for n in ('assets','weights','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--mode',choices=('calibration','evaluate'),default='calibration');p.add_argument('--protocol');p.add_argument('--seed',type=int,default=201000);p.add_argument('--record-indices',default='')
    print(json.dumps(run(p.parse_args())['overall'],indent=2));return 0
