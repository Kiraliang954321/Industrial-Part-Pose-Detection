"""A4-compatible image inspection, returning a fresh recipe-aligned estimate."""
import json
from pathlib import Path
import cv2
from yolo_conveyor.orientation.common import write_json
from yolo_conveyor.pose_cell.scene import VIEWS
from yolo_conveyor.pose_cell.runtime import observe,POLICY as EXECUTION
from yolo_conveyor.pose_cell.geometry import error
from yolo_conveyor.active_pose.policy import POLICY,observation_gate,grip_transform,agreement


def inspect(scene,controller,predictor,folder,target,feature,blind,monitor,record):
    reference=None;anchor=None;accepted=None;reason=None
    while len(record['observations'])<1+POLICY['max_observation_moves']+EXECUTION['max_corrections']:
        i=len(record['observations']);directory=folder/f'observation_{i}'
        obs,panel=observe(scene,predictor,directory,i,target,feature,blind)
        evidence=json.loads((directory/'evidence.json').read_text(encoding='utf-8'))
        images={v:cv2.imread(str(directory/f'{v}.png')) for v in VIEWS}
        gate=observation_gate(images,evidence['predictions'],scene.kind);tool=scene.tool_transform().copy();measured=obs['measurement'];check=None
        if measured is not None:
            relative=grip_transform(measured,tool);check=agreement(reference,relative)
            if not check['consistent']:measured=None;obs['policy_reason']='inconsistent_grip_transform'
            elif reference is None:reference=relative
        obs.update(tool_fk=tool.tolist(),consistency=check,accepted=measured is not None,observation_gate=gate,
                   observation_moves_before=record['observation_moves'],corrections_before=record['corrections'])
        record['observations'].append(obs)
        write_json(directory/'decision.json',{k:obs.get(k) for k in ('accepted','policy_reason','consistency','observation_gate','tool_fk')})
        if monitor.video:monitor.video.snapshot=panel
        if measured is None:
            reason=obs.get('policy_reason') or obs['reason']
            if not gate['allowed']:return None,gate['reason']
            if anchor is None:anchor=tool.copy()
            moved=False
            while len(record['observation_attempts'])<POLICY['max_observation_moves']:
                slot=len(record['observation_attempts']);attempt=dict(slot=slot,executed=False);record['observation_attempts'].append(attempt)
                try:stages=controller.observe_pose(anchor,slot)
                except ValueError as exc:attempt['reason']=str(exc);continue
                attempt['executed']=True;record['observation_moves']+=1
                if monitor.video:monitor.video.message=f'Active observation {slot+1}/2; no insertion permission'
                controller.execute(stages,monitor);moved=True;break
            if moved:continue
            return None,reason
        anchor=None;target_error=error(measured['rotation'],target)
        if monitor.video:monitor.video.message=f'Visual target error: axis {target_error["axis_deg"]:.1f} deg | rotation {target_error["rotation_deg"]:.1f} deg'
        if target_error['rotation_deg']<=EXECUTION['rotation_gate_deg'] and target_error['axis_deg']<=EXECUTION['axis_gate_deg']:
            if float(scene.data.time)-obs['captured_sim_s']>EXECUTION['frame_lifetime_s']:return None,'stale_visual_confirmation'
            return measured,None
        if record['corrections']>=EXECUTION['max_corrections']:return None,'correction_budget'
        stages=controller.correction(measured,target);record['corrections']+=1;controller.execute(stages,monitor)
    return accepted,'observation_budget'
