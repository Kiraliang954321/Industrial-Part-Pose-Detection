"""Image/robot-feedback gates; construction flags and object truth are forbidden inputs."""
import numpy as np
from yolo_conveyor.orientation.geometry import bbox_iou
from yolo_conveyor.pose_cell.geometry import error
from .control import OBSERVATION_ANGLES

POLICY=dict(max_observation_moves=2,observation_angles_deg=list(OBSERVATION_ANGLES),
            detection_confidence=.5,min_target_views=2,min_image_std=3.,
            relative_rotation_tolerance_deg=8.,relative_translation_tolerance_m=.0015)


def observation_gate(images,predictions,kind):
    base=('bearing','gear').index(kind)*2;usable=[];failures=[]
    for name,image in images.items():
        if image[180:460,180:460].std()<POLICY['min_image_std']:continue
        items=sorted(predictions[name],key=lambda p:p['confidence'],reverse=True)
        if not items or items[0]['confidence']<POLICY['detection_confidence']:continue
        if items[0]['class_id'] not in (base,base+1):failures.append('wrong_part_type');continue
        if any(bbox_iou(items[0]['bbox'],p['bbox'])<.6 for p in items[1:] if p['confidence']>.35):failures.append('multiple_objects');continue
        box=np.array(items[0]['bbox']);center=(box[:2]+box[2:])/2
        if not np.isfinite(box).all() or np.any(center<180) or np.any(center>460):continue
        usable.append(name)
    reason=failures[0] if failures else ('insufficient_single_object_views' if len(usable)<POLICY['min_target_views'] else None)
    return dict(allowed=reason is None,reason=reason,views=usable)


def grip_transform(measurement,tool):
    tool=np.asarray(tool);r=np.asarray(measurement['rotation']);t=np.asarray(measurement['translation_m'])
    return dict(rotation=(tool[:3,:3].T@r).tolist(),translation_m=(tool[:3,:3].T@(t-tool[:3,3])).tolist())


def agreement(previous,current):
    if previous is None:return dict(consistent=True,rotation_deg=None,translation_m=None)
    angle=error(previous['rotation'],current['rotation'])['rotation_deg']
    distance=float(np.linalg.norm(np.subtract(previous['translation_m'],current['translation_m'])))
    return dict(consistent=angle<=POLICY['relative_rotation_tolerance_deg'] and distance<=POLICY['relative_translation_tolerance_m'],rotation_deg=angle,translation_m=distance)
