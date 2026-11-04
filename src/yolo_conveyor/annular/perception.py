"""Six-class RGB gate for a known job recipe; no simulator truth inputs."""
import numpy as np
from .assets import NAMES
from .training import verify_model

POLICY=dict(confidence=.80,min_finger_m=.004,max_finger_m=.018,finger_change_m=.001,
            evidence_age_s=5.,final_tilt_deg=5.,engagement_tolerance_m=.0012,
            radial_margin_m=.0001,max_penetration_m=.0005)


class Detector:
    def __init__(self,weights):
        from ultralytics import YOLO
        import torch
        self.model=YOLO(str(weights));verify_model(self.model);self.torch=torch

    def predict(self,image):
        result=self.model.predict(image,imgsz=640,device='0',conf=.2,verbose=False)[0]
        return [dict(class_id=int(cls),name=NAMES[int(cls)],confidence=float(conf),bbox_xyxy=box)
                for box,conf,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]

    def sync(self):
        self.torch.cuda.synchronize()


def decide(image,predictions,kind,fingers):
    fingers=np.asarray(fingers)
    if fingers.shape!=(2,) or not np.isfinite(fingers).all():return dict(decision='UNCERTAIN',reason='invalid_gripper_feedback')
    if np.any(fingers<POLICY['min_finger_m']) or np.any(fingers>POLICY['max_finger_m']):return dict(decision='UNCERTAIN',reason='no_valid_grip')
    roi=image[200:440,200:440]
    if image.shape!=(640,640,3) or roi.mean()<5 or roi.std()<3:return dict(decision='UNCERTAIN',reason='image_quality')
    if len(predictions)!=1:return dict(decision='UNCERTAIN',reason='no_unique_part')
    p=predictions[0];box=np.asarray(p['bbox_xyxy']);center=(box[:2]+box[2:])/2
    if not np.isfinite(box).all() or not np.isfinite(p['confidence']) or p['confidence']<POLICY['confidence']:
        return dict(decision='UNCERTAIN',reason='low_confidence')
    if np.any(center<200) or np.any(center>440) or box[2]-box[0]<25 or box[3]-box[1]<15:
        return dict(decision='UNCERTAIN',reason='outside_held_roi')
    if p['name'].split('_')[0]!=kind:return dict(decision='REJECT',reason='wrong_part_type')
    if p['name'].endswith('_back'):return dict(decision='REJECT',reason='wrong_face')
    if p['name'].endswith('_tilted'):return dict(decision='REJECT',reason='obvious_tilt')
    return dict(decision='PASS',reason='correct_part_and_face')
