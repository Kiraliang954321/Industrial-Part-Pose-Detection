"""Image/encoder-only inspection and a single-use, grasp-bound simulation permit."""
from dataclasses import dataclass
from copy import deepcopy
import math
import numpy as np
from .geometry import stereo_estimate,observable_measures
from .scene import INSPECT_TCP,INSPECT_R

POLICY={"box_confidence":.6,"point_confidence":.5,"max_ray_gap_m":.0015,
        "max_reprojection_px":3.,"max_tilt_deg":7.5,"protrusion_min_m":.020,"protrusion_max_m":.028,
        "max_lateral_m":.004,"finger_min_m":.001,"finger_max_m":.005,
        "finger_change_m":.0003,"permit_lifetime_s":30.,"max_force_delta_n":2.,"max_torque_delta_nm":.2,
        "inspection_position_tolerance_m":.001,"inspection_angle_tolerance_deg":1.,
        "gt_max_tilt_deg":8.,"gt_protrusion_min_m":.019,"gt_protrusion_max_m":.029,
        "final_max_tilt_deg":3.,"final_radial_m":.001,"final_depth_tolerance_m":.0015,
        "max_penetration_m":.0005}


def rotation_error(a,b):
    return math.degrees(math.acos(float(np.clip((np.trace(np.asarray(a).T@np.asarray(b))-1)/2,-1,1))))


def inspect(images,predictions,cameras,tool,fingers):
    """No simulator object pose, segmentation, depth or GT is accepted."""
    try:
        tool=np.asarray(tool,float);fingers=np.asarray(fingers,float)
        if tool.shape!=(4,4) or fingers.shape!=(2,) or not np.isfinite(tool).all() or not np.isfinite(fingers).all():
            raise ValueError('invalid_robot_feedback')
        if np.linalg.norm(tool[:3,3]-INSPECT_TCP)>POLICY['inspection_position_tolerance_m'] or rotation_error(tool[:3,:3],INSPECT_R)>POLICY['inspection_angle_tolerance_deg']:
            raise ValueError('outside_inspection_pose')
        if np.any(fingers<POLICY['finger_min_m']) or np.any(fingers>POLICY['finger_max_m']):
            raise ValueError('gripper_width_invalid')
        for name in ('inspect_a','inspect_b'):
            image=np.asarray(images[name]);roi=image[200:440,200:440]
            if image.shape!=(640,640,3) or not np.isfinite(image).all() or roi.mean()<5 or roi.std()<3:
                raise ValueError('image_quality')
            for prediction in predictions[name]:
                values=[prediction['confidence'],*prediction['keypoint_confidences'],*np.asarray(prediction['keypoints_xy']).ravel()]
                if not np.isfinite(values).all() or prediction['confidence']<POLICY['box_confidence']:
                    raise ValueError('unreliable_prediction')
        estimate=stereo_estimate(predictions,cameras,POLICY['point_confidence'])
        measures=observable_measures(estimate,tool)
        if not np.isfinite(list(measures.values())).all():raise ValueError('nonfinite_geometry')
        if estimate['ray_gap_m']>POLICY['max_ray_gap_m'] or estimate['reprojection_error_px']>POLICY['max_reprojection_px']:
            raise ValueError('stereo_inconsistent')
        if measures['tilt_deg']>90:return dict(decision='REJECT',reason='head_down',estimate=estimate,measures=measures)
        if measures['tip_protrusion_m']<POLICY['gt_protrusion_min_m']:
            return dict(decision='REJECT',reason='insufficient_protrusion',estimate=estimate,measures=measures)
        if not (measures['tilt_deg']<=POLICY['max_tilt_deg'] and
                POLICY['protrusion_min_m']<=measures['tip_protrusion_m']<=POLICY['protrusion_max_m'] and
                measures['tip_lateral_offset_m']<=POLICY['max_lateral_m']):
            return dict(decision='UNCERTAIN',reason='outside_correction_envelope',estimate=estimate,measures=measures)
        return dict(decision='PASS',reason='measurable_correctable_held_part',estimate=estimate,measures=measures)
    except (ValueError,KeyError,TypeError,IndexError,np.linalg.LinAlgError) as exc:
        return dict(decision='UNCERTAIN',reason=str(exc),estimate=None,measures=None)


@dataclass
class Permit:
    grasp_id:str
    epoch:int
    tool_id:str
    issued_s:float
    tool:np.ndarray
    fingers:np.ndarray
    estimate:dict
    used:bool=False
    invalid_reason:str|None=None

    @classmethod
    def issue(cls,decision,grasp_id,epoch,tool_id,now,tool,fingers):
        if decision['decision']!='PASS':raise ValueError('inspection did not pass')
        return cls(grasp_id,epoch,tool_id,float(now),np.array(tool).copy(),np.array(fingers).copy(),deepcopy(decision['estimate']))

    def invalidate(self,reason):
        self.invalid_reason=reason

    def consume(self,grasp_id,epoch,tool_id,now,tool,fingers,gripper_command):
        if self.used or self.invalid_reason:raise ValueError('permit_used_or_invalidated')
        self.used=True
        if (grasp_id,epoch,tool_id)!=(self.grasp_id,self.epoch,self.tool_id):raise ValueError('grasp_or_tool_changed')
        if not np.isfinite(now) or not 0<=now-self.issued_s<=POLICY['permit_lifetime_s']:raise ValueError('permit_expired')
        tool=np.asarray(tool);fingers=np.asarray(fingers)
        if (tool.shape!=(4,4) or fingers.shape!=(2,) or not np.isfinite(tool).all() or not np.isfinite(fingers).all()
                or not np.isfinite(gripper_command)):raise ValueError('invalid_feedback')
        if gripper_command>1e-6 or np.max(np.abs(fingers-self.fingers))>POLICY['finger_change_m']:raise ValueError('grasp_changed')
        if np.linalg.norm(tool[:3,3]-self.tool[:3,3])>.001 or rotation_error(tool[:3,:3],self.tool[:3,:3])>1:
            raise ValueError('inspection_pose_stale')
        return self.estimate
