"""Two-stage visual decisions and typed-purpose one-shot motion permits."""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from yolo_conveyor.held.gate import inspect as initial_gate, POLICY as H1_POLICY, rotation_error
from yolo_conveyor.held.geometry import stereo_estimate,observable_measures
from yolo_conveyor.held.scene import GUIDE_XY,GUIDE_TOP

POLICY = {**H1_POLICY, 'recheck_frame_age_s': .25, 'stereo_skew_s': .02,
          'max_relative_shift_m': .0015, 'max_relative_axis_deg': 2.,
          'gt_relative_shift_m': .002, 'gt_relative_axis_deg': 3.,
          'insert_permit_lifetime_s': 2., 'prealign_max_radial_m': .002,
          'prealign_gap_min_m': .010, 'prealign_gap_max_m': .020}


def tool_relative(estimate,tool):
    tool=np.asarray(tool,float)
    return dict(tip=(tool[:3,:3].T@(np.array(estimate['tip_world_m'])-tool[:3,3])).tolist(),
                axis=(tool[:3,:3].T@np.array(estimate['axis_world'])).tolist())


def relative_change(first,second):
    return dict(shift_m=float(np.linalg.norm(np.subtract(first['tip'],second['tip']))),
                axis_deg=float(np.degrees(np.arccos(np.clip(np.dot(first['axis'],second['axis']),-1,1)))))


def first_decision(observation):
    return initial_gate(observation['images'],observation['corrected'],observation['cameras'],
                        observation['tool'],observation['fingers'])


def recheck(observation,first_estimate,first_tool,now):
    estimate=None;change=None
    try:
        tool=np.asarray(observation['tool']);fingers=np.asarray(observation['fingers'])
        times=np.asarray(observation['capture_times'],float)
        if times.shape!=(2,) or not np.isfinite(times).all() or not np.isfinite(now):raise ValueError('invalid_time')
        if np.ptp(times)>POLICY['stereo_skew_s'] or not np.all((now-times>=0)&(now-times<=POLICY['recheck_frame_age_s'])):
            raise ValueError('stale_or_unsynchronized_recheck')
        if tool.shape!=(4,4) or fingers.shape!=(2,) or not np.isfinite(tool).all() or not np.isfinite(fingers).all():
            raise ValueError('invalid_robot_feedback')
        if np.any(fingers<POLICY['finger_min_m']) or np.any(fingers>POLICY['finger_max_m']):raise ValueError('gripper_width_invalid')
        if np.linalg.norm(tool[:2,3]-GUIDE_XY)>.015 or not .14<=tool[2,3]<=.18:raise ValueError('outside_preinsert_station')
        for name,image in observation['images'].items():
            roi=image[200:440,200:440]
            if image.shape!=(640,640,3) or roi.mean()<5 or roi.std()<3:raise ValueError('image_quality')
            if any(p['confidence']<POLICY['box_confidence'] for p in observation['corrected'][name]):raise ValueError('unreliable_prediction')
        mapped={f'inspect_{suffix}':observation['corrected'][f'recheck_{suffix}'] for suffix in ('a','b')}
        cameras={f'inspect_{suffix}':observation['cameras'][f'recheck_{suffix}'] for suffix in ('a','b')}
        estimate=stereo_estimate(mapped,cameras,POLICY['point_confidence'])
        if estimate['ray_gap_m']>POLICY['max_ray_gap_m'] or estimate['reprojection_error_px']>POLICY['max_reprojection_px']:
            raise ValueError('stereo_inconsistent')
        change=relative_change(tool_relative(first_estimate,first_tool),tool_relative(estimate,tool))
        if change['shift_m']>POLICY['max_relative_shift_m'] or change['axis_deg']>POLICY['max_relative_axis_deg']:
            return dict(decision='REJECT',reason='relative_pose_changed',estimate=estimate,change=change)
        measures=observable_measures(estimate,tool);tip=np.array(estimate['tip_world_m'])
        if not (measures['tilt_deg']<=POLICY['final_max_tilt_deg'] and
                POLICY['protrusion_min_m']<=measures['tip_protrusion_m']<=POLICY['protrusion_max_m'] and
                np.linalg.norm(tip[:2]-GUIDE_XY)<=POLICY['prealign_max_radial_m'] and
                POLICY['prealign_gap_min_m']<=tip[2]-GUIDE_TOP<=POLICY['prealign_gap_max_m']):
            raise ValueError('outside_preinsert_correction_envelope')
        return dict(decision='PASS',reason='fresh_stable_grasp',estimate=estimate,change=change)
    except (ValueError,KeyError,TypeError,IndexError,np.linalg.LinAlgError) as exc:
        return dict(decision='UNCERTAIN',reason=str(exc),estimate=estimate,change=change)


@dataclass
class MotionPermit:
    purpose:str
    grasp_id:str
    epoch:int
    tool_id:str
    issued_s:float
    lifetime_s:float
    tool:np.ndarray
    fingers:np.ndarray
    estimate:dict
    evidence_id:str
    used:bool=False

    @classmethod
    def issue(cls,purpose,grasp_id,epoch,tool_id,decision,observation,now):
        if purpose not in ('transfer','insert') or decision['decision']!='PASS':raise ValueError('permit_not_authorized')
        expected={'recheck_a','recheck_b'} if purpose=='insert' else {'inspect_a','inspect_b'}
        expected_reason='fresh_stable_grasp' if purpose=='insert' else 'measurable_correctable_held_part'
        if set(observation['images'])!=expected or decision['reason']!=expected_reason:
            raise ValueError('permit_requires_matching_inspection_stage')
        times=np.asarray(observation['capture_times'],float)
        if times.shape!=(2,) or not np.isfinite(times).all() or not np.isfinite(now):raise ValueError('invalid_evidence_time')
        if np.ptp(times)>POLICY['stereo_skew_s'] or not np.all((now-times>=0)&(now-times<=POLICY['recheck_frame_age_s'])):
            raise ValueError('cannot_issue_from_stale_evidence')
        lifetime=POLICY['insert_permit_lifetime_s'] if purpose=='insert' else POLICY['permit_lifetime_s']
        return cls(purpose,grasp_id,epoch,tool_id,float(min(times)),lifetime,np.array(observation['tool']).copy(),
                   np.array(observation['fingers']).copy(),deepcopy(decision['estimate']),observation['evidence_id'])

    def consume(self,purpose,grasp_id,epoch,tool_id,now,tool,fingers,command,evidence_id):
        if self.used:raise ValueError('permit_already_used')
        self.used=True
        if (purpose,grasp_id,epoch,tool_id,evidence_id)!=(self.purpose,self.grasp_id,self.epoch,self.tool_id,self.evidence_id):
            raise ValueError('permit_context_mismatch')
        if not np.isfinite(now) or not 0<=now-self.issued_s<=self.lifetime_s:raise ValueError('permit_expired')
        tool=np.asarray(tool);fingers=np.asarray(fingers)
        if tool.shape!=(4,4) or fingers.shape!=(2,) or not np.isfinite(tool).all() or not np.isfinite(fingers).all() or not np.isfinite(command):
            raise ValueError('invalid_robot_feedback')
        if command>1e-6 or np.max(np.abs(fingers-self.fingers))>POLICY['finger_change_m']:raise ValueError('grasp_changed')
        if np.linalg.norm(tool[:3,3]-self.tool[:3,3])>.001 or rotation_error(tool[:3,:3],self.tool[:3,:3])>1:
            raise ValueError('permit_pose_stale')
        return deepcopy(self.estimate)
