"""Insertion guard based only on simulated wrist sensor and robot encoders."""
import numpy as np
from yolo_conveyor.pose_cell.runtime import POLICY as PREVIOUS

GUARD=dict(force_delta_n=2.,torque_delta_nm=.05,hard_force_delta_n=5.,consecutive_samples=3,
           finger_change_m=PREVIOUS['grip_width_change_m'],tare_seconds=.25)


class ContactStop(ValueError):pass


class Monitor:
    def __init__(self,scene,score,video=None):
        self.scene=scene;self.score=score;self.video=video;self.fingers=None;self.baseline=None;self.armed=False;self.trace=[];self.window=[]
        self.streak=0;self.max_force=0.;self.max_torque=0.;self.trigger=None

    def __call__(self,stage):
        s=self.scene;self.score.observe();w=s.wrist_wrench();f=s.data.qpos[s.finger_qpos]
        if not np.isfinite(w).all() or not np.isfinite(f).all():raise ValueError('invalid_robot_feedback')
        self.window.append((float(s.data.time),w.copy()))
        self.window=[p for p in self.window if p[0]>=s.data.time-GUARD['tare_seconds']]
        force=torque=0.
        if self.baseline is not None:
            delta=w-self.baseline;force=float(np.linalg.norm(delta[:3]));torque=float(np.linalg.norm(delta[3:]))
            if self.armed:
                self.max_force=max(self.max_force,force);self.max_torque=max(self.max_torque,torque)
                self.streak=self.streak+1 if force>GUARD['force_delta_n'] or torque>GUARD['torque_delta_nm'] else 0
        row=dict(time=float(s.data.time),stage=stage,wrench=w.tolist(),force_delta_n=force,torque_delta_nm=torque,fingers=f.tolist(),guard_armed=self.armed)
        if int(round(s.data.time*1000))%10==0:self.trace.append(row)
        if self.video:
            self.video.force=force;self.video.torque=torque;self.video.guard_armed=self.armed;self.video.sample(stage)
        if self.armed and (force>GUARD['hard_force_delta_n'] or self.streak>=GUARD['consecutive_samples']):
            self.trigger=dict(time=float(s.data.time),stage=stage,force_delta_n=force,torque_delta_nm=torque)
            self.trace.append({**row,'event':'CONTACT_STOP'})
            raise ContactStop('wrist_contact_limit')
        if s.illegal_contacts():raise ValueError('robot_fixed_environment_collision')
        if self.fingers is not None and np.max(np.abs(f-self.fingers))>GUARD['finger_change_m']:raise ValueError('grip_changed')

    def tare(self):
        if len(self.window)<50:raise ValueError('insufficient_wrench_tare')
        self.baseline=np.mean([w for _,w in self.window],axis=0);self.streak=0;self.armed=True
        return self.baseline.tolist()

    def disarm(self):self.armed=False;self.streak=0

    def report(self):return dict(max_force_delta_n=self.max_force,max_torque_delta_nm=self.max_torque,trigger=self.trigger)
