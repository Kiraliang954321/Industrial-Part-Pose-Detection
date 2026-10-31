"""Joint-feedback IK/trajectory control. Never reads object poses or contact GT."""

from dataclasses import dataclass
import math
import numpy as np

from .grasp_geometry import downward_rotation
from .panda_scene import GRASP_Z, GOAL_XY


@dataclass
class Stage:
    name: str
    q: np.ndarray
    gripper: float
    duration: float


class PandaController:
    def __init__(self, scene):
        self.scene = scene
        self.mj, self.m = scene.mj, scene.model
        self.ik_data = self.mj.MjData(self.m)

    def inverse_kinematics(self, position, rotation, initial):
        s, d = self.scene, self.ik_data
        d.qpos[:] = self.m.qpos0
        d.qpos[s.finger_qpos] = 0.04
        q = np.asarray(initial).copy()
        target_quat = np.zeros(4)
        self.mj.mju_mat2Quat(target_quat, np.asarray(rotation).ravel())
        for _ in range(240):
            d.qpos[s.arm_qpos] = q
            self.mj.mj_forward(self.m, d)
            dp = np.asarray(position) - d.site_xpos[s.tcp_id]
            current = np.zeros(4); delta = np.zeros(4); angular = np.zeros(3)
            self.mj.mju_mat2Quat(current, d.site_xmat[s.tcp_id])
            inverse = current * [1, -1, -1, -1]
            self.mj.mju_mulQuat(delta, target_quat, inverse)
            self.mj.mju_quat2Vel(angular, delta, 1.0)
            if np.linalg.norm(dp) < 0.0003 and np.linalg.norm(angular) < 0.008:
                return q
            jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
            self.mj.mj_jacSite(self.m, d, jp, jr, s.tcp_id)
            j = np.vstack([jp[:, s.arm_dofs], 0.15 * jr[:, s.arm_dofs]])
            error = np.concatenate([dp, 0.15 * angular])
            dq = j.T @ np.linalg.solve(j @ j.T + 0.002**2 * np.eye(6), error)
            maximum = max(np.max(np.abs(dq)) / 0.12, 1)
            q = np.clip(q + dq / maximum, s.arm_ranges[:, 0] + 0.015, s.arm_ranges[:, 1] - 0.015)
        raise ValueError("IK did not converge within the joint limits")

    def _preflight(self, stages, initial):
        s, d = self.scene, self.ik_data
        previous = initial
        # Only fixed-environment/self contacts enter planning; no object truth is consulted.
        for stage in stages:
            for fraction in np.linspace(0, 1, 30):
                d.qpos[:] = self.m.qpos0
                d.qpos[s.arm_qpos] = previous * (1 - fraction) + stage.q * fraction
                d.qpos[s.finger_qpos] = max(0.002, stage.gripper)
                self.mj.mj_forward(self.m, d)
                errors = s.illegal_contacts(d)
                if errors:
                    raise ValueError(f"unsafe joint path at {stage.name}: {errors[0]}")
            previous = stage.q

    def plan(self, observation):
        initial = self.scene.data.qpos[self.scene.arm_qpos].copy()
        candidates, failures = [], []
        for flip in (0, 1):
            try:
                grab_r = downward_rotation(observation.heading_rad + flip * math.pi)
                place_r = downward_rotation(flip * math.pi)
                xy = observation.center_xy
                pre = self.inverse_kinematics([*xy, 0.12], grab_r, initial)
                grip = self.inverse_kinematics([*xy, GRASP_Z], grab_r, pre)
                lift = self.inverse_kinematics([*xy, 0.12], grab_r, grip)
                transfer = self.inverse_kinematics([*GOAL_XY, 0.12], place_r, lift)
                lower = self.inverse_kinematics([*GOAL_XY, 0.010], place_r, transfer)
                stages = [Stage("approach", pre, 0.04, 1.5), Stage("descend", grip, 0.04, 1.2),
                          Stage("align_settle", grip, 0.04, 0.3),
                          Stage("close", grip, 0.0, 0.8), Stage("lift", lift, 0.0, 1.2),
                          Stage("hold", lift, 0.0, 0.6), Stage("transfer", transfer, 0.0, 1.6),
                          Stage("lower", lower, 0.0, 1.2), Stage("release", lower, 0.04, 0.6),
                          Stage("retreat", transfer, 0.04, 1.0), Stage("settle", transfer, 0.04, 0.7)]
                previous = initial
                for stage in stages:
                    stage.duration = max(stage.duration, float(1.875 * np.max(np.abs(stage.q - previous)) / 1.2))
                    previous = stage.q
                self._preflight(stages, initial)
                travel = sum(float(np.linalg.norm(stage.q - (initial if i == 0 else stages[i-1].q))) for i, stage in enumerate(stages))
                candidates.append((travel, flip, stages))
            except ValueError as exc:
                failures.append(str(exc))
        if not candidates:
            raise ValueError("; ".join(failures))
        _, flip, stages = min(candidates, key=lambda item: item[0])
        return stages, {"symmetric_gripper_branch": flip, "goal_xy_m": GOAL_XY, "goal_heading_rad": 0,
                        "fixed_grasp_z_m": GRASP_Z, "rejected_candidate_reasons": failures}

    def execute(self, stages, observer):
        s, d = self.scene, self.scene.data
        for stage in stages:
            start = d.qpos[s.arm_qpos].copy()
            start_gripper = float(d.ctrl[s.gripper_actuator])
            steps = int(math.ceil(stage.duration / self.m.opt.timestep))
            for step in range(steps):
                fraction = (step + 1) / steps
                smooth = 10*fraction**3 - 15*fraction**4 + 6*fraction**5
                d.ctrl[s.arm_actuators] = (stage.q if stage.name in ("align_settle", "close", "hold", "release", "settle")
                                          else start + smooth * (stage.q - start))
                g = min(1.0, 2 * fraction)
                gripper_smooth = 10*g**3 - 15*g**4 + 6*g**5
                d.ctrl[s.gripper_actuator] = start_gripper + gripper_smooth * (stage.gripper - start_gripper)
                self.mj.mj_step(self.m, d)
                observer(stage.name)
