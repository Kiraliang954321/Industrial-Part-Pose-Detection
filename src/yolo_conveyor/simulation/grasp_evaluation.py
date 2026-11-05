"""Read-only physical evaluator and trace recorder, isolated from grasp control."""

from collections import deque
import json
import math
import numpy as np

from yolo_conveyor.orientation.geometry import directed_error
from .panda_scene import GOAL_XY

CRITERIA = {"lift_above_initial_m": 0.04, "hold_bilateral_fraction": 0.8,
            "close_tail_bilateral_fraction": 0.8, "close_tail_seconds": 0.2,
            "placement_xy_tolerance_m": 0.015, "placement_heading_tolerance_deg": 10,
            "maximum_soft_contact_penetration_m": 0.001,
            "settled_linear_speed_m_s": 0.015, "settled_angular_speed_rad_s": 0.2,
            "note": "Simulation development criteria, not real industrial tolerances"}


class PhysicalEvaluator:
    def __init__(self, scene, trace, frame_callback=None):
        self.scene, self.trace = scene, trace
        self.frame_callback = frame_callback
        self.initial = scene.object_anchors().mean(axis=0)
        self.start_time = float(scene.data.time)
        self.stages = {}
        self.close_tail = deque(maxlen=int(CRITERIA["close_tail_seconds"] / scene.model.opt.timestep))
        self.last_trace = -1.0
        self.last_frame = -1.0
        self.max_step_displacement = 0.0
        self.previous_center = self.initial.copy()

    def sample(self, stage):
        s, d = self.scene, self.scene.data
        anchors = s.object_anchors()
        center = anchors.mean(axis=0)
        axis = anchors[0] - anchors[1]
        angle = math.degrees(math.atan2(axis[1], axis[0])) % 360
        self.max_step_displacement = max(self.max_step_displacement, float(np.linalg.norm(center - self.previous_center)))
        self.previous_center = center.copy()
        contacted = set(); max_pen = 0.0; table_contact = False; nonfinger_contact = False
        for contact in d.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            ba, bb = int(s.model.geom_bodyid[a]), int(s.model.geom_bodyid[b])
            if s.body_id not in (ba, bb):
                continue
            other = bb if ba == s.body_id else ba
            max_pen = max(max_pen, float(-contact.dist))
            if contact.dist <= 0:
                if other in s.finger_bodies:
                    contacted.add(other)
                if a == s.table_id or b == s.table_id:
                    table_contact = True
                if other in s.robot_bodies and other not in s.finger_bodies:
                    nonfinger_contact = True
        bilateral = contacted == s.finger_bodies
        illegal = bool(s.illegal_contacts()) or nonfinger_contact
        values = self.stages.setdefault(stage, {"steps": 0, "bilateral_steps": 0, "illegal_contact_steps": 0,
                                               "min_center_z_m": 10.0, "max_center_z_m": -10.0,
                                               "maximum_penetration_m": 0.0})
        values["steps"] += 1
        values["bilateral_steps"] += int(bilateral)
        values["illegal_contact_steps"] += int(illegal)
        values["min_center_z_m"] = min(values["min_center_z_m"], float(center[2]))
        values["max_center_z_m"] = max(values["max_center_z_m"], float(center[2]))
        values["maximum_penetration_m"] = max(values["maximum_penetration_m"], max_pen)
        if stage == "close":
            self.close_tail.append(bilateral)
        self.final = {"center_m": center.tolist(), "heading_deg": angle, "table_contact": table_contact,
                      "finger_contact": bool(contacted), "linear_speed_m_s": float(np.linalg.norm(d.qvel[:3])),
                      "angular_speed_rad_s": float(np.linalg.norm(d.qvel[3:6]))}
        t = float(d.time - self.start_time)
        if t - self.last_trace >= 0.02:
            self.trace.write(json.dumps({"time_s": t, "stage": stage, "robot_q": d.qpos[s.arm_qpos].tolist(),
                                         "tcp_world_m": d.site_xpos[s.tcp_id].tolist(), "object_truth": self.final,
                                         "bilateral": bilateral, "illegal_contact": illegal,
                                         "penetration_m": max_pen}, allow_nan=False) + "\n")
            self.last_trace = t
        if self.frame_callback and t - self.last_frame >= 1 / 12:
            self.frame_callback(stage, t)
            self.last_frame = t

    def result(self):
        hold, transfer = self.stages["hold"], self.stages["transfer"]
        close_fraction = sum(self.close_tail) / len(self.close_tail)
        def sustained(stage):
            return (stage["min_center_z_m"] >= self.initial[2] + CRITERIA["lift_above_initial_m"] and
                    stage["bilateral_steps"] / stage["steps"] >= CRITERIA["hold_bilateral_fraction"])
        max_z = max(s["max_center_z_m"] for s in self.stages.values())
        max_pen = max(s["maximum_penetration_m"] for s in self.stages.values())
        illegal_steps = sum(s["illegal_contact_steps"] for s in self.stages.values())
        xy_error = float(np.linalg.norm(np.asarray(self.final["center_m"][:2]) - GOAL_XY))
        angle_error = directed_error(self.final["heading_deg"], 0)
        conditions = {"bilateral_grasp": close_fraction >= CRITERIA["close_tail_bilateral_fraction"],
                      "lifted": max_z >= self.initial[2] + CRITERIA["lift_above_initial_m"],
                      "held": sustained(hold), "transported": sustained(transfer),
                      "released": self.final["table_contact"] and not self.final["finger_contact"],
                      "placement_xy": xy_error <= CRITERIA["placement_xy_tolerance_m"],
                      "placement_heading": angle_error <= CRITERIA["placement_heading_tolerance_deg"],
                      "settled": self.final["linear_speed_m_s"] <= CRITERIA["settled_linear_speed_m_s"] and
                                 self.final["angular_speed_rad_s"] <= CRITERIA["settled_angular_speed_rad_s"],
                      "collision_free": illegal_steps == 0,
                      "penetration_bounded": max_pen <= CRITERIA["maximum_soft_contact_penetration_m"]}
        conditions = {key: bool(value) for key, value in conditions.items()}
        return {"attempted": True, "success": all(conditions.values()), "conditions": conditions,
                "failure_reasons": [k for k, passed in conditions.items() if not passed],
                "placement_xy_error_m": xy_error, "placement_heading_error_deg": angle_error,
                "close_tail_bilateral_fraction": close_fraction, "maximum_penetration_m": max_pen,
                "illegal_contact_steps": illegal_steps, "maximum_step_displacement_m": self.max_step_displacement,
                "simulated_seconds": float(self.scene.data.time - self.start_time), "final": self.final, "stages": self.stages}
