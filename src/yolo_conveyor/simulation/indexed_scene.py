"""L1 physical indexing pallet; M4 source and frozen scene stay unchanged."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .robot_assets import audit_panda
from .screw_scene import ScrewScene
from .panda_scene import PandaScene, panda_xml, HOME

INFEED_Y = -0.18
GATE_Q = 0.18
OUTFEED_Q = 0.36
INDEX_SPEED = 0.06
LINE_SETUP = {"kind": "single_indexing_pallet_episode", "infeed_world_y_m": INFEED_Y,
              "gate_joint_m": GATE_Q, "outfeed_joint_m": OUTFEED_Q, "speed_limit_m_s": INDEX_SPEED,
              "pallet_half_size_m": [0.085, 0.08, 0.006], "pallet_mass_kg": 0.8,
              "index_kp": 3000, "index_kv": 100, "index_force_limit_N": 100,
              "pallet_friction": [1.0, 0.005, 0.001], "pallet_contact_dimensions": 6,
              "reject_support_friction": [0.8, 0.01, 0.001], "reject_support_contact_dimensions": 6,
              "reject_world_xy_m": [0.13, 0.04], "support_top_z_m": 0.0,
              "limits": "High-friction positioning pallet, not a plain endless belt; no continuous queue, unloading, return cycle or automatic review recovery"}


def indexed_xml(assets, size):
    root = ET.fromstring(panda_xml(assets, size))
    root.set("model", "screw_indexed_inspection_l1")
    asset, world = root.find("asset"), root.find("worldbody")
    ET.SubElement(asset, "material", name="factory_floor", rgba="0.26 0.29 0.32 1")
    ET.SubElement(asset, "material", name="machine_frame", rgba="0.18 0.22 0.27 1")
    table = world.find("geom[@name='table']")
    table.set("pos", "0 0 -0.03"); table.set("material", "factory_floor")
    ET.SubElement(world, "geom", name="robot_pedestal", type="box", pos="-0.45 0 -0.015",
                  size="0.095 0.095 0.015", material="machine_frame")
    ET.SubElement(world, "geom", name="reject_support", type="box", pos="0.13 0.04 -0.015",
                  size="0.045 0.05 0.015", material="surface", friction="0.8 0.01 0.001", condim="6",
                  solref="0.003 1", solimp="0.95 0.99 0.0001")
    for geom in world.findall("geom"):
        if geom.get("name", "").startswith("goal_mark_"):
            geom.set("rgba", "0.75 0.15 0.12 1")
    for side in (-1, 1):
        ET.SubElement(world, "geom", name=f"index_rail_{side}", type="box",
                      pos=f"{side*0.105} 0 -0.017", size="0.009 0.29 0.008", material="machine_frame")
    tray = ET.SubElement(world, "body", name="index_pallet", pos=f"0 {INFEED_Y} -0.006")
    ET.SubElement(tray, "joint", name="index_slide", type="slide", axis="0 1 0", range="0 0.36", damping="5", armature="0.1")
    ET.SubElement(tray, "geom", name="pallet_surface", type="box", size="0.085 0.08 0.006", mass="0.8",
                  material="surface", friction="1 0.005 0.001", condim="6", solref="0.003 1", solimp="0.95 0.99 0.0001")
    for name, y, color in (("infeed_mark", -0.28, "0.1 0.4 0.7 1"), ("outfeed_mark", 0.28, "0.1 0.7 0.25 1")):
        ET.SubElement(world, "geom", name=name, type="box", pos=f"0 {y} -0.015", size="0.085 0.008 0.003",
                      rgba=color, contype="0", conaffinity="0")
    ET.SubElement(root.find("actuator"), "general", name="index_drive", joint="index_slide",
                  dyntype="none", biastype="affine", gainprm="3000", biasprm="0 -3000 -100",
                  ctrlrange="0 0.36", forcerange="-100 100")
    return ET.tostring(root, encoding="unicode")


class IndexedScene(PandaScene):
    def __init__(self, assets: str | Path, size=640):
        self.asset_root = audit_panda(assets)
        ScrewScene.__init__(self, size, xml=indexed_xml(self.asset_root, size))
        # Separate adapter bindings preserve the byte-frozen M4 implementation.
        m = self.model
        joints = [m.joint(f"joint{i}").id for i in range(1, 8)]
        self.arm_qpos = m.jnt_qposadr[joints].copy(); self.arm_dofs = m.jnt_dofadr[joints].copy()
        self.arm_ranges = m.jnt_range[joints].copy()
        self.arm_actuators = np.array([m.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.gripper_actuator = m.actuator("actuator8").id
        self.finger_qpos = np.array([m.jnt_qposadr[m.joint(f"finger_joint{i}").id] for i in (1, 2)])
        self.tcp_id = m.site("grasp_tcp").id
        self.robot_bodies = {m.body(n).id for n in [*(f"link{i}" for i in range(8)), "hand", "left_finger", "right_finger"]}
        self.finger_bodies = {m.body(n).id for n in ("left_finger", "right_finger")}
        self.robot_geoms = {i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in self.robot_bodies}
        self.base_id = m.body("link0").id
        self.table_id = m.geom("reject_support").id  # M4 scoring can certify landing on this support.
        self.support_ids = {m.geom(name).id for name in ("table", "robot_pedestal", "reject_support", "pallet_surface", "index_rail_-1", "index_rail_1")}
        self.pallet_geom = m.geom("pallet_surface").id
        self.pallet_q = int(m.jnt_qposadr[m.joint("index_slide").id])
        self.pallet_dof = int(m.jnt_dofadr[m.joint("index_slide").id])
        self.pallet_actuator = m.actuator("index_drive").id
        self.spawn_heading_rad = 0.0
        if m.ngravcomp != 11 or m.body_gravcomp[self.body_id] != 0:
            raise ValueError("unexpected gravity compensation configuration")

    def _initialize_additional_bodies(self):
        super()._initialize_additional_bodies()
        self.data.qpos[self.pallet_q] = 0
        self.data.ctrl[self.pallet_actuator] = 0
        self.data.qpos[0] *= 0.5
        self.data.qpos[1] = INFEED_Y + self.data.qpos[1] * 0.5
        angle = self.spawn_heading_rad
        self.data.qpos[3:7] = [math.cos(angle/2), 0, 0, math.sin(angle/2)]

    def initialize(self, rng, condition, heading_rad):
        self.spawn_heading_rad = heading_rad
        result = self.render(rng, condition, True)
        result[-1]["initial_yaw_rad"] = heading_rad
        return result[4], result[-1]

    def illegal_contacts(self, data=None):
        d = self.data if data is None else data
        errors = []
        for contact in d.contact:
            if contact.dist >= -0.00005:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            ba, bb = int(self.model.geom_bodyid[a]), int(self.model.geom_bodyid[b])
            if (a in self.support_ids and bb in self.robot_bodies and bb != self.base_id) or (b in self.support_ids and ba in self.robot_bodies and ba != self.base_id):
                errors.append("robot_support")
            elif ba in self.robot_bodies and bb in self.robot_bodies:
                if not ({ba, bb} <= self.finger_bodies and contact.dist >= -0.0005):
                    errors.append("robot_self")
        return errors

    def advance(self, seconds, observer, stage):
        steps = int(round(seconds / self.model.opt.timestep))
        for _ in range(steps):
            self.mj.mj_step(self.model, self.data)
            observer(stage)

    def index_to(self, target, observer, stage):
        if not 0 <= target <= OUTFEED_Q:
            raise ValueError("index target outside travel limits")
        start = float(self.data.qpos[self.pallet_q])
        duration = max(0.5, 1.875 * abs(target - start) / INDEX_SPEED)
        steps = int(math.ceil(duration / self.model.opt.timestep))
        for step in range(steps):
            f = (step + 1) / steps
            smooth = 10*f**3 - 15*f**4 + 6*f**5
            self.data.ctrl[self.pallet_actuator] = start + smooth * (target - start)
            self.mj.mj_step(self.model, self.data)
            observer(stage)
        self.data.ctrl[self.pallet_actuator] = target
        self.advance(0.3, observer, f"{stage}_settle")
        if abs(float(self.data.qpos[self.pallet_q]) - target) > 0.001:
            raise ValueError("index encoder did not reach target")
