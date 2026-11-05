"""Pinned Panda model, explicitly adapted fingertips and the original screw proxy."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .robot_assets import audit_panda, PANDA_COMMIT
from .screw_scene import ScrewScene, scene_xml, ANCHORS

ROBOT_BASE = [-0.45, 0.0, 0.0]
HOME = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.7854])
GOAL_XY = [0.13, 0.04]
GRASP_Z = 0.0045
ADAPTATION = {"robot_commit": PANDA_COMMIT, "base_world_m": ROBOT_BASE,
              "tcp_in_hand_m": [0, 0, 0.115], "pad_half_size_m": [0.008, 0.002, 0.002],
              "pad_local_finger_pos_m": [0, 0.003, 0.0566], "pad_friction": [1.0, 0.005, 0.0001],
              "gripper_kp": 1500, "gripper_kv": 10, "gripper_force_limit_N": 10,
              "robot_gravity_compensation": True, "object_gravity_compensation": False,
              "timestep_s": 0.001, "contact_solref": [0.003, 1],
              "friction_cone": "elliptic", "impratio": 10, "solver": "Newton", "solver_tolerance": 1e-10,
              "nominal_grasp_z_m": GRASP_Z,
              "notes": ["Custom thin fingertips; not an unmodified OEM Panda gripper",
                        "Existing finger inertia retained; small adapter inertia neglected",
                        "Screw collision geometry and scale unchanged; no adhesive or weld constraint"]}


def _camera(parent, name, eye, target, fovy):
    z = np.subtract(eye, target); z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(parent, "camera", name=name, pos=" ".join(map(str, eye)),
                  xyaxes=" ".join(map(str, [*x, *y])), fovy=str(fovy))


def panda_xml(asset_root: Path, size: int) -> str:
    panda = ET.parse(asset_root / "panda.xml").getroot()
    root = ET.fromstring(scene_xml(size))
    root.set("model", "panda_screw_m4")
    root.find("compiler").set("meshdir", (asset_root / "assets").as_posix())
    root.find("option").set("timestep", "0.001")
    root.find("option").set("cone", "elliptic")
    root.find("option").set("impratio", "10")
    root.find("option").set("solver", "Newton")
    root.find("option").set("tolerance", "1e-10")
    root.append(deepcopy(panda.find("default")))
    asset = root.find("asset")
    for element in panda.find("asset"):
        asset.append(deepcopy(element))
    ET.SubElement(asset, "material", name="thin_pad", rgba="0.22 0.32 0.42 1")
    robot = deepcopy(panda.find("worldbody/body"))
    robot.set("pos", " ".join(map(str, ROBOT_BASE)))
    for body in robot.iter("body"):
        body.set("gravcomp", "1")
    for name in ("left_finger", "right_finger"):
        finger = robot.find(f".//body[@name='{name}']")
        ET.SubElement(finger, "geom", name=f"{name}_thin_pad", type="box", size="0.008 0.002 0.002",
                      pos="0 0.003 0.0566", material="thin_pad", friction="1 0.005 0.0001",
                      condim="4", solref="0.003 1", solimp="0.95 0.99 0.0001", mass="0")
    hand = robot.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="grasp_tcp", pos="0 0 0.115", size="0.001", rgba="0 0 0 0")
    world = root.find("worldbody")
    world.find("geom[@name='table']").set("size", "1.2 1.2 0.01")
    for geom in [world.find("geom[@name='table']"), *world.findall("body[@name='screw']/geom")]:
        if geom.get("name") in ("table", "shaft", "head"):
            geom.set("solref", "0.003 1")
            geom.set("solimp", "0.95 0.99 0.0001")
    world.append(robot)
    _camera(world, "overview", [0.75, -0.9, 0.75], [-0.15, 0, 0.25], 48)
    _camera(world, "closeup", [0.30, -0.24, 0.27], [0.055, 0.01, 0.045], 45)
    for i, (dx, dy, sx, sy) in enumerate([(-0.025, 0, 0.0008, 0.025), (0.025, 0, 0.0008, 0.025),
                                         (0, -0.025, 0.025, 0.0008), (0, 0.025, 0.025, 0.0008)]):
        ET.SubElement(world, "geom", name=f"goal_mark_{i}", type="box",
                      pos=f"{GOAL_XY[0]+dx} {GOAL_XY[1]+dy} 0.0002", size=f"{sx} {sy} 0.0002",
                      rgba="0.1 0.65 0.25 1", contype="0", conaffinity="0")
    for name in ("tendon", "equality", "actuator", "contact"):
        root.append(deepcopy(panda.find(name)))
    gripper = root.find("actuator/general[@name='actuator8']")
    gripper.set("gainprm", "1500")
    gripper.set("biasprm", "0 -1500 -10")
    gripper.set("ctrlrange", "0 0.04")
    gripper.set("forcerange", "-10 10")
    return ET.tostring(root, encoding="unicode")


class PandaScene(ScrewScene):
    def __init__(self, assets: str | Path, size=640):
        self.asset_root = audit_panda(assets)
        super().__init__(size, xml=panda_xml(self.asset_root, size))
        m = self.model
        joints = [m.joint(f"joint{i}").id for i in range(1, 8)]
        self.arm_qpos = m.jnt_qposadr[joints].copy()
        self.arm_dofs = m.jnt_dofadr[joints].copy()
        self.arm_ranges = m.jnt_range[joints].copy()
        self.arm_actuators = np.array([m.actuator(f"actuator{i}").id for i in range(1, 8)])
        self.gripper_actuator = m.actuator("actuator8").id
        self.finger_qpos = np.array([m.jnt_qposadr[m.joint(f"finger_joint{i}").id] for i in (1, 2)])
        self.tcp_id = m.site("grasp_tcp").id
        self.robot_bodies = {m.body(n).id for n in [*(f"link{i}" for i in range(8)), "hand", "left_finger", "right_finger"]}
        self.finger_bodies = {m.body(n).id for n in ("left_finger", "right_finger")}
        m.body_gravcomp[list(self.robot_bodies)] = 1
        if m.ngravcomp != len(self.robot_bodies) or m.body_gravcomp[self.body_id] != 0:
            raise ValueError("gravity compensation must be compiled for the robot only")
        self.robot_geoms = {i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in self.robot_bodies}
        self.table_id = m.geom("table").id
        self.base_id = m.body("link0").id

    def _initialize_additional_bodies(self):
        self.data.qpos[self.arm_qpos] = HOME
        self.data.qpos[self.finger_qpos] = 0.04
        self.data.ctrl[self.arm_actuators] = HOME
        self.data.ctrl[self.gripper_actuator] = 0.04

    def snapshot(self):
        spec = self.mj.mjtState.mjSTATE_INTEGRATION
        state = np.empty(self.mj.mj_stateSize(self.model, spec))
        self.mj.mj_getState(self.model, self.data, state, spec)
        return state

    def restore(self, state):
        # Initialization of a paired branch only; never called during execution.
        self.mj.mj_setState(self.model, self.data, state, self.mj.mjtState.mjSTATE_INTEGRATION)
        self.mj.mj_forward(self.model, self.data)

    def capture(self, camera="overview"):
        import cv2
        option = self.mj.MjvOption(); option.sitegroup[:] = 0
        self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def object_anchors(self):
        """GT/evaluator only. Control code must not call this method."""
        return ANCHORS @ self.data.xmat[self.body_id].reshape(3, 3).T + self.data.xpos[self.body_id]

    def illegal_contacts(self, data=None):
        data = self.data if data is None else data
        errors = []
        for contact in data.contact:
            if contact.dist >= -0.00005:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            ba, bb = int(self.model.geom_bodyid[a]), int(self.model.geom_bodyid[b])
            if a == self.table_id and bb in self.robot_bodies and bb != self.base_id:
                errors.append("robot_table")
            elif b == self.table_id and ba in self.robot_bodies and ba != self.base_id:
                errors.append("robot_table")
            elif ba in self.robot_bodies and bb in self.robot_bodies:
                if not ({ba, bb} <= self.finger_bodies and contact.dist >= -0.0005):
                    errors.append("robot_self")
        return errors
