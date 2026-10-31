"""Two-view held-screw station and a genuinely open segmented guide bore."""

from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np

from yolo_conveyor.simulation.robot_assets import audit_panda
from yolo_conveyor.simulation.panda_scene import PandaScene,panda_xml,_camera
from yolo_conveyor.simulation.screw_scene import ScrewScene

INSPECT_TCP = np.array([-0.10,0.,0.25])
_yaw = math.radians(-30)
INSPECT_R = np.array([[math.cos(_yaw),-math.sin(_yaw),0],
                      [math.sin(_yaw),math.cos(_yaw),0],[0,0,1.]]) @ np.column_stack([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
GUIDE_XY = np.array([-0.10,0.10])
GUIDE_TOP = 0.12
GUIDE_RADIUS = 0.005
INSERT_DEPTH = 0.008
SETUP = {"inspection_tcp_m":INSPECT_TCP.tolist(),"inspection_rotation":INSPECT_R.tolist(),
         "guide_xy_m":GUIDE_XY.tolist(),"guide_top_m":GUIDE_TOP,"guide_inner_radius_m":GUIDE_RADIUS,
         "guide_segments":24,"guide_wall_height_m":0.010,"preinsert_depth_m":INSERT_DEPTH,
         "task":"Held two-view axis/position inspection and held pre-insertion; not screw tightening"}


def held_xml(assets,size):
    root=ET.fromstring(panda_xml(assets,size));root.set("model","held_screw_h1")
    world=root.find("worldbody");asset=root.find("asset")
    for geom in list(world.findall("geom")):
        if geom.get("name","").startswith("goal_mark_"):world.remove(geom)
    ET.SubElement(asset,"texture",name="held_sky",type="skybox",builtin="gradient",rgb1="0.85 0.86 0.88",rgb2="0.65 0.68 0.72",width="512",height="512")
    ET.SubElement(asset,"material",name="guide_blue",rgba="0.18 0.45 0.62 1")
    _camera(world,"inspect_a",[-0.25,-0.259807621,0.25],INSPECT_TCP,30)
    _camera(world,"inspect_b",[-0.359807621,0.15,0.25],INSPECT_TCP,30)
    guide=ET.SubElement(world,"body",name="guide_fixture",pos=f"{GUIDE_XY[0]} {GUIDE_XY[1]} 0")
    ET.SubElement(guide,"geom",name="guide_base",type="box",pos="0 0 0.005",size="0.033 0.025 0.005",material="guide_blue")
    for sign in (-1,1):
        half=(GUIDE_TOP-.020)/2
        ET.SubElement(guide,"geom",name=f"guide_post_{sign}",type="box",pos=f"{sign*.015} 0 {half+.010}",size=f"0.004 0.008 {half}",material="guide_blue")
    for i in range(24):
        a,b=2*math.pi*i/24,2*math.pi*(i+1)/24
        vertices=[]
        for z in (GUIDE_TOP-.010,GUIDE_TOP):
            for radius,angle in ((GUIDE_RADIUS,a),(.02,a),(.02,b),(GUIDE_RADIUS,b)):
                vertices.extend([radius*math.cos(angle),radius*math.sin(angle),z])
        name=f"guide_wedge_{i}"
        ET.SubElement(asset,"mesh",name=name,vertex=" ".join(map(str,vertices)))
        ET.SubElement(guide,"geom",name=name,type="mesh",mesh=name,material="guide_blue",friction="0.3 0.005 0.0001",solref="0.003 1",solimp="0.95 0.99 0.0001")
    hand=root.find(".//body[@name='hand']")
    ET.SubElement(hand,"site",name="wrist_sensor",pos="0 0 0.115",size="0.001",rgba="0 0 0 0")
    sensors=ET.SubElement(root,"sensor")
    ET.SubElement(sensors,"force",name="wrist_force",site="wrist_sensor")
    ET.SubElement(sensors,"torque",name="wrist_torque",site="wrist_sensor")
    return ET.tostring(root,encoding="unicode")


class HeldScene(PandaScene):
    def __init__(self,assets,size=640):
        self.asset_root=audit_panda(assets)
        ScrewScene.__init__(self,size,xml=held_xml(self.asset_root,size))
        m=self.model;joints=[m.joint(f"joint{i}").id for i in range(1,8)]
        self.arm_qpos=m.jnt_qposadr[joints].copy();self.arm_dofs=m.jnt_dofadr[joints].copy();self.arm_ranges=m.jnt_range[joints].copy()
        self.arm_actuators=np.array([m.actuator(f"actuator{i}").id for i in range(1,8)])
        self.gripper_actuator=m.actuator("actuator8").id
        self.finger_qpos=np.array([m.jnt_qposadr[m.joint(f"finger_joint{i}").id] for i in (1,2)])
        self.tcp_id=m.site("grasp_tcp").id
        self.robot_bodies={m.body(n).id for n in [*(f"link{i}" for i in range(8)),"hand","left_finger","right_finger"]}
        self.finger_bodies={m.body(n).id for n in ("left_finger","right_finger")}
        self.robot_geoms={i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in self.robot_bodies}
        self.table_id=m.geom("table").id;self.base_id=m.body("link0").id
        self.fixture_body=m.body("guide_fixture").id
        self.fixture_geoms={i for i in range(m.ngeom) if int(m.geom_bodyid[i])==self.fixture_body}
        self.force_adr=int(m.sensor_adr[m.sensor("wrist_force").id])
        self.torque_adr=int(m.sensor_adr[m.sensor("wrist_torque").id])
        self.spawn={"offset_x":0.,"offset_y":0.,"yaw":0.,"empty":False}
        if m.ngravcomp!=11 or m.body_gravcomp[self.body_id]!=0:raise ValueError("unexpected gravity setup")

    def _initialize_additional_bodies(self):
        super()._initialize_additional_bodies()
        self.data.qpos[:3]=[-.0015+self.spawn["offset_x"],self.spawn["offset_y"]+(0.13 if self.spawn["empty"] else 0),.025]
        yaw=self.spawn["yaw"];self.data.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]

    def initialize(self,rng,spawn,light="nominal"):
        self.spawn=spawn
        result=self.render(rng,light,True)
        result[-1]["initial_yaw_rad"]=spawn["yaw"]
        return result[-1]

    def tool_transform(self):
        t=np.eye(4);t[:3,:3]=self.data.site_xmat[self.tcp_id].reshape(3,3);t[:3,3]=self.data.site_xpos[self.tcp_id]
        return t

    def camera_info(self,name):
        i=self.model.camera(name).id;r=self.data.cam_xmat[i].reshape(3,3)@np.diag([1.,-1.,-1.])
        t=np.eye(4);t[:3,:3]=r;t[:3,3]=self.data.cam_xpos[i]
        focal=.5*self.size/math.tan(math.radians(self.model.cam_fovy[i])*.5)
        return {"name":name,"K":[[focal,0,self.size/2],[0,focal,self.size/2],[0,0,1]],"T_world_camera_cv":t.tolist(),"distortion":[0]*5}

    def project(self,name,points):
        cam=self.camera_info(name);t=np.array(cam["T_world_camera_cv"]);k=np.array(cam["K"])
        p=(np.asarray(points)-t[:3,3])@t[:3,:3];uv=p@k.T
        return uv[:,:2]/uv[:,2:3]

    def wrist_wrench(self):
        return np.r_[self.data.sensordata[self.force_adr:self.force_adr+3],self.data.sensordata[self.torque_adr:self.torque_adr+3]].copy()

    def illegal_contacts(self,data=None):
        d=self.data if data is None else data
        result=super().illegal_contacts(d)
        for contact in d.contact:
            if contact.dist>=-.00005:continue
            a,b=int(contact.geom1),int(contact.geom2)
            if (a in self.fixture_geoms and int(self.model.geom_bodyid[b]) in self.robot_bodies) or (b in self.fixture_geoms and int(self.model.geom_bodyid[a]) in self.robot_bodies):result.append("robot_fixture")
        return result
