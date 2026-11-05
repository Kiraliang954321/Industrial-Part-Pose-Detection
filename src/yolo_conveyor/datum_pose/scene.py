"""Three fixed RGB views and physical presentation of custom mechanical-datum parts."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from yolo_conveyor.annular.scene import AssemblyScene,scene_xml,INSPECT_TCP
from yolo_conveyor.annular.control import AssemblyController
from yolo_conveyor.simulation.panda_scene import _camera
from yolo_conveyor.simulation.screw_scene import ScrewScene
from yolo_conveyor.simulation.robot_assets import audit_panda
from yolo_conveyor.simulation.panda_control import Stage
from yolo_conveyor.simulation.grasp_geometry import downward_rotation
from yolo_conveyor.held.scene import HeldScene
from .assets import PARTS,DATUMS,replace_part,anchors
from .geometry import tilt_rotation

VIEWS=('datum_a','datum_b','datum_c')
SETUP=dict(center_m=INSPECT_TCP.tolist(),camera_radius_m=.11,camera_z_offset_m=-.17,camera_azimuths_deg=[20,140,260],
           fovy_deg=24,tilt_range_deg=[0,30],spin_range_deg=[0,360],datums=DATUMS,
           asset_scope='custom notched outer bearing / keyed bevel proxies; not original standard parts')


def xml(assets,kind,feature,size):
    root=replace_part(ET.fromstring(scene_xml(assets,kind,size)),kind,feature);root.set('model','datum_pose_a2_'+kind)
    world=root.find('worldbody')
    for name,az in zip(VIEWS,SETUP['camera_azimuths_deg']):
        a=np.radians(az);eye=INSPECT_TCP+np.array([.11*np.cos(a),.11*np.sin(a),-.17]);_camera(world,name,eye,INSPECT_TCP,24)
        direction=INSPECT_TCP-eye
        ET.SubElement(world,'light',name=name+'_fill',pos=' '.join(map(str,eye)),dir=' '.join(map(str,direction)),diffuse='.22 .22 .22',castshadow='false')
    return ET.tostring(root,encoding='unicode')


class DatumScene(AssemblyScene):
    def __init__(self,assets,kind,feature=True,size=640):
        self.kind=kind;self.spec=PARTS[kind];self.feature=feature;self.asset_root=audit_panda(assets)
        ScrewScene.__init__(self,size,xml=xml(self.asset_root,kind,feature,size))
        m=self.model;joints=[m.joint(f'joint{i}').id for i in range(1,8)]
        self.arm_qpos=m.jnt_qposadr[joints].copy();self.arm_dofs=m.jnt_dofadr[joints].copy();self.arm_ranges=m.jnt_range[joints].copy()
        self.arm_actuators=np.array([m.actuator(f'actuator{i}').id for i in range(1,8)]);self.gripper_actuator=m.actuator('actuator8').id
        self.finger_qpos=np.array([m.jnt_qposadr[m.joint(f'finger_joint{i}').id] for i in (1,2)]);self.tcp_id=m.site('grasp_tcp').id
        self.robot_bodies={m.body(n).id for n in [*(f'link{i}' for i in range(8)),'hand','left_finger','right_finger']}
        self.finger_bodies={m.body(n).id for n in ('left_finger','right_finger')}
        self.robot_geoms={i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in self.robot_bodies}
        self.table_id=m.geom('table').id;self.base_id=m.body('link0').id;self.fixture_id=m.body('shaft_fixture').id

    def initialize_pose(self,rng,spin,empty=False):
        m,d=self.model,self.data;self.mj.mj_resetData(m,d);a=np.radians(spin)/2
        d.qpos[:3]=[rng.uniform(-.0002,.0002),rng.uniform(-.0002,.0002)+(.12 if empty else 0),self.spec['height']/2+.001]
        d.qpos[3:7]=[np.cos(a),0,0,np.sin(a)];self._initialize_additional_bodies();self.mj.mj_forward(m,d)
        for _ in range(300):self.mj.mj_step(m,d)
        shade=rng.uniform(.20,.44);m.mat_rgba[m.material('part_metal').id,:3]=[shade*.95,shade,shade*1.04];self.appearance(rng)

    camera_info=HeldScene.camera_info
    project=HeldScene.project

    def datum_world(self):return anchors(self.kind)@self.data.xmat[self.body_id].reshape(3,3).T+self.data.xpos[self.body_id]


class PoseController(AssemblyController):
    def present(self,tilt,azimuth):
        r=tilt_rotation(tilt,azimuth)@downward_rotation(0)
        q=self.inverse_kinematics(INSPECT_TCP,r,self.scene.data.qpos[self.scene.arm_qpos])
        return self.stages([Stage('orient',q,0,1.2),Stage('hold',q,0,.7)])
