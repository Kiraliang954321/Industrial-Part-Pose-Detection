"""One RGB inspection station, physical outer-rim grasp, and a clearance shaft fixture."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from yolo_conveyor.simulation.panda_scene import PandaScene,panda_xml,_camera,HOME
from yolo_conveyor.simulation.screw_scene import ScrewScene
from yolo_conveyor.simulation.robot_assets import audit_panda
from .assets import PARTS,build_part

INSPECT_TCP=np.array([0.,0.,.22])
SHAFT_XY=np.array([-.10,.12]);SHAFT_TOP=.12;ENGAGEMENT=.003
SETUP=dict(inspection_tcp_m=INSPECT_TCP.tolist(),shaft_xy_m=SHAFT_XY.tolist(),shaft_top_m=SHAFT_TOP,
           engagement_m=ENGAGEMENT,radial_clearance_m=.001,thin_pad_half_size_m=[.003,.002,.0015],rgb_cameras_used=1)


def scene_xml(assets,kind,size):
    root=build_part(ET.fromstring(panda_xml(assets,size)),kind);root.set('model','annular_a1_'+kind)
    world=root.find('worldbody');asset=root.find('asset')
    for g in list(world.findall('geom')):
        if g.get('name','').startswith('goal_mark_'):world.remove(g)
    for name in ('left_finger','right_finger'):
        root.find(f".//geom[@name='{name}_thin_pad']").set('size','.003 .002 .0015')
    ET.SubElement(asset,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.86 .87 .90',rgb2='.65 .68 .72',width='512',height='512')
    ET.SubElement(asset,'material',name='fixture_blue',rgba='.12 .40 .58 1')
    _camera(world,'inspect_rgb',[.18,-.08,.15],INSPECT_TCP,22)
    ET.SubElement(world,'light',name='inspection_fill',pos='.18 -.08 .15',dir='-.18 .08 .07',diffuse='.65 .65 .65',specular='.15 .15 .15',castshadow='false')
    _camera(world,'assembly_view',[.17,-.17,.25],[*SHAFT_XY,.12],35)
    fixture=ET.SubElement(world,'body',name='shaft_fixture',pos=f'{SHAFT_XY[0]} {SHAFT_XY[1]} 0')
    ET.SubElement(fixture,'geom',name='shaft_base',type='box',pos='0 0 .02',size='.025 .025 .02',material='fixture_blue')
    radius=PARTS[kind]['bore']/2-.001
    ET.SubElement(fixture,'geom',name='shaft',type='cylinder',pos='0 0 .08',size=f'{radius} .04',material='fixture_blue',friction='.3 .005 .0001')
    return ET.tostring(root,encoding='unicode')


class AssemblyScene(PandaScene):
    def __init__(self,assets,kind,size=640):
        self.kind=kind;self.spec=PARTS[kind];self.asset_root=audit_panda(assets)
        ScrewScene.__init__(self,size,xml=scene_xml(self.asset_root,kind,size))
        m=self.model;joints=[m.joint(f'joint{i}').id for i in range(1,8)]
        self.arm_qpos=m.jnt_qposadr[joints].copy();self.arm_dofs=m.jnt_dofadr[joints].copy();self.arm_ranges=m.jnt_range[joints].copy()
        self.arm_actuators=np.array([m.actuator(f'actuator{i}').id for i in range(1,8)])
        self.gripper_actuator=m.actuator('actuator8').id
        self.finger_qpos=np.array([m.jnt_qposadr[m.joint(f'finger_joint{i}').id] for i in (1,2)])
        self.tcp_id=m.site('grasp_tcp').id
        self.robot_bodies={m.body(n).id for n in [*(f'link{i}' for i in range(8)),'hand','left_finger','right_finger']}
        self.finger_bodies={m.body(n).id for n in ('left_finger','right_finger')}
        self.robot_geoms={i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in self.robot_bodies}
        self.table_id=m.geom('table').id;self.base_id=m.body('link0').id
        self.fixture_id=m.body('shaft_fixture').id
        if m.body_gravcomp[self.body_id]!=0:raise ValueError('part must remain gravity driven')

    def initialize(self,rng,face='front',empty=False):
        m,d=self.model,self.data;self.mj.mj_resetData(m,d)
        yaw=float(rng.uniform(-math.pi,math.pi));qz=np.array([math.cos(yaw/2),0,0,math.sin(yaw/2)])
        q=np.zeros(4);self.mj.mju_mulQuat(q,qz,np.array([0.,1,0,0]) if face=='back' else np.array([1.,0,0,0]))
        d.qpos[:3]=[rng.uniform(-.00025,.00025),rng.uniform(-.00025,.00025)+(.12 if empty else 0),self.spec['height']/2+.001]
        d.qpos[3:7]=q;self._initialize_additional_bodies();self.mj.mj_forward(m,d)
        for _ in range(300):self.mj.mj_step(m,d)
        shade=rng.uniform(.16,.45);m.mat_rgba[m.material('part_metal').id,:3]=[shade*.95,shade,shade*1.04]
        m.mat_rgba[m.material('shield_metal').id,:3]=shade*.85
        self.appearance(rng)

    def appearance(self,rng):
        m=self.model;light=np.array([rng.uniform(-.1,.3),rng.uniform(-.3,.1),rng.uniform(.35,.6)])
        m.light_pos[0]=light;direction=INSPECT_TCP-light;m.light_dir[0]=direction/np.linalg.norm(direction)
        m.light_diffuse[0]=rng.uniform(.5,1.1);m.mat_specular[m.material('part_metal').id]=rng.uniform(.15,.55)
        m.mat_rgba[m.material('surface').id,:3]=rng.uniform(.5,.8,3)

    def tool_transform(self):
        t=np.eye(4);t[:3,:3]=self.data.site_xmat[self.tcp_id].reshape(3,3);t[:3,3]=self.data.site_xpos[self.tcp_id];return t

    def object_anchors(self):
        h=self.spec['height']/2
        return np.array([[0,0,h],[0,0,-h]])@self.data.xmat[self.body_id].reshape(3,3).T+self.data.xpos[self.body_id]

    def illegal_contacts(self,data=None):
        d=self.data if data is None else data;errors=super().illegal_contacts(d)
        for c in d.contact:
            if c.dist>=-.00005:continue
            a,b=(int(self.model.geom_bodyid[i]) for i in (c.geom1,c.geom2))
            if (a==self.fixture_id and b in self.robot_bodies) or (b==self.fixture_id and a in self.robot_bodies):errors.append('robot_fixture')
        return errors
