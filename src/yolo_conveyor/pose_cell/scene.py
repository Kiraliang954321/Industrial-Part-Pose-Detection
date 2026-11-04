"""Side-grip cell with a raised feed nest and six fixed opposing RGB views."""
import xml.etree.ElementTree as ET
import numpy as np
from yolo_conveyor.annular.scene import AssemblyScene,scene_xml,SHAFT_XY,SHAFT_TOP,ENGAGEMENT
from yolo_conveyor.datum_pose.assets import PARTS,DATUMS,replace_part
from yolo_conveyor.simulation.panda_scene import _camera
from yolo_conveyor.simulation.screw_scene import ScrewScene
from yolo_conveyor.simulation.robot_assets import audit_panda
from yolo_conveyor.held.scene import HeldScene,INSPECT_R as SIDE_R
from .geometry import about,rz

CENTER=np.array([-.10,0.,.35]);NEST=np.array([-.10,-.10]);NEST_TOP=.18
VIEWS=tuple(f'cell_{hemisphere}_{i}' for hemisphere in ('lower','upper') for i in range(3))
SETUP=dict(center_m=CENTER.tolist(),nest_xy_m=NEST.tolist(),nest_top_m=NEST_TOP,
           side_grip_rotation=SIDE_R.tolist(),camera_radius_m=.13,camera_height_offset_m=.12,
           camera_azimuths_deg=[30,150,270],fovy_deg=26,shaft_xy_m=SHAFT_XY.tolist(),shaft_top_m=SHAFT_TOP,
           engagement_m=ENGAGEMENT,scope='custom mechanical-datum proxies; side grip; no real hardware')
CLOSE_BY_KIND={'bearing':.007,'gear':.007}
SETUP.update(close_width_m=CLOSE_BY_KIND,feed_post_half_size_m=[.0045,.0045,.09],contact_dimension=6,
             pad_friction=[1.,.005,.001],part_friction=[.8,.005,.001],finger_solref=[.003,1.],finger_solimp=[.95,.99,.0001],
             contact_scope='finite torsional/rolling resistance; all contacting finger surfaces share pad compliance; no weld or adhesion')


def xml(assets,kind,feature,size):
    root=replace_part(ET.fromstring(scene_xml(assets,kind,size)),kind,feature);root.set('model','pose_cell_a3_'+kind)
    world=root.find('worldbody')
    for geom in root.findall("worldbody/body[@name='screw']/geom"):
        if geom.get('contype')!='0':geom.set('condim','6');geom.set('friction','.8 .005 .001')
    for name in ('left_finger','right_finger'):
        pad=root.find(f".//geom[@name='{name}_thin_pad']");pad.set('condim','6');pad.set('friction','1 .005 .001')
        for geom in root.findall(f".//body[@name='{name}']/geom"):
            geom.set('solref','.003 1');geom.set('solimp','.95 .99 .0001')
    for sign,hemisphere in ((-1,'lower'),(1,'upper')):
        for i,az in enumerate(SETUP['camera_azimuths_deg']):
            a=np.radians(az);eye=CENTER+[.13*np.cos(a),.13*np.sin(a),sign*.12];name=f'cell_{hemisphere}_{i}'
            _camera(world,name,eye,CENTER,26)
            ET.SubElement(world,'light',name=name+'_fill',pos=' '.join(map(str,eye)),dir=' '.join(map(str,CENTER-eye)),diffuse='.16 .16 .16',castshadow='false')
    nest=ET.SubElement(world,'body',name='feed_nest',pos=f'{NEST[0]} {NEST[1]} 0')
    ET.SubElement(nest,'geom',name='feed_post',type='box',pos=f'0 0 {NEST_TOP/2}',size=f'.0045 .0045 {NEST_TOP/2}',material='fixture_blue')
    return ET.tostring(root,encoding='unicode')


class CellScene(AssemblyScene):
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
        self.nest_id=m.body('feed_nest').id

    def initialize_pose(self,rng,spin,flipped=False,empty=False):
        m,d=self.model,self.data;self.mj.mj_resetData(m,d)
        r=(about(SIDE_R[:,2],180) if flipped else np.eye(3))@rz(spin);quat=np.zeros(4);self.mj.mju_mat2Quat(quat,r.ravel())
        d.qpos[:3]=[* (NEST+rng.uniform(-.00015,.00015,2)+np.array([.10 if empty else 0,0])),NEST_TOP+self.spec['height']/2+.001]
        d.qpos[3:7]=quat;self._initialize_additional_bodies();self.mj.mj_forward(m,d)
        for _ in range(300):self.mj.mj_step(m,d)
        shade=rng.uniform(.20,.42);m.mat_rgba[m.material('part_metal').id,:3]=[shade*.95,shade,shade*1.04];self.appearance(rng)

    camera_info=HeldScene.camera_info
    project=HeldScene.project

    def illegal_contacts(self,data=None):
        d=self.data if data is None else data;result=super().illegal_contacts(d)
        for c in d.contact:
            if c.dist>=-.00005:continue
            a,b=(int(self.model.geom_bodyid[i]) for i in (c.geom1,c.geom2))
            if (a==self.nest_id and b in self.robot_bodies) or (b==self.nest_id and a in self.robot_bodies):result.append('robot_nest')
        return result
