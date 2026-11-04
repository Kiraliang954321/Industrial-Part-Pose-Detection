"""Actual collision keys/pins and a wrist force/torque sensor."""
import xml.etree.ElementTree as ET
import numpy as np
from yolo_conveyor.pose_cell.scene import CellScene,xml as base_xml,SETUP as BASE_SETUP,SHAFT_XY,SHAFT_TOP,ENGAGEMENT
from yolo_conveyor.pose_cell.faces import PARTS
from yolo_conveyor.pose_cell.geometry import rz
from yolo_conveyor.simulation.screw_scene import ScrewScene
from yolo_conveyor.simulation.robot_assets import audit_panda

FIXTURES={
    'gear':dict(type='shaft_key',root_radius_m=.0017,tip_radius_m=.00425,width_m=.001,top_m=SHAFT_TOP,bottom_m=SHAFT_TOP-.008),
    'bearing':dict(type='index_pin',center_radius_m=.0102,radius_m=.0007,segments=32,top_m=SHAFT_TOP,bottom_m=SHAFT_TOP-.008)}
SETUP={**BASE_SETUP,'stage':'A5','indexing_fixtures':FIXTURES,'sensor':'ideal simulated wrist force and torque',
       'engagement_m':ENGAGEMENT,'scope':'project-authored clearance key/pin proxies; no real fit certification'}


def xml(assets,kind,feature,fixture_angle,size):
    root=ET.fromstring(base_xml(assets,kind,feature,size));root.set('model','keyed_cell_a5_'+kind)
    asset=root.find('asset');ET.SubElement(asset,'material',name='index_orange',rgba='.95 .45 .08 1',specular='.2')
    fixture=root.find("worldbody/body[@name='shaft_fixture']");spec=FIXTURES[kind];angle=np.radians(fixture_angle)
    contact=dict(solref='.003 1',solimp='.95 .99 .0001',friction='.3 .005 .0001',condim='4')
    # The existing shaft remains the physical centering feature.
    for g in fixture.findall('geom'):
        for key,value in contact.items():g.set(key,value)
    if kind=='gear':
        mid=(spec['root_radius_m']+spec['tip_radius_m'])/2;r=rz(fixture_angle)
        center=r@np.array([mid,0,(spec['top_m']+spec['bottom_m'])/2])
        ET.SubElement(fixture,'geom',name='shaft_key',type='box',pos=' '.join(map(str,center)),
            size=f"{(spec['tip_radius_m']-spec['root_radius_m'])/2} {spec['width_m']/2} {(spec['top_m']-spec['bottom_m'])/2}",
            quat=f'{np.cos(angle/2)} 0 0 {np.sin(angle/2)}',material='index_orange',**contact)
    else:
        xy=spec['center_radius_m']*np.array([np.cos(angle),np.sin(angle)])
        vertices=np.array([[spec['radius_m']*np.cos(a),spec['radius_m']*np.sin(a),z]
            for z in (spec['bottom_m'],spec['top_m']) for a in np.linspace(0,2*np.pi,spec['segments'],endpoint=False)])
        ET.SubElement(asset,'mesh',name='index_pin_prism',vertex=' '.join(map(str,vertices.ravel())))
        ET.SubElement(fixture,'geom',name='index_pin',type='mesh',mesh='index_pin_prism',pos=f'{xy[0]} {xy[1]} 0',
                      quat=f'{np.cos(angle/2)} 0 0 {np.sin(angle/2)}',material='index_orange',**contact)
        low=.04;high=spec['bottom_m']
        ET.SubElement(fixture,'geom',name='index_pin_support',type='cylinder',pos=f'{xy[0]} {xy[1]} {(low+high)/2}',
                      size=f'.0018 {(high-low)/2}',material='fixture_blue',**contact)
    hand=root.find(".//body[@name='hand']")
    ET.SubElement(hand,'site',name='a5_wrist_sensor',pos='0 0 .115',size='.001',rgba='0 0 0 0')
    sensors=root.find('sensor')
    if sensors is None:sensors=ET.SubElement(root,'sensor')
    ET.SubElement(sensors,'force',name='a5_wrist_force',site='a5_wrist_sensor')
    ET.SubElement(sensors,'torque',name='a5_wrist_torque',site='a5_wrist_sensor')
    return ET.tostring(root,encoding='unicode')


class KeyedScene(CellScene):
    def __init__(self,assets,kind,feature=True,fixture_angle=0.,size=640):
        self.kind=kind;self.spec=PARTS[kind];self.feature=feature;self.fixture_angle=float(fixture_angle);self.asset_root=audit_panda(assets)
        ScrewScene.__init__(self,size,xml=xml(self.asset_root,kind,feature,fixture_angle,size))
        m=self.model;joints=[m.joint(f'joint{i}').id for i in range(1,8)]
        self.arm_qpos=m.jnt_qposadr[joints].copy();self.arm_dofs=m.jnt_dofadr[joints].copy();self.arm_ranges=m.jnt_range[joints].copy()
        self.arm_actuators=np.array([m.actuator(f'actuator{i}').id for i in range(1,8)]);self.gripper_actuator=m.actuator('actuator8').id
        self.finger_qpos=np.array([m.jnt_qposadr[m.joint(f'finger_joint{i}').id] for i in (1,2)]);self.tcp_id=m.site('grasp_tcp').id
        self.robot_bodies={m.body(n).id for n in [*(f'link{i}' for i in range(8)),'hand','left_finger','right_finger']}
        self.finger_bodies={m.body(n).id for n in ('left_finger','right_finger')}
        self.robot_geoms={i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in self.robot_bodies}
        self.table_id=m.geom('table').id;self.base_id=m.body('link0').id;self.fixture_id=m.body('shaft_fixture').id;self.nest_id=m.body('feed_nest').id
        self.force_adr=int(m.sensor_adr[m.sensor('a5_wrist_force').id]);self.torque_adr=int(m.sensor_adr[m.sensor('a5_wrist_torque').id])
        self.index_id=m.geom(FIXTURES[kind]['type']).id
        self.fixture_geoms={i for i in range(m.ngeom) if int(m.geom_bodyid[i])==self.fixture_id}
        if m.body_gravcomp[self.body_id]!=0:raise ValueError('workpiece gravity compensation forbidden')

    def wrist_wrench(self):
        return np.r_[self.data.sensordata[self.force_adr:self.force_adr+3],self.data.sensordata[self.torque_adr:self.torque_adr+3]].copy()
