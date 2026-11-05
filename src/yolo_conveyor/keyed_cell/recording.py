"""Original physical recording plus an explicitly labeled evaluation plan view."""
import cv2
import numpy as np
from yolo_conveyor.active_pose.recording import camera
from yolo_conveyor.pose_cell.scene import SHAFT_XY
from yolo_conveyor.pose_cell.geometry import rz
from yolo_conveyor.datum_pose.assets import DATUMS
from .scene import FIXTURES
from .guard import GUARD


def plan_view(scene):
    """Physics-state visualization only; this image is never an inference input."""
    s=scene;size=300;scale=10000.;panel=np.full((size,size,3),245,np.uint8)
    def pixels(points):return np.round((np.asarray(points)[:,:2]-SHAFT_XY)*[scale,-scale]+[150,160]).astype(np.int32)
    def disk(center,radius,color):cv2.circle(panel,tuple(pixels([center])[0]),int(round(radius*scale)),color,-1)
    center=s.data.xpos[s.body_id];r=s.data.xmat[s.body_id].reshape(3,3);datum=DATUMS[s.kind]
    if np.linalg.norm(center[:2]-SHAFT_XY)<.04:
        angles=np.linspace(0,2*np.pi,96);outer=.011 if s.kind=='bearing' else .008
        poly=np.array([[outer*np.cos(a),outer*np.sin(a),-s.spec['height']/2] for a in angles])@r.T+center
        cv2.fillPoly(panel,[pixels(poly)],(85,88,92))
        hole=np.array([[s.spec['bore']/2*np.cos(a),s.spec['bore']/2*np.sin(a),-s.spec['height']/2] for a in angles])@r.T+center
        cv2.fillPoly(panel,[pixels(hole)],(245,245,245))
        if s.feature:
            x0,x1=(datum['root'],.016) if s.kind=='bearing' else (0.,datum['root'])
            slot=np.array([[x0,-datum['width']/2,0],[x1,-datum['width']/2,0],[x1,datum['width']/2,0],[x0,datum['width']/2,0]])@r.T+center
            cv2.fillPoly(panel,[pixels(slot)],(245,245,245))
    disk(np.r_[SHAFT_XY,0],s.spec['bore']/2-.001,(155,90,20));spec=FIXTURES[s.kind];rf=rz(s.fixture_angle)
    if s.kind=='gear':
        key=np.array([[x,y,0] for x,y in [(spec['root_radius_m'],-spec['width_m']/2),(spec['tip_radius_m'],-spec['width_m']/2),
            (spec['tip_radius_m'],spec['width_m']/2),(spec['root_radius_m'],spec['width_m']/2)]])@rf.T+np.r_[SHAFT_XY,0]
        cv2.fillPoly(panel,[pixels(key)],(30,145,245))
    else:
        pin=rf@np.array([spec['center_radius_m'],0,0])+np.r_[SHAFT_XY,0];disk(pin,spec['radius_m'],(30,145,245))
    cv2.putText(panel,'Physics plan view',(8,20),0,.55,(25,25,25),1)
    cv2.putText(panel,'Display only - not camera input',(8,291),0,.43,(25,25,25),1)
    return panel


class Recorder:
    def __init__(self,scene,path,title):
        self.scene=scene;self.title=title;self.next=0.;self.snapshot=None;self.message='';self.force=0.;self.torque=0.;self.guard_armed=False
        camera(scene,'overview',[.65,1.,.85],np.array([-.20,.02,.30]),50)
        self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),12,(1280,840))
        if not self.writer.isOpened():raise ValueError('video writer unavailable')

    def sample(self,stage):
        s=self.scene
        if s.data.time<self.next:return
        self.next=float(s.data.time)+1/12;target=s.tool_transform()[:3,3]
        camera(s,'assembly_view',target+[-.14,-.06,.09],target,30)
        frame=np.full((840,1280,3),240,np.uint8);frame[80:720,:640]=s.capture('overview');frame[80:720,640:]=s.capture('assembly_view')
        frame[80:380,980:1280]=plan_view(s)
        if self.snapshot is not None:frame[400:720,:320]=cv2.resize(self.snapshot,(320,320))
        cv2.putText(frame,f'A5 | {self.title} | {stage} | sim {s.data.time:.1f}s',(16,28),0,.65,(25,25,25),1)
        cv2.putText(frame,'RGB pose -> key/pin alignment -> sensor-guarded pre-insertion',(16,57),0,.65,(25,25,25),1)
        cv2.putText(frame,self.message,(16,751),0,.60,(25,25,25),1)
        color=(30,40,210) if self.force>GUARD['force_delta_n'] or self.torque>GUARD['torque_delta_nm'] else (40,100,30)
        cv2.putText(frame,f'Wrist delta: {self.force:.2f} N | {self.torque:.3f} Nm | guard {"armed" if self.guard_armed else "off"}',(16,782),0,.65,color,1)
        cv2.putText(frame,'PHYSICS SIMULATION | closed friction grasp | no object weld or pose reset',(16,818),0,.58,(25,25,25),1)
        self.writer.write(frame)

    def close(self):self.writer.release()
