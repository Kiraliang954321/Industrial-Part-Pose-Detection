"""Spectator views only; the six detector cameras and physics are unchanged."""
import cv2
import numpy as np
from yolo_conveyor.pose_cell.recording import draw


def camera(scene,name,eye,target,fovy):
    z=np.asarray(eye)-target;z/=np.linalg.norm(z);x=np.cross([0,0,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    i=scene.model.camera(name).id;scene.model.cam_pos[i]=eye
    scene.mj.mju_mat2Quat(scene.model.cam_quat[i],np.column_stack([x,y,z]).ravel());scene.model.cam_fovy[i]=fovy
    scene.mj.mj_camlight(scene.model,scene.data)


class Recorder:
    def __init__(self,scene,path,title):
        self.scene=scene;self.title=title;self.next=0.;self.snapshot=None;self.message=''
        camera(scene,'overview',[.65,1.,.85],np.array([-.20,.02,.30]),50)
        self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),12,(1280,800))
        if not self.writer.isOpened():raise ValueError('video writer unavailable')

    def sample(self,stage):
        s=self.scene
        if s.data.time<self.next:return
        self.next=float(s.data.time)+1/12;target=s.tool_transform()[:3,3]
        camera(s,'assembly_view',target+[-.18,-.08,.12],target,30)
        f=np.full((800,1280,3),240,np.uint8);f[80:720,:640]=s.capture('overview');f[80:720,640:]=s.capture('assembly_view')
        if self.snapshot is not None:f[400:720,:320]=cv2.resize(self.snapshot,(320,320))
        cv2.putText(f,f'A4 | {self.title} | {stage} | sim {s.data.time:.1f}s',(16,28),0,.65,(25,25,25),1)
        cv2.putText(f,'Inspect -> bounded observation turns if needed -> correct -> recheck -> pre-position',(16,57),0,.6,(25,25,25),1)
        cv2.putText(f,self.message,(16,751),0,.60,(25,25,25),1)
        cv2.putText(f,'PHYSICS SIMULATION | observation turns are separate from correction/insertion',(16,780),0,.55,(25,25,25),1)
        self.writer.write(f)

    def close(self):self.writer.release()
