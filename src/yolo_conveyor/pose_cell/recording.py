import cv2
import numpy as np
from yolo_conveyor.held.geometry import project
from .scene import VIEWS
from .faces import CENTER_SCHEMA


def draw(image,predictions,camera,measurement):
    result=image.copy()
    for p in predictions:
        for x,y,v in p['keypoints'][:2] if p.get('point_schema')==CENTER_SCHEMA else p['keypoints']:
            if v>=.6:cv2.circle(result,(int(round(x)),int(round(y))),3,(0,200,230),-1)
    if measurement:
        r=np.array(measurement['rotation']);t=np.array(measurement['translation_m']);origin=tuple(np.round(project(camera,t)).astype(int))
        for i,col in enumerate(((20,30,240),(20,220,20),(240,100,20))):cv2.arrowedLine(result,origin,tuple(np.round(project(camera,t+r[:,i]*.01)).astype(int)),col,2,tipLength=.15)
    return result


class Recorder:
    def __init__(self,scene,path,title):
        self.scene=scene;self.title=title;self.next=0.;self.snapshot=None;self.message=''
        self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),12,(1280,800))
        if not self.writer.isOpened():raise RuntimeError('video writer')
    def sample(self,stage):
        s=self.scene
        if s.data.time<self.next:return
        self.next=float(s.data.time)+1/12;f=np.full((800,1280,3),240,np.uint8)
        f[80:720,:640]=s.capture('overview');f[80:720,640:]=s.capture('assembly_view')
        if self.snapshot is not None:f[400:720,:320]=cv2.resize(self.snapshot,(320,320))
        cv2.putText(f,f'A3 | {self.title} | {stage} | sim {s.data.time:.1f}s',(16,28),0,.65,(25,25,25),1)
        cv2.putText(f,'RGB signed-face pose -> visual correction -> fresh verification -> shaft pre-position',(16,57),0,.6,(25,25,25),1)
        cv2.putText(f,self.message,(16,751),0,.60,(25,25,25),1);cv2.putText(f,'PHYSICS SIMULATION | finite friction grip; no object weld or pose reset',(16,780),0,.55,(25,25,25),1)
        self.writer.write(f)
    def close(self):self.writer.release()
