import cv2
import numpy as np
from .scene import VIEWS
from yolo_conveyor.held.geometry import project


def draw(image,predictions,camera,measurement):
    out=image.copy()
    if measurement:predictions=measurement.get('refined_predictions',{}).get(camera['name'],predictions)
    for p in predictions:
        for x,y,confidence in p['keypoints']:
            if confidence>=.6:cv2.circle(out,(int(round(x)),int(round(y))),3,(0,200,230),-1)
    if measurement:
        r=np.array(measurement['rotation']);t=np.array(measurement['translation_m']);origin=tuple(np.round(project(camera,t)).astype(int))
        for i,color in enumerate(((20,20,240),(30,220,30),(240,100,30))):
            tip=tuple(np.round(project(camera,t+r[:,i]*.008)).astype(int));cv2.arrowedLine(out,origin,tip,color,2,tipLength=.15)
    return out


class Recorder:
    def __init__(self,scene,path,title):
        self.scene=scene;self.title=title;self.next=0.;self.panel=None;self.lines=[]
        self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),12,(1280,800))
        if not self.writer.isOpened():raise RuntimeError('video writer')

    def sample(self,stage):
        s=self.scene
        if s.data.time<self.next:return
        self.next=float(s.data.time)+1/12;frame=np.full((800,1280,3),240,np.uint8)
        frame[80:720,:640]=s.capture('overview');frame[80:720,640:]=self.panel if self.panel is not None else s.capture(VIEWS[0])
        cv2.putText(frame,f'A2 | {self.title} | {stage} | sim {s.data.time:.1f}s',(16,28),0,.65,(25,25,25),1)
        cv2.putText(frame,'RGB-derived continuous orientation | mechanical datums | no GT or robot angle supplied to estimator',(16,55),0,.52,(25,25,25),1)
        for i,line in enumerate(self.lines[:2]):cv2.putText(frame,line,(16,748+26*i),0,.62,(25,25,25),1)
        self.writer.write(frame)

    def close(self):self.writer.release()
