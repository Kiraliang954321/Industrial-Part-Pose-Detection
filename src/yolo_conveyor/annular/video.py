"""Actual physics views with the latest raw RGB detection evidence."""
import cv2
import numpy as np


def draw(image,predictions,title):
    result=image.copy()
    for p in predictions:
        x0,y0,x1,y1=np.round(p['bbox_xyxy']).astype(int)
        cv2.rectangle(result,(x0,y0),(x1,y1),(30,220,50),2)
        cv2.putText(result,f"{p['name']} {p['confidence']:.2f}",(max(4,x0),max(45,y0-8)),0,.5,(20,20,240),1)
    cv2.putText(result,title,(10,26),0,.55,(20,20,220),1);return result


class Recorder:
    def __init__(self,scene,path,title):
        self.scene=scene;self.title=title;self.next=0.;self.evidence=None
        self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),12,(1280,768))
        if not self.writer.isOpened():raise RuntimeError('video unavailable')

    def sample(self,stage):
        if self.scene.data.time<self.next:return
        self.next=float(self.scene.data.time)+1/12
        frame=np.full((768,1280,3),242,np.uint8)
        frame[80:720,:640]=self.scene.capture('overview');frame[80:720,640:]=self.scene.capture('assembly_view')
        if self.evidence is not None:frame[400:720,:320]=cv2.resize(self.evidence,(320,320))
        cv2.putText(frame,f'A1 | {self.title} | {stage} | sim {self.scene.data.time:.1f}s',(16,28),0,.65,(30,30,30),1)
        cv2.putText(frame,'RGB orientation gate -> clearance shaft pre-positioning; no press fit or gear meshing',(16,57),0,.6,(30,30,30),1)
        self.writer.write(frame)

    def close(self):self.writer.release()
