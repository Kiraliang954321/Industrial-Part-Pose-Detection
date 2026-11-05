"""Physics render recording; overlay visual predictions only after inspection."""
import cv2
import numpy as np
from yolo_conveyor.orientation.evaluate import draw_predictions
from .scene import GUIDE_XY,GUIDE_TOP


class Recorder:
    def __init__(self,scene,path,title):
        self.scene=scene;self.title=title;self.next_time=0.;self.frames=0;self.evidence=None
        self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),12,(1280,768))
        if not self.writer.isOpened():raise RuntimeError('video encoder unavailable')

    def assembly_view(self):
        s=self.scene;cam=s.mj.MjvCamera();cam.lookat[:]=[*GUIDE_XY,GUIDE_TOP+.025]
        cam.distance=.22;cam.azimuth=45;cam.elevation=-20
        option=s.mj.MjvOption();option.sitegroup[:]=0
        s.renderer.update_scene(s.data,camera=cam,scene_option=option)
        return cv2.cvtColor(s.renderer.render().copy(),cv2.COLOR_RGB2BGR)

    def sample(self,stage):
        now=float(self.scene.data.time)
        if now<self.next_time:return
        self.next_time=now+1/12
        frame=np.full((768,1280,3),240,np.uint8)
        frame[80:720,:640]=self.scene.capture('overview')
        frame[80:720,640:]=self.assembly_view()
        if self.evidence is not None:
            for i,name in enumerate(('inspect_a','inspect_b')):
                rendered=draw_predictions(self.evidence['images'][name],self.evidence['predictions'][name],name+' | inspection snapshot')
                frame[440:720,i*280:i*280+280]=cv2.resize(rendered,(280,280))
        cv2.putText(frame,f'{self.title} | {stage} | sim {now:.1f}s',(16,26),cv2.FONT_HERSHEY_SIMPLEX,.64,(30,30,30),1)
        cv2.putText(frame,'PHYSICS + RGB PREDICTIONS | held pre-insertion only; no thread tightening',(16,54),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,30),1)
        cv2.putText(frame,'Panda overview                                       Guide bore / held screw detail',(16,748),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,30),1)
        self.writer.write(frame);self.frames+=1

    def close(self):
        self.writer.release()
