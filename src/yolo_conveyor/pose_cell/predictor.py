import numpy as np
from yolo_conveyor.datum_pose.crop import crop_image
from .training import verify_model
from .faces import CORNER_SCHEMA


class Predictor:
    def __init__(self,weights):
        from ultralytics import YOLO
        import torch
        self.model=YOLO(str(weights));verify_model(self.model);self.torch=torch
        self.schema=getattr(self.model.model,'a3_point_schema',CORNER_SCHEMA)

    def predict(self,images,preprocessed=False):
        outputs=self.model.predict([im if preprocessed else crop_image(im) for im in images.values()],imgsz=640,device='0',conf=.25,verbose=False)
        result={name:[dict(class_id=int(cls),confidence=float(score),bbox=box,keypoints=kp,point_schema=self.schema) for box,cls,score,kp in
                      zip(o.boxes.xyxy.cpu().tolist(),o.boxes.cls.cpu().tolist(),o.boxes.conf.cpu().tolist(),o.keypoints.data.cpu().tolist())] for name,o in zip(images,outputs)}
        if not preprocessed:
            for items in result.values():
                for p in items:
                    p['bbox']=(np.array(p['bbox'])/2+160).tolist();k=np.array(p['keypoints']);k[:,:2]=k[:,:2]/2+160;p['keypoints']=k.tolist()
        return result

    def sync(self):self.torch.cuda.synchronize()
