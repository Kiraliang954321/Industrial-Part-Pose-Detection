from copy import deepcopy
import numpy as np
from yolo_conveyor.datum_pose.crop import crop_image


def roi_inputs(images,predictions,cameras):
    images={v:crop_image(im) for v,im in images.items()};predictions=deepcopy(predictions);cameras=deepcopy(cameras)
    for c in cameras.values():c['K']=(np.array([[2,0,-320],[0,2,-320],[0,0,1.]])@np.array(c['K'])).tolist()
    for items in predictions.values():
        for p in items:
            p['bbox']=(np.array(p['bbox'])*2-320).tolist();points=np.array(p['keypoints']);points[:,:2]=points[:,:2]*2-320;p['keypoints']=points.tolist()
    return images,predictions,cameras
