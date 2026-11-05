"""RGB-only landmark prediction and calibrated rigid orientation measurement."""
import numpy as np
from itertools import combinations
from copy import deepcopy
from .training import verify_model
from .geometry import estimate,errors
from .assets import anchors,NAMES
from .crop import crop_image
from .refinement import refine,metrology_points
from .assets import PARTS
from yolo_conveyor.held.geometry import project

POLICY=dict(point_confidence=.6,max_reprojection_px=4.,max_fit_m=.0008,max_ray_m=.0008,
            axis_tolerance_deg=10.,spin_tolerance_deg=10.,guard_deg=4.,max_measured_tilt_deg=38.)


class Predictor:
    def __init__(self,weights):
        import torch
        from ultralytics import YOLO
        self.model=YOLO(str(weights));verify_model(self.model);self.torch=torch

    def predict(self,images,preprocessed=False):
        outputs=self.model.predict([im if preprocessed else crop_image(im) for im in images.values()],imgsz=640,device='0',conf=.3,verbose=False)
        result={name:[dict(class_id=int(cls),confidence=float(conf),bbox=box,keypoints=kp)
                      for box,cls,conf,kp in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),
                                                 result.boxes.conf.cpu().tolist(),result.keypoints.data.cpu().tolist())]
                for name,result in zip(images,outputs)}
        if not preprocessed:
            for items in result.values():
                for item in items:
                    item['bbox']=(np.array(item['bbox'])/2+160).tolist()
                    points=np.array(item['keypoints']);points[:,:2]=points[:,:2]/2+160;item['keypoints']=points.tolist()
        return result

    def sync(self):self.torch.cuda.synchronize()


def _measure(images,predictions,cameras,kind,pixel_scale=1.,model_points=None):
    usable={}
    for name,image in images.items():
        if image.shape!=(640,640,3) or image[200:440,200:440].std()<3:continue
        items=predictions[name]
        if len(items)==1 and items[0]['class_id']==('bearing','gear').index(kind):usable[name]=items
    if len(usable)<2:raise ValueError('insufficient_part_views')
    result=estimate(usable,cameras,anchors(kind) if model_points is None else model_points,POLICY['point_confidence'])
    if (result['max_fit_m']>POLICY['max_fit_m'] or result['max_ray_m']>POLICY['max_ray_m']
            or result['max_reprojection_px']>POLICY['max_reprojection_px']*pixel_scale):raise ValueError('inconsistent_datum_geometry')
    if result['angles']['inclination_deg']>POLICY['max_measured_tilt_deg']:raise ValueError('outside_validated_axis_envelope')
    return result


def measure(images,predictions,cameras,kind,pixel_scale=1.):
    refined={name:refine(images[name],items,kind,pixel_scale) for name,items in predictions.items()}
    cad=anchors(kind);z=cad[0,2];radius=PARTS[kind]['bore']/2
    model_points=np.array([cad[0],[radius,0,z],[0,radius,z],[0,-radius,z]])
    metrology={}
    for name,items in refined.items():
        metrology[name]=[]
        for item in items:
            points=metrology_points(item)
            if points is not None:metrology[name].append({**item,'keypoints':points})
    candidates=[];names=list(images);threshold=POLICY['max_reprojection_px']*pixel_scale
    for views in [tuple(names),*combinations(names,2)]:
        for dropped in (None,):
            subset={name:deepcopy(metrology[name]) for name in views}
            if dropped is not None:
                for items in subset.values():
                    for p in items:p['keypoints'][dropped][2]=0.
            try:fitted=_measure({v:images[v] for v in views},subset,cameras,kind,pixel_scale,model_points)
            except (ValueError,np.linalg.LinAlgError):continue
            r=np.array(fitted['rotation']);t=np.array(fitted['translation_m']);per_view={};residual=[]
            for name,items in metrology.items():
                if len(items)!=1 or items[0]['class_id']!=('bearing','gear').index(kind) or items[0]['confidence']<.5:continue
                inliers=[]
                for i,(point,kp) in enumerate(zip(model_points,items[0]['keypoints'])):
                    if kp[2]<POLICY['point_confidence']:continue
                    error=float(np.linalg.norm(project(cameras[name],r@point+t)-kp[:2]))
                    if error<=threshold:inliers.append(i);residual.append(error)
                per_view[name]=inliers
            supported=[n for n,p in per_view.items() if len(p)>=4 and 0 in p]
            if len(supported)<2:continue
            support=sum(len(p) for p in per_view.values());mean=float(np.mean(residual))
            fitted.update(supported_views=supported,inliers=per_view,refined_predictions=refined,metrology_predictions=metrology,
                          method='RGB bore ellipse plus mechanical key center; upward-axis envelope')
            candidates.append((support,-mean,fitted))
    if not candidates:raise ValueError('insufficient_consistent_image_datums')
    candidates.sort(key=lambda c:(c[0],c[1]),reverse=True);best=candidates[0]
    for alternative in candidates[1:]:
        if (alternative[0]>=best[0]-1 and -alternative[1]<=-best[1]+.25*pixel_scale
                and errors(alternative[2]['rotation'],best[2]['rotation'])['rotation_deg']>6):
            raise ValueError('ambiguous_multiview_orientation')
    return best[2]


def compare(measurement,target,limits=None):
    error=errors(measurement['rotation'],target)
    if error['spin_deg'] is None:return dict(decision='UNCERTAIN',reason='axis_flip',error=error)
    config=POLICY if limits is None else limits
    axis,spin=error['axis_deg'],error['spin_deg'];a=config['axis_tolerance_deg'];b=config['spin_tolerance_deg'];g=config['guard_deg']
    if axis<=a-g and spin<=b-g:decision,reason='PASS','within_target_guard'
    elif axis>a+g or spin>b+g:decision,reason='REJECT','target_orientation_mismatch'
    else:decision,reason='UNCERTAIN','near_target_boundary'
    return dict(decision=decision,reason=reason,error=error)
