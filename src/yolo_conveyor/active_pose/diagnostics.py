"""Image-only diagnostics, plus explicitly separate oracle-seed offline analysis."""
import sys
import numpy as np
from yolo_conveyor.pose_cell import metrology
from yolo_conveyor.pose_cell.faces import CENTER_SCHEMA,landmarks,class_id
from yolo_conveyor.pose_cell.profile import mesh
from yolo_conveyor.held.geometry import project


def diagnose(images,predictions,cameras,kind,pixel_scale=2.):
    views={};base=('bearing','gear').index(kind)*2
    for name,image in images.items():
        items=[p for p in predictions[name] if p['class_id'] in (base,base+1) and p['confidence']>=metrology.LIMITS['confidence']]
        accepted=metrology.observe({name:image},{name:predictions[name]},{name:cameras[name]},kind,pixel_scale)
        views[name]=dict(image_std=float(image.std()),detections=len(items),
            bore_confidences=[p['keypoints'][0][2] for p in items],key_confidences=[p['keypoints'][1][2] for p in items],
            usable_observations=len(accepted),ellipses=[o['ellipse'] for o in accepted])
    captured={};profile_values=[];key_values=[]
    def profile(frame,event,arg):
        if event!='return':return
        if frame.f_code is metrology.solve.__code__:
            captured.update(seeds=len(frame.f_locals.get('seeds',[])),candidates=len(frame.f_locals.get('candidates',[])))
        elif frame.f_code is metrology.profile_score.__code__ and arg is not None:profile_values.append(float(arg)/pixel_scale)
        elif frame.f_code is metrology.key_support.__code__ and arg is not None:key_values.append(int(arg[0]))
    previous=sys.getprofile();sys.setprofile(profile)
    try:
        measurement=metrology.measure(images,predictions,cameras,kind,pixel_scale);reason=None
    except (ValueError,np.linalg.LinAlgError) as exc:measurement=None;reason=str(exc)
    finally:sys.setprofile(previous)
    return dict(views=views,solver=captured,profile_errors_raw_px=profile_values,key_support_view_counts=key_values,measurement=measurement,reason=reason)


def oracle_predictions(predictions,cameras,kind,gt):
    """For offline diagnosis only. Never called by runtime or observation policy."""
    r=np.array(gt['rotation']);t=np.array(gt['center']);result={}
    for name,camera in cameras.items():
        d=np.array(camera['T_world_camera_cv'])[:3,3]-t;cos=float(r[:,2]@d/np.linalg.norm(d));face=1 if cos>0 else -1
        if abs(cos)<.28:result[name]=[];continue
        p=landmarks(kind,face);xy=np.array([project(camera,r@q+t) for q in (p[0],p[1:].mean(0))])
        # A projected CAD box is only an ellipse search-scale prior in this oracle diagnostic.
        cloud=mesh(kind)[0]
        uv=np.array([project(camera,r@q+t) for q in cloud]);box=np.r_[uv.min(0),uv.max(0)]
        result[name]=[dict(class_id=class_id(kind,face),confidence=1.,bbox=box.tolist(),point_schema=CENTER_SCHEMA,
                           keypoints=[[float(x),float(y),1.] for x,y in xy]+[[0.,0.,0.]]*3)]
    return result
