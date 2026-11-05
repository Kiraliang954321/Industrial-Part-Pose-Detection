"""Calibrated two-view endpoints, observable axis and prediction-only compensation."""
import math
import numpy as np


def camera_ray(camera,pixel):
    k=np.asarray(camera["K"],float);t=np.asarray(camera["T_world_camera_cv"],float);p=np.asarray(pixel,float)
    if k.shape!=(3,3) or t.shape!=(4,4) or p.shape!=(2,) or not all(np.isfinite(x).all() for x in (k,t,p)):
        raise ValueError("invalid camera or pixel")
    direction=t[:3,:3]@np.linalg.solve(k,[*p,1.]);direction/=np.linalg.norm(direction)
    return t[:3,3],direction


def project(camera,point):
    t=np.asarray(camera["T_world_camera_cv"],float);k=np.asarray(camera["K"],float)
    p=(np.asarray(point)-t[:3,3])@t[:3,:3]
    if p[2]<=0:raise ValueError("point behind camera")
    uv=k@p;return uv[:2]/uv[2]


def triangulate(camera_a,pixel_a,camera_b,pixel_b):
    ca,da=camera_ray(camera_a,pixel_a);cb,db=camera_ray(camera_b,pixel_b)
    if np.linalg.norm(np.cross(da,db))<.15:raise ValueError("stereo rays nearly parallel")
    distance=np.linalg.lstsq(np.column_stack([da,-db]),cb-ca,rcond=None)[0]
    if np.any(distance<=0):raise ValueError("triangulated point behind camera")
    a,b=ca+distance[0]*da,cb+distance[1]*db
    point=(a+b)/2
    error=max(np.linalg.norm(project(camera_a,point)-pixel_a),np.linalg.norm(project(camera_b,point)-pixel_b))
    return point,float(np.linalg.norm(a-b)),float(error)


def stereo_estimate(predictions,cameras,min_point_conf=.5):
    names=("inspect_a","inspect_b")
    if any(len(predictions[name])!=1 for name in names):raise ValueError("no_unique_part_in_both_views")
    for name in names:
        p=predictions[name][0]
        if p["heading_deg"] is None or min(p["keypoint_confidences"])<min_point_conf:
            raise ValueError("unreliable_keypoints")
    points=[];gaps=[];errors=[]
    for i in (0,1):
        p,g,e=triangulate(cameras[names[0]],predictions[names[0]][0]["keypoints_xy"][i],
                          cameras[names[1]],predictions[names[1]][0]["keypoints_xy"][i])
        points.append(p);gaps.append(g);errors.append(e)
    direction=points[0]-points[1];length=float(np.linalg.norm(direction))
    if not .035<=length<=.051:raise ValueError("implausible_part_length")
    return {"head_world_m":points[0].tolist(),"tip_world_m":points[1].tolist(),
            "axis_world":(direction/length).tolist(),"length_m":length,
            "ray_gap_m":max(gaps),"reprojection_error_px":max(errors)}


def align_to_vertical(axis):
    a=np.asarray(axis,float);a/=np.linalg.norm(a);b=np.array([0.,0.,1.]);v=np.cross(a,b);c=float(a@b)
    if c<0:raise ValueError("upside-down part requires regrasp")
    if np.linalg.norm(v)<1e-10:return np.eye(3)
    skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+skew+skew@skew/(1+c)


def compensated_target(estimate,tool_transform,tip_target):
    """Only estimated endpoints and robot FK; no object truth is accepted."""
    t=np.asarray(tool_transform,float);tip=np.asarray(estimate["tip_world_m"])
    tip_local=t[:3,:3].T@(tip-t[:3,3])
    rotation=align_to_vertical(estimate["axis_world"])@t[:3,:3]
    return np.asarray(tip_target)-rotation@tip_local,rotation


def observable_measures(estimate,tool_transform):
    t=np.asarray(tool_transform,float);tip_local=t[:3,:3].T@(np.asarray(estimate["tip_world_m"])-t[:3,3])
    axis=np.asarray(estimate["axis_world"])
    return {"tilt_deg":math.degrees(math.acos(float(np.clip(axis[2],-1,1)))),
            "tip_protrusion_m":float(-tip_local[0]),"tip_lateral_offset_m":float(np.linalg.norm(tip_local[1:])),
            "length_m":estimate["length_m"]}
