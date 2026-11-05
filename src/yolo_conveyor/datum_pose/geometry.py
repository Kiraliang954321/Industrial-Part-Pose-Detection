"""Rotation conventions and multi-view rigid alignment; no robot/object state access."""
import numpy as np
import cv2
from yolo_conveyor.held.geometry import camera_ray,project


def rz(deg):
    a=np.radians(deg);c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def tilt_rotation(tilt,azimuth):
    a=np.radians(tilt);c,s=np.cos(a),np.sin(a)
    return rz(azimuth)@np.array([[c,0,s],[0,1,0],[-s,0,c]])@rz(-azimuth)


def rotation(tilt,azimuth,spin):return tilt_rotation(tilt,azimuth)@rz(spin)


def describe(r):
    r=np.asarray(r);n=r[:,2];inclination=float(np.degrees(np.arccos(np.clip(n[2],-1,1))))
    raw_azimuth=float(np.degrees(np.arctan2(n[1],n[0]))%360)
    q=tilt_rotation(inclination,raw_azimuth);local=q.T@r
    return dict(inclination_deg=inclination,azimuth_deg=raw_azimuth if inclination>=1. else None,
                spin_deg=float(np.degrees(np.arctan2(local[1,0],local[0,0]))%360))


def errors(predicted,target):
    a=np.asarray(predicted);b=np.asarray(target);u=a[:,2];v=b[:,2]
    axis=float(np.degrees(np.arccos(np.clip(u@v,-1,1))))
    cross=np.cross(u,v);cos=float(u@v)
    if cos<-.99:twist=None
    else:
        skew=np.array([[0,-cross[2],cross[1]],[cross[2],0,-cross[0]],[-cross[1],cross[0],0]])
        align=np.eye(3)+skew+skew@skew/(1+cos)
        x=align@a[:,0];y=b[:,0]
        twist=abs(float(np.degrees(np.arctan2(np.dot(v,np.cross(y,x)),np.dot(y,x)))))
    full=float(np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1))))
    return dict(axis_deg=axis,spin_deg=twist,rotation_deg=full)


def refine_rotation(r,t,anchors,observed,cameras):
    """Optimize only image reprojection of the known CAD points; never use GT/FK."""
    params=np.r_[cv2.Rodrigues(r)[0].ravel(),t*1000]
    def residual(p):
        rot=cv2.Rodrigues(p[:3])[0];translation=p[3:]/1000
        return np.array([project(cameras[name],rot@np.asarray(anchors)[i]+translation)-uv
                         for i,rays in observed.items() for _,_,name,uv in rays]).ravel()
    current=residual(params);damping=.001
    for _ in range(20):
        columns=[]
        for j in range(6):
            step=1e-5 if j<3 else .001;p=params.copy();p[j]+=step
            columns.append((residual(p)-current)/step)
        jac=np.column_stack(columns);normal=jac.T@jac
        delta=np.linalg.solve(normal+damping*np.diag(np.maximum(np.diag(normal),1.)), -jac.T@current)
        scale=max(1.,np.linalg.norm(delta[:3])/.1,np.linalg.norm(delta[3:])/.5);delta/=scale
        candidate=params+delta
        try:error=residual(candidate)
        except ValueError:damping*=10;continue
        if error@error<current@current:
            params=candidate;current=error;damping=max(1e-7,damping/3)
            if np.linalg.norm(delta)<1e-6:break
        else:damping*=10
    return cv2.Rodrigues(params[:3])[0],params[3:]/1000


def estimate(predictions,cameras,anchors,point_conf=.6):
    observed={};world=[];indices=[];ray_residuals=[]
    for i in range(len(anchors)):
        rays=[]
        for name,items in predictions.items():
            if len(items)!=1 or items[0]['confidence']<.5:continue
            point=items[0]['keypoints'][i]
            if len(point)!=3 or not np.isfinite(point).all() or point[2]<point_conf:continue
            if not 0<=point[0]<640 or not 0<=point[1]<640:continue
            origin,direction=camera_ray(cameras[name],point[:2]);rays.append((origin,direction,name,point[:2]))
        if len(rays)<2:continue
        if max(np.linalg.norm(np.cross(a[1],b[1])) for a in rays for b in rays)<.25:continue
        matrices=[np.eye(3)-np.outer(d,d) for _,d,_,_ in rays]
        p=np.linalg.solve(sum(matrices),sum(m@o for m,(o,_,_,_) in zip(matrices,rays)))
        if any((p-o)@d<=0 for o,d,_,_ in rays):continue
        world.append(p);indices.append(i);observed[i]=rays
        ray_residuals.extend(float(np.linalg.norm(m@(p-o))) for m,(o,_,_,_) in zip(matrices,rays))
    if len(indices)<4 or 0 not in indices:raise ValueError('insufficient_multiview_datums')
    x=np.asarray(anchors)[indices];y=np.asarray(world);xc=x.mean(0);yc=y.mean(0)
    u,s,vh=np.linalg.svd((x-xc).T@(y-yc))
    if s[1]<1e-7:raise ValueError('degenerate_datums')
    fix=np.eye(3);fix[2,2]=np.linalg.det(vh.T@u.T);r=vh.T@fix@u.T;t=yc-r@xc
    r,t=refine_rotation(r,t,anchors,observed,cameras)
    residual=float(np.max(np.linalg.norm((x@r.T+t)-y,axis=1)))
    pixels=[float(np.linalg.norm(project(cameras[name],r@np.asarray(anchors)[i]+t)-uv))
            for i,rays in observed.items() for _,_,name,uv in rays]
    return dict(rotation=r.tolist(),translation_m=t.tolist(),angles=describe(r),points_used=indices,
                max_ray_m=max(ray_residuals),max_fit_m=residual,max_reprojection_px=max(pixels))
