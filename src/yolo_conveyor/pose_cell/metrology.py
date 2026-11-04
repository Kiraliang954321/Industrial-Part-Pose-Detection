"""Face-aware multi-camera planar pose: signed faces remove the A2 upward-only assumption."""
import cv2
import numpy as np
from yolo_conveyor.held.geometry import project
from .faces import PARTS,circle_points,CENTER_SCHEMA
from .geometry import describe,error
from .ellipse import fit_bore
from .profile import edge_maps,score as profile_score,key_support
from yolo_conveyor.orientation.geometry import bbox_iou

LIMITS=dict(confidence=.30,point_confidence=.6,min_views=2,max_reprojection_px=3.5,max_profile_px=2.,ambiguity_deg=6.,
            aligned_rotation_deg=7.,aligned_axis_deg=4.,maximum_corrections=3)


def observe(images,predictions,cameras,kind,pixel_scale=1.):
    result=[]
    for name,image in images.items():
        items=predictions[name]
        if image[180:460,180:460].std()<3 or not items:continue
        items=sorted(items,key=lambda p:p['confidence'],reverse=True);base=('bearing','gear').index(kind)*2
        if items[0]['class_id'] not in (base,base+1):continue
        if any(bbox_iou(items[0]['bbox'],p['bbox'])<.6 for p in items[1:] if p['confidence']>.35):continue
        # Two overlapping face labels are alternatives for the same physical object.
        for p in items:
            if p['class_id'] in (base,base+1) and p['confidence']>=LIMITS['confidence']:
                observation=_observation(image,p,cameras[name],name,kind,pixel_scale)
                if observation is not None:result.append(observation)
                alternative=_observation(image,{**p,'class_id':p['class_id']^1},cameras[name],name,kind,pixel_scale)
                if alternative is not None:
                    alternative['geometric_face_alternative']=True;result.append(alternative)
    return result


def _observation(image,p,camera,name,kind,pixel_scale):
    face=1 if p['class_id']%2 else -1
    # Local contrast equalization helps both upward and downward metallic faces.
    # It changes intensities only; geometric coordinates and RGB evidence stay intact.
    points=np.array(p['keypoints'])
    if points[0,2]<LIMITS['point_confidence']:return None
    midpoints=[]
    if p.get('point_schema')==CENTER_SCHEMA:
        if points[1,2]>=LIMITS['point_confidence']:midpoints.append(points[1,:2])
    else:
        for a,b in ((1,3),(2,4)):
            if min(points[a,2],points[b,2])>=LIMITS['point_confidence']:midpoints.append((points[a,:2]+points[b,:2])/2)
    if not midpoints:return None
    key=np.mean(midpoints,axis=0);box=np.asarray(p['bbox']);expected=max(box[2]-box[0],box[3]-box[1])*(.008/.025 if kind=='bearing' else .006/.02241)
    try:ellipse=fit_bore(image,points[0,:2],expected,key-points[0,:2] if kind=='gear' else None,pixel_scale)
    except ValueError:return None
    center=np.array(ellipse['center']);angle=np.radians(ellipse['angle_deg'])
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);mapping=rot@np.diag(np.array(ellipse['sizes'])/2)
    d=np.linalg.solve(mapping,key-center);length=np.linalg.norm(d)
    if not np.isfinite(length) or length<.3:return None
    d/=length;side=-face*np.array([-d[1],d[0]])
    uv=np.array([center+mapping@v/np.sqrt(2) for v in (-d+side,d+side,d-side,-d-side)])
    return dict(view=name,face=face,object_points=circle_points(kind,face),pixels=uv,camera=camera,
                ellipse=ellipse,key_center=key.tolist(),refined_keypoints=points.tolist(),confidence=p['confidence'])

def reprojection(r,t,observation):
    try:return np.array([project(observation['camera'],r@p+t)-uv for p,uv in zip(observation['object_points'],observation['pixels'])])
    except ValueError:return np.full((4,2),1e6)


def facing(r,t,o):
    d=np.array(o['camera']['T_world_camera_cv'])[:3,3]-t
    return o['face']*(r[:,2]@d)/np.linalg.norm(d)>.15


def optimize(r,t,observations):
    params=np.r_[cv2.Rodrigues(r)[0].ravel(),t*1000]
    def residual(p):return np.concatenate([reprojection(cv2.Rodrigues(p[:3])[0],p[3:]/1000,o).ravel() for o in observations])
    current=residual(params);damping=.001
    for _ in range(18):
        columns=[]
        for j in range(6):
            step=1e-5 if j<3 else .001;p=params.copy();p[j]+=step;columns.append((residual(p)-current)/step)
        jac=np.column_stack(columns);normal=jac.T@jac
        delta=np.linalg.solve(normal+damping*np.diag(np.maximum(np.diag(normal),1.)), -jac.T@current)
        delta/=max(1.,np.linalg.norm(delta[:3])/.15,np.linalg.norm(delta[3:])/1.)
        candidate=params+delta;next_error=residual(candidate)
        if next_error@next_error<current@current:
            params=candidate;current=next_error;damping=max(1e-7,damping/3)
            if np.linalg.norm(delta)<1e-6:break
        else:damping*=10
    return cv2.Rodrigues(params[:3])[0],params[3:]/1000


def solve(observations,pixel_scale=1.,images=None,cameras=None,kind=None):
    if len(observations)<LIMITS['min_views']:raise ValueError('insufficient_visible_faces')
    seeds=[]
    for o in observations:
        points=np.ascontiguousarray(o['object_points'],dtype=np.float64);uv=np.ascontiguousarray(o['pixels'],dtype=np.float64)
        try:ret,rvecs,tvecs,_=cv2.solvePnPGeneric(points,uv,np.array(o['camera']['K']),np.zeros(5),flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:continue
        T=np.array(o['camera']['T_world_camera_cv'])
        for rv,tv in zip(rvecs,tvecs):
            r=T[:3,:3]@cv2.Rodrigues(rv)[0];t=T[:3,:3]@tv.ravel()+T[:3,3]
            if facing(r,t,o) and np.isfinite(r).all() and np.isfinite(t).all():seeds.append((r,t))
    candidates=[];maps=edge_maps(images) if images is not None else None
    def select(r,t,threshold):
        grouped={}
        for o in observations:
            residual=float(np.max(np.linalg.norm(reprojection(r,t,o),axis=1)))
            if facing(r,t,o) and residual<threshold and (o['view'] not in grouped or residual<grouped[o['view']][0]):grouped[o['view']]=(residual,o)
        return [item[1] for item in grouped.values()]
    for r,t in seeds:
        # A one-view circle scale error can shift depth by millimetres. This is
        # only the optimizer's initialization window; final acceptance stays 3.5 px.
        support=select(r,t,40*pixel_scale)
        if len(support)<2:continue
        r,t=optimize(r,t,support)
        support=select(r,t,LIMITS['max_reprojection_px']*pixel_scale)
        if len(support)<LIMITS['min_views']:continue
        r,t=optimize(r,t,support);residual=[float(np.linalg.norm(e)) for o in support for e in reprojection(r,t,o)]
        if max(residual)>LIMITS['max_reprojection_px']*pixel_scale:continue
        profile=profile_score(r,t,kind,cameras,maps,pixel_scale) if maps is not None else 0.
        if profile>LIMITS['max_profile_px']*pixel_scale:continue
        key_views,key_cost=key_support(r,t,kind,cameras,maps,pixel_scale) if maps is not None else (2,0.)
        if key_views<2:continue
        candidates.append((len(support),float(np.mean(residual))+.75*profile+.5*key_cost,r,t,support,max(residual),profile))
    if not candidates:raise ValueError('inconsistent_face_geometry')
    # Circular bores can support both signed normals in several cameras. View
    # count alone must not override the asymmetric flange/hub/key image evidence.
    # Every candidate already has at least two independent supporting cameras.
    candidates.sort(key=lambda c:(c[1],-c[0]));best=candidates[0]
    if any(c[1]<=best[1]+.75*pixel_scale and error(c[2],best[2])['rotation_deg']>LIMITS['ambiguity_deg'] for c in candidates[1:]):
        raise ValueError('ambiguous_signed_orientation')
    return dict(rotation=best[2].tolist(),translation_m=best[3].tolist(),angles=describe(best[2]),supported_views=[o['view'] for o in best[4]],
                visible_faces={o['view']:o['face'] for o in best[4]},max_reprojection_px=best[5],image_fit_score=best[1],profile_error_px=best[6])


def measure(images,predictions,cameras,kind,pixel_scale=1.):return solve(observe(images,predictions,cameras,kind,pixel_scale),pixel_scale,images,cameras,kind)
