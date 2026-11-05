"""Known outer-profile edge agreement disambiguates opposite face hypotheses."""
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from functools import lru_cache
from yolo_conveyor.held.geometry import project
from .faces import PARTS
from .faces import landmarks


def points(kind):
    p=PARTS[kind];h=p['height']/2
    levels=[(-h,.011),(h-.0015,.011),(h-.0015,.0125),(h,.0125)] if kind=='bearing' else [(-h,.008),(-h+.007,.008),(-h+.007,.011205),(h,.0061)]
    angles=np.linspace(0,2*np.pi,96,endpoint=False)
    return np.array([[r*np.cos(a),r*np.sin(a),z] for z,r in levels for a in angles])


def edge_maps(images):
    result={}
    for name,image in images.items():
        if image.std()<3:continue
        gray=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
        edge=cv2.Canny(gray,25,70);result[name]=cv2.distanceTransform(255-edge,cv2.DIST_L2,3)
    return result


@lru_cache(maxsize=2)
def mesh(kind):
    """Compile static authored CAD only. No simulation state or object pose is read."""
    import mujoco
    from yolo_conveyor.simulation.screw_scene import scene_xml
    from yolo_conveyor.annular.assets import build_part
    from yolo_conveyor.datum_pose.assets import replace_part
    root=replace_part(build_part(ET.fromstring(scene_xml(640)),kind),kind,True)
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));body=model.body('screw').id;vertices=[];triangles=[];offset=0
    for g in np.flatnonzero(model.geom_bodyid==body):
        mid=int(model.geom_dataid[g]);va=int(model.mesh_vertadr[mid]);vn=int(model.mesh_vertnum[mid]);fa=int(model.mesh_faceadr[mid]);fn=int(model.mesh_facenum[mid])
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,model.geom_quat[g])
        v=model.mesh_vert[va:va+vn]@rotation.reshape(3,3).T+model.geom_pos[g]
        vertices.append(v);triangles.append(model.mesh_face[fa:fa+fn]+offset);offset+=vn
    return np.vstack(vertices),np.vstack(triangles)


def key_support(r,t,kind,cameras,maps,pixel_scale):
    count=0;costs=[]
    for name,dist in maps.items():
        cam=cameras[name];direction=np.array(cam['T_world_camera_cv'])[:3,3]-t;cos=float(r[:,2]@direction/np.linalg.norm(direction))
        if abs(cos)<.28:continue
        face=1 if cos>0 else -1;points=landmarks(kind,face)
        try:uv=np.array([project(cam,r@p+t) for p in points])
        except ValueError:continue
        pairs=((1,4),(1,2),(4,3)) if kind=='bearing' else ((2,3),(1,2),(4,3));samples=[]
        for a,b in pairs:
            n=max(4,int(np.linalg.norm(uv[a]-uv[b])/pixel_scale));samples.extend(uv[a]+f*(uv[b]-uv[a]) for f in np.linspace(.1,.9,n))
        pts=np.round(samples).astype(int)
        if np.any(pts<0) or np.any(pts>=640):continue
        values=dist[pts[:,1],pts[:,0]];mean=float(np.mean(values));costs.append(mean)
        if mean<=2.5*pixel_scale and np.mean(values<=3*pixel_scale)>=.65:count+=1
    return count,float(np.mean(sorted(costs)[:2])) if len(costs)>=2 else 1e6


def score(r,t,kind,cameras,maps,pixel_scale=1.):
    vertices,triangles=mesh(kind);model=vertices@r.T+t;values=[]
    for name,distance in maps.items():
        T=np.array(cameras[name]['T_world_camera_cv']);camera_points=(model-T[:3,3])@T[:3,:3]
        if np.any(camera_points[:,2]<=0):continue
        uv=camera_points@np.array(cameras[name]['K']).T;uv=uv[:,:2]/uv[:,2:3]
        mask=np.zeros((320,320),np.uint8);pixels=np.round(uv*.5*16).astype(np.int32)
        for triangle in triangles:cv2.fillConvexPoly(mask,pixels[triangle],255,shift=4)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not contours:continue
        samples=np.vstack([c[:,0,:] for c in contours])*2;inside=(samples[:,0]>=0)&(samples[:,0]<640)&(samples[:,1]>=0)&(samples[:,1]<640);samples=samples[inside]
        if len(samples)<20:continue
        residual=np.sort(distance[samples[:,1],samples[:,0]])
        # Known finger occlusion and tooth gaps are not counted as visible silhouette support.
        values.append(float(np.mean(residual[:max(10,int(len(residual)*.65))])))
    return float(np.mean(values)) if len(values)>=2 else 1e6
