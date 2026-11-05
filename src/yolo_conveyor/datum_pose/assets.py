"""Mechanical cutouts for custom proxies; neither asset is a manufacturer CAD."""
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
import numpy as np
from yolo_conveyor.annular.assets import PARTS

NAMES=['notched_outer_bearing','keyed_bevel_gear']
KEYPOINTS=['bore_center','near_plus','far_plus','far_minus','near_minus']
DATUMS={'bearing':dict(root=.009,width=.0045,kind='outer_notch'),
        'gear':dict(root=.0045,width=.0024,kind='bore_keyway')}


def anchors(kind):
    p=PARTS[kind];d=DATUMS[kind];w=d['width']/2;z=-p['height']/2
    if kind=='bearing':near=d['root'];far=math.sqrt((p['diameter']/2)**2-w*w)
    else:near=math.sqrt((p['bore']/2)**2-w*w);far=d['root']
    return np.array([[0,0,z],[near,w,z],[far,w,z],[far,-w,z],[near,-w,z]])


def clip(poly,axis,bound,less):
    result=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        va=(a[axis]-bound)*(1 if less else -1);vb=(b[axis]-bound)*(1 if less else -1)
        if va<=1e-10:result.append(a)
        if (va<0<vb) or (vb<0<va):result.append(a+(b-a)*va/(va-vb))
    unique=[]
    for p in result:
        if not unique or np.linalg.norm(p-unique[-1])>1e-9:unique.append(p)
    if len(unique)>1 and np.linalg.norm(unique[0]-unique[-1])<1e-9:unique.pop()
    return unique


def cut_ring(asset,body,name,inner,outer0,outer1,z0,z1,mass,kind,feature,material='part_metal'):
    d=DATUMS[kind];w=d['width']/2;root=d['root'];pieces=[]
    regions=[[]]
    if feature:
        regions=[[(0,root,True)],[(0,root,False),(1,w,False)],[(0,root,False),(1,-w,True)]] if kind=='bearing' else [
            [(0,0,True)],[(0,root,False)],[(0,0,False),(0,root,True),(1,w,False)],[(0,0,False),(0,root,True),(1,-w,True)]]
    for i in range(48):
        a,b=2*math.pi*i/48,2*math.pi*(i+1)/48
        for region in regions:
            layers=[]
            for outer,z in ((outer0,z0),(outer1,z1)):
                poly=[np.array([r*math.cos(t),r*math.sin(t)]) for r,t in ((inner,a),(outer,a),(outer,b),(inner,b))]
                for axis,bound,less in region:poly=clip(poly,axis,bound,less) if poly else []
                layers.extend([[*p,z] for p in poly])
            if len(layers)>=6:
                points=np.array(layers)
                if np.linalg.matrix_rank(points-points.mean(0),tol=1e-8)==3:pieces.append(points)
    for i,vertices in enumerate(pieces):
        key=f'{name}_{i}'
        ET.SubElement(asset,'mesh',name=key,vertex=' '.join(map(str,vertices.ravel())))
        ET.SubElement(body,'geom',name=key,type='mesh',mesh=key,material=material,mass=str(mass/len(pieces)),
                      friction='.8 .005 .0001',condim='4',solref='.003 1',solimp='.95 .99 .0001')


def replace_part(root,kind,feature=True):
    body=root.find("worldbody/body[@name='screw']");asset=root.find('asset');p=PARTS[kind];h=p['height']/2;inner=p['bore']/2
    teeth=[deepcopy(g) for g in body.findall('geom') if g.get('name','').startswith('tooth_')]
    for child in list(body):body.remove(child)
    ET.SubElement(body,'freejoint')
    if kind=='bearing':
        cut_ring(asset,body,'datum_barrel',inner,.011,.011,-h,h-.0015,p['mass']*.78,kind,feature)
        cut_ring(asset,body,'datum_flange',inner,.0125,.0125,h-.0015,h,p['mass']*.22,kind,feature)
    else:
        top=-h+p['hub_width']
        cut_ring(asset,body,'datum_hub',inner,.008,.008,-h,top,p['mass']*.45,kind,feature)
        cut_ring(asset,body,'datum_cone',inner,.0102,.0055,top,h,p['mass']*.55,kind,feature)
        for g in teeth:body.append(g)
    return root
