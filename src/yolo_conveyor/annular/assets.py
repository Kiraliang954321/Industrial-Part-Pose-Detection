"""Project-authored open annular collision proxies, with catalog dimensions as references."""
import math
import xml.etree.ElementTree as ET

PARTS={
    'bearing':dict(reference='F608ZZ',bore=.008,diameter=.022,height=.007,flange_diameter=.025,flange_width=.0015,mass=.013,
                   source='https://koyo.jtekt.co.jp/en/products/detail/?pno=F608+ZZ',front='flange_up'),
    'gear':dict(reference='SB1-2040',bore=.006,diameter=.02241,height=.01378,hub_diameter=.016,hub_width=.007,teeth=20,mass=.019,
                source='https://catalog.khkgears.us/item/bevel-gears/steel-bevel-gears-sb/sb1-2040',front='toothed_cone_up_hub_down')}
NAMES=['bearing_front','bearing_back','bearing_tilted','gear_front','gear_back','gear_tilted']


def annulus(asset,body,name,inner,outer0,outer1,z0,z1,segments=48,material='part_metal',mass=0):
    """Each convex sector leaves a real bore; this is not one filled convex hull."""
    for i in range(segments):
        a,b=2*math.pi*i/segments,2*math.pi*(i+1)/segments;vertices=[]
        for z,outer in ((z0,outer0),(z1,outer1)):
            for radius,angle in ((inner,a),(outer,a),(outer,b),(inner,b)):
                vertices.extend([radius*math.cos(angle),radius*math.sin(angle),z])
        key=f'{name}_{i}'
        ET.SubElement(asset,'mesh',name=key,vertex=' '.join(map(str,vertices)))
        ET.SubElement(body,'geom',name=key,type='mesh',mesh=key,material=material,mass=str(mass/segments),
                      friction='0.8 0.005 0.0001',condim='4',solref='0.003 1',solimp='0.95 0.99 0.0001')


def build_part(root,kind):
    spec=PARTS[kind];asset=root.find('asset');body=root.find("worldbody/body[@name='screw']")
    # Keep the legacy free-body identifier for the inherited Panda bindings only.
    for child in list(body):body.remove(child)
    ET.SubElement(body,'freejoint');h=spec['height']/2;inner=spec['bore']/2
    ET.SubElement(asset,'material',name='part_metal',rgba='.28 .30 .32 1',specular='.35',shininess='.35')
    ET.SubElement(asset,'material',name='shield_metal',rgba='.23 .25 .27 1',specular='.3',shininess='.3')
    if kind=='bearing':
        annulus(asset,body,'bearing_body',inner,.011,.011,-h,h-.0015,mass=spec['mass']*.78)
        annulus(asset,body,'bearing_flange',inner,.0125,.0125,h-.0015,h,mass=spec['mass']*.22)
        # Symmetric face recess bands represent shields; no artificial face color code.
        for sign in (-1,1):
            z=sign*(h+.00002)
            annulus(asset,body,f'shield_{sign}',.0052,.0095,.0095,z-.00001,z+.00001,material='shield_metal')
    else:
        hub_top=-h+spec['hub_width']
        annulus(asset,body,'gear_hub',inner,.008,.008,-h,hub_top,mass=spec['mass']*.45)
        annulus(asset,body,'gear_core',inner,.0102,.0055,hub_top,h,mass=spec['mass']*.55)
        # Straight radial teeth are simplified visual geometry, not a valid Gleason profile.
        for i in range(spec['teeth']):
            angle=2*math.pi*i/spec['teeth'];width=math.pi/spec['teeth']*.45;vertices=[]
            for z,r0,r1 in ((hub_top,.0098,.011205),(h,.0052,.0061)):
                for r,a in ((r0,angle-width),(r1,angle-width),(r1,angle+width),(r0,angle+width)):
                    vertices.extend([r*math.cos(a),r*math.sin(a),z])
            key=f'tooth_{i}'
            ET.SubElement(asset,'mesh',name=key,vertex=' '.join(map(str,vertices)))
            ET.SubElement(body,'geom',name=key,type='mesh',mesh=key,material='part_metal',mass='0',contype='0',conaffinity='0')
    ET.SubElement(body,'site',name='positive_face',pos=f'0 0 {h}',size='.0001',rgba='0 0 0 0')
    ET.SubElement(body,'site',name='negative_face',pos=f'0 0 {-h}',size='.0001',rgba='0 0 0 0')
    return root
