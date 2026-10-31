"""Append a fixed pre-insertion stereo pair without altering H1 geometry or dynamics."""
import xml.etree.ElementTree as ET
import numpy as np
from yolo_conveyor.held.scene import HeldScene, SETUP as H1_SETUP
from yolo_conveyor.simulation.panda_scene import _camera

RECHECK_CENTER = np.array([-.10, .10, .157])
RECHECK_VIEWS = ('recheck_a', 'recheck_b')
SETUP = {**H1_SETUP, 'recheck_camera_center_m': RECHECK_CENTER.tolist(),
         'recheck_camera_offsets_m': [[-.15, -.259807621, 0], [-.259807621, .15, 0]],
         'recheck_fovy_deg': 30, 'total_fixed_rgb_cameras': 4}


class RecheckScene(HeldScene):
    def __init__(self, assets, size=640):
        super().__init__(assets, size)
        root = ET.fromstring(self.xml)
        root.set('model', 'held_recheck_h2')
        world = root.find('worldbody')
        for name, offset in zip(RECHECK_VIEWS, SETUP['recheck_camera_offsets_m']):
            _camera(world, name, RECHECK_CENTER + offset, RECHECK_CENTER, 30)
        xml = ET.tostring(root, encoding='unicode')
        model = self.mj.MjModel.from_xml_string(xml)
        # Adding cameras must preserve every inherited body/actuator/geometry binding.
        for kind, count in (('body', self.model.nbody), ('geom', self.model.ngeom),
                            ('joint', self.model.njnt), ('actuator', self.model.nu),
                            ('site', self.model.nsite), ('sensor', self.model.nsensor),
                            ('camera', self.model.ncam)):
            for index in range(count):
                if getattr(model, kind)(index).name != getattr(self.model, kind)(index).name:
                    self.close()
                    raise ValueError('H1 binding changed while adding cameras')
        self.renderer.close()
        self.xml, self.model = xml, model
        self.data = self.mj.MjData(model)
        self.renderer = self.mj.Renderer(model, height=size, width=size)
