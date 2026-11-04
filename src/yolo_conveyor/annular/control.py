"""Known-nest grasp and recipe motion; RGB grants permission, object truth is never read."""
import numpy as np
from yolo_conveyor.simulation.panda_control import PandaController,Stage
from yolo_conveyor.simulation.grasp_geometry import downward_rotation
from .scene import INSPECT_TCP,SHAFT_XY,SHAFT_TOP,ENGAGEMENT


class AssemblyController(PandaController):
    def stages(self,items):
        previous=self.scene.data.qpos[self.scene.arm_qpos].copy();initial=previous.copy()
        for stage in items:
            stage.duration=max(stage.duration,float(1.875*np.max(np.abs(stage.q-previous))/1.2));previous=stage.q
        self._preflight(items,initial);return items

    def pick(self,tilt_deg=0):
        q=self.scene.data.qpos[self.scene.arm_qpos].copy();r=downward_rotation(0)
        z=self.scene.spec['height']/2
        approach=self.inverse_kinematics([0,0,.10],r,q);grip=self.inverse_kinematics([0,0,z],r,approach)
        lift=self.inverse_kinematics(INSPECT_TCP,r,grip)
        items=[Stage('approach',approach,.04,1.2),Stage('descend',grip,.04,1.),Stage('align_settle',grip,.04,.2),
               Stage('close',grip,0,.6),Stage('lift',lift,0,1.4)]
        if tilt_deg:
            a=np.radians(tilt_deg);rot=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
            lift=self.inverse_kinematics(INSPECT_TCP,rot@r,lift);items.append(Stage('present_tilt',lift,0,1.))
        items.append(Stage('hold',lift,0,.7));return self.stages(items)

    def preposition(self):
        q=self.scene.data.qpos[self.scene.arm_qpos].copy();r=downward_rotation(0);h=self.scene.spec['height']/2
        high=self.inverse_kinematics([*SHAFT_XY,.22],r,q)
        above=self.inverse_kinematics([*SHAFT_XY,SHAFT_TOP+h+.012],r,high)
        items=[Stage('transfer',high,0,1.5),Stage('above_shaft',above,0,1.2),Stage('hold',above,0,.4)]
        for z in np.linspace(SHAFT_TOP+h+.012,SHAFT_TOP+h-ENGAGEMENT,16)[1:]:
            above=self.inverse_kinematics([*SHAFT_XY,z],r,above);items.append(Stage('preposition',above,0,.25))
        items.append(Stage('hold',above,0,.6));return self.stages(items)
