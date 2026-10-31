"""Fixed-nest pickup and inspection presentation; later motion uses measured predictions."""

import numpy as np
from yolo_conveyor.simulation.panda_control import PandaController,Stage
from yolo_conveyor.simulation.grasp_geometry import downward_rotation
from .scene import INSPECT_TCP,INSPECT_R,GUIDE_XY,GUIDE_TOP,INSERT_DEPTH
from .geometry import compensated_target


class HeldController(PandaController):
    def prepare_stages(self,stages):
        initial=self.scene.data.qpos[self.scene.arm_qpos].copy();previous=initial
        for stage in stages:
            stage.duration=max(stage.duration,float(1.875*np.max(np.abs(stage.q-previous))/1.2));previous=stage.q
        self._preflight(stages,initial)
        return stages

    def plan_pick_and_inspect(self):
        q=self.scene.data.qpos[self.scene.arm_qpos].copy();r=downward_rotation(0)
        cached=getattr(self,"_pick_cache",None)
        if cached is not None and np.allclose(q,cached[0],atol=1e-6):
            return [Stage(s.name,s.q.copy(),s.gripper,s.duration) for s in cached[1]]
        pre=self.inverse_kinematics([0,0,.12],r,q)
        grip=self.inverse_kinematics([0,0,.0045],r,pre)
        lift=self.inverse_kinematics([0,0,.28],r,grip)
        try:inspect=self.inverse_kinematics(INSPECT_TCP,INSPECT_R,lift)
        except ValueError:inspect=self.inverse_kinematics(INSPECT_TCP,INSPECT_R,q)
        stages=self.prepare_stages([Stage("approach",pre,.04,1.5),Stage("descend",grip,.04,1.2),
               Stage("align_settle",grip,.04,.3),Stage("close",grip,0,.8),Stage("lift",lift,0,1.2),
               Stage("present",inspect,0,2.),Stage("hold",inspect,0,1.)])
        self._pick_cache=(q.copy(),stages)
        return [Stage(s.name,s.q.copy(),s.gripper,s.duration) for s in stages]

    def plan_prealign(self,estimate,inspection_tool):
        position,rotation=compensated_target(estimate,inspection_tool,[*GUIDE_XY,GUIDE_TOP+.015])
        current=self.scene.data.qpos[self.scene.arm_qpos].copy()
        # Transfer above the fixture before descending to the pre-insertion gap.
        high=position.copy();high[2]=max(.25,position[2]+.08)
        transfer=self.inverse_kinematics(high,rotation,current)
        aligned=self.inverse_kinematics(position,rotation,transfer)
        return self.prepare_stages([Stage("transfer",transfer,0,2.),Stage("prealign",aligned,0,2.),Stage("hold",aligned,0,.8)])

    def plan_insert(self,estimate,inspection_tool):
        q=self.scene.data.qpos[self.scene.arm_qpos].copy();stages=[]
        for height in np.linspace(GUIDE_TOP+.015,GUIDE_TOP-INSERT_DEPTH,24)[1:]:
            position,rotation=compensated_target(estimate,inspection_tool,[*GUIDE_XY,height])
            q=self.inverse_kinematics(position,rotation,q)
            stages.append(Stage("insert",q,0,.4))
        stages.append(Stage("hold",q,0,.8))
        return self.prepare_stages(stages)
