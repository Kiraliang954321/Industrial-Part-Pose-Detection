"""Indexed-cell robot adapter; only the pallet encoder augments fixed-path checks."""

import numpy as np

from yolo_conveyor.simulation.panda_control import PandaController


class IndexedPandaController(PandaController):
    def _preflight(self, stages, initial):
        s, d = self.scene, self.ik_data
        previous = initial
        encoder = float(s.data.qpos[s.pallet_q])
        for stage in stages:
            for fraction in np.linspace(0, 1, 30):
                d.qpos[:] = self.m.qpos0
                d.qpos[s.arm_qpos] = previous * (1-fraction) + stage.q * fraction
                d.qpos[s.finger_qpos] = max(0.002, stage.gripper)
                d.qpos[s.pallet_q] = encoder  # Known actuator state, not object ground truth.
                self.mj.mj_forward(self.m, d)
                errors = s.illegal_contacts(d)
                if errors:
                    raise ValueError(f"unsafe indexed-cell path at {stage.name}: {errors[0]}")
            previous = stage.q
