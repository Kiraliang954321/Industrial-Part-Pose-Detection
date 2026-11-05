"""Declared external-force fault harness, not part of visual decision/control inputs."""
from yolo_conveyor.simulation.panda_control import Stage

FAULTS={'axial_slip': {'force_world_n':[0,0,-3.], 'seconds':.03},
        'strong_disturbance': {'force_world_n':[0,0,-4.], 'seconds':.05}}


def disturb(scene,controller,case,observer):
    if case not in FAULTS:return None
    spec=FAULTS[case];q=scene.data.qpos[scene.arm_qpos].copy()
    try:
        scene.data.xfrc_applied[scene.body_id,:3]=spec['force_world_n']
        controller.execute([Stage('hold',q,0,spec['seconds'])],lambda _:observer('external_disturbance'))
    finally:
        scene.data.xfrc_applied[scene.body_id,:]=0
    controller.execute([Stage('hold',q,0,.6)],lambda _:observer('disturbance_settle'))
    return spec
