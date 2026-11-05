"""Explicit, validated user-configurable target orientations."""
import json
from pathlib import Path
import numpy as np
from .geometry import rotation


def load(path):
    config=json.loads(Path(path).read_text(encoding='utf-8'))
    if config['coordinate_frame']!='simulation_world_axes_parallel_to_robot_base':raise ValueError('unsupported target frame')
    a,b,g=(config[k] for k in ('axis_tolerance_deg','spin_tolerance_deg','guard_deg'))
    if not np.isfinite([a,b,g]).all() or not 0<g<min(a,b)<=30:raise ValueError('invalid target tolerance/guard')
    targets=config['targets']
    if not 1<=len(targets)<=8 or len({t['id'] for t in targets})!=len(targets):raise ValueError('invalid target list')
    for t in targets:
        values=[t[k] for k in ('inclination_deg','azimuth_deg','spin_deg')]
        if not np.isfinite(values).all() or not 0<=values[0]<=30:raise ValueError('target outside first development envelope')
    return config


def matrix(target):return rotation(target['inclination_deg'],target['azimuth_deg'],target['spin_deg'])
