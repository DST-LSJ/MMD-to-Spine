"""Runtime paths and configurable model-independent humanoid."""
import math
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / '文档'
ASSETS = ROOT / '资源'
RIG_PATH = ASSETS / '通用骨架.json'
LIMBS = ('armF', 'handF', 'armB', 'handB', 'thighF', 'legF', 'thighB', 'legB')


def subtract(a, b):
    return tuple(x-y for x,y in zip(a,b))


def add(a, b):
    return tuple(x+y for x,y in zip(a,b))


def load_profile():
    data = json.loads(RIG_PATH.read_text(encoding='utf-8'))
    for key, value in data['geometry'].items():
        if len(value) != 3 or not all(isinstance(v, (int,float)) and math.isfinite(v) for v in value):
            raise ValueError(f'Invalid generic geometry: {key}')
        if key.startswith(('thigh_', 'shin_', 'upper_arm_', 'forearm_')) and sum(v*v for v in value) < 1e-10:
            raise ValueError(f'Zero-length generic limb: {key}')
    return data
