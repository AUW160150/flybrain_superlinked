"""Outdoor arena: landmarks as primitives, pond (no-fly zone), hidden gust zone, home pad.

Coordinates: x east, y north, z up, metres. Arena is 40 x 40 x 12 m centred on the origin.
"""
from dataclasses import dataclass, field

import numpy as np

HALF = 20.0
CEILING = 12.0


@dataclass
class Primitive:
    kind: str                  # "cyl" (vertical cylinder) or "box" (axis-aligned)
    center: tuple              # (x, y, z_bottom) for cyl; (x, y, z_center) for box
    size: tuple                # (radius, height) for cyl; (hx, hy, hz) half extents for box
    color: tuple               # RGB 0..1


@dataclass
class Landmark:
    name: str                  # canonical name used by planner and memory
    label: str                 # plain-words label for the UI
    color_name: str
    goal: tuple                # (x, y) point a drone flies to when this landmark is its target
    prims: list = field(default_factory=list)


def _trees():
    out = []
    for i, (x, y) in enumerate([(-15, 2), (-6, 16), (15, -4), (-2, 6), (12, 2)]):
        out.append(Landmark(f"green_tree_{i+1}", f"green tree {i+1}", "green", (x, y),
                            [Primitive("cyl", (x, y, 0.0), (0.9, 7.0), (0.18, 0.62, 0.22))]))
    return out


def build_landmarks():
    lms = [
        Landmark("red_tower", "red tower", "red", (8.0, 13.0),
                 [Primitive("cyl", (10.0, 15.0, 0.0), (1.4, 11.0), (0.86, 0.12, 0.10))]),
        Landmark("blue_arch", "blue arch", "blue", (-12.0, 11.0),
                 [Primitive("box", (-14.0, 12.0, 3.5), (0.5, 0.5, 3.5), (0.12, 0.28, 0.85)),
                  Primitive("box", (-10.0, 12.0, 3.5), (0.5, 0.5, 3.5), (0.12, 0.28, 0.85)),
                  Primitive("box", (-12.0, 12.0, 7.4), (2.5, 0.5, 0.4), (0.12, 0.28, 0.85))]),
        Landmark("yellow_cubes", "yellow cube cluster", "yellow", (-11.0, -9.0),
                 [Primitive("box", (-13.0, -12.0, 1.0), (1.0, 1.0, 1.0), (0.95, 0.82, 0.12)),
                  Primitive("box", (-10.5, -13.0, 1.5), (1.5, 1.5, 1.5), (0.95, 0.82, 0.12)),
                  Primitive("box", (-12.0, -14.5, 0.75), (0.75, 0.75, 0.75), (0.95, 0.82, 0.12))]),
    ]
    return lms + _trees()


LANDMARKS = build_landmarks()
LANDMARK_BY_NAME = {lm.name: lm for lm in LANDMARKS}
HOME = (0.0, -16.0)
HOME_RADIUS = 1.5
POND = {"center": (2.0, -7.0), "radius": 3.0}               # no-fly zone (marked on every map)
GUST = {"min": (2.0, 0.0), "max": (11.0, 9.0), "wind": (-2.5, 0.0, -0.5), "turb": 2.0}   # hidden at start
PRIMS = [p for lm in LANDMARKS for p in lm.prims]


def in_pond(x, y):
    return (x - POND["center"][0]) ** 2 + (y - POND["center"][1]) ** 2 <= POND["radius"] ** 2


def in_gust(x, y):
    return GUST["min"][0] <= x <= GUST["max"][0] and GUST["min"][1] <= y <= GUST["max"][1]


def collide(pos, r=0.35):
    """Return the landmark name the drone sphere (radius r) intersects, 'ground', or None."""
    x, y, z = pos
    if z <= r * 0.5:
        return "ground"
    for lm in LANDMARKS:
        for p in lm.prims:
            if p.kind == "cyl":
                cx, cy, z0 = p.center; rad, h = p.size
                if z0 - r <= z <= z0 + h + r and (x - cx) ** 2 + (y - cy) ** 2 <= (rad + r) ** 2:
                    return lm.name
            else:
                c, s = p.center, p.size
                if all(abs(v - cv) <= sv + r for v, cv, sv in zip((x, y, z), c, s)):
                    return lm.name
    return None


def describe():
    """Static arena description for the frontend (gust zone included; the UI reveals it on discovery)."""
    return {
        "half": HALF, "ceiling": CEILING, "home": HOME, "home_radius": HOME_RADIUS, "pond": POND, "gust": GUST,
        "landmarks": [{"name": lm.name, "label": lm.label, "color": lm.color_name, "goal": lm.goal,
                       "prims": [{"kind": p.kind, "center": p.center, "size": p.size, "color": p.color} for p in lm.prims]}
                      for lm in LANDMARKS],
    }


def prim_arrays():
    """Vectorised primitive tables for the eye's ray caster."""
    cyl = [p for p in PRIMS if p.kind == "cyl"]
    box = [p for p in PRIMS if p.kind == "box"]
    owner = {id(p): i for i, lm in enumerate(LANDMARKS) for p in lm.prims}
    return {
        "cyl_c": np.array([p.center for p in cyl], float).reshape(-1, 3),
        "cyl_s": np.array([p.size for p in cyl], float).reshape(-1, 2),
        "cyl_col": np.array([p.color for p in cyl], float).reshape(-1, 3),
        "cyl_lm": np.array([owner[id(p)] for p in cyl], int),
        "box_c": np.array([p.center for p in box], float).reshape(-1, 3),
        "box_s": np.array([p.size for p in box], float).reshape(-1, 3),
        "box_col": np.array([p.color for p in box], float).reshape(-1, 3),
        "box_lm": np.array([owner[id(p)] for p in box], int),
    }
