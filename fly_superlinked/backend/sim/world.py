"""Swarm simulation: kinematic quadrotors, compound eyes, fly-brain or reactive steering.

Control loop at 25 Hz (one eye frame, 40 ms of connectome time); physics at 50 Hz.
Altitude is a plain PID, not the fly brain.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import arena
from .eye import CompoundEye, EyeSignals
from ..brain.controller import FlyBrains, Reactive

CTRL_DT = 0.04
PHYS_SUB = 2
CRUISE_Z = 4.0
REACH_R = 2.0
GOAL_R = 3.2          # visit/survey goals sit at a landmark's centre; within 3.2 m counts as visited
DELIVER_R = 2.5
_EYE = None


def shared_eye():
    global _EYE
    if _EYE is None:
        _EYE = CompoundEye()
    return _EYE


@dataclass
class Waypoint:
    x: float
    y: float
    z: float = CRUISE_Z
    label: str = ""
    kind: str = "route"          # route | goal | deliver | home


@dataclass
class Drone:
    name: str
    role: str
    pos: np.ndarray
    yaw: float
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    waypoints: list = field(default_factory=list)
    wp_i: int = 0
    active: bool = False
    done: bool = False
    delivered: bool = False
    collisions: int = 0
    gust_entries: int = 0
    pond_entries: int = 0
    in_contact: str = None
    in_gust: bool = False
    in_pond: bool = False
    disturb: float = 0.0
    gust_detect_t: float = 0.0
    gust_reports: list = field(default_factory=list)
    trail: list = field(default_factory=list)
    t_start: float = 0.0
    t_done: float = None
    last_cmd: dict = field(default_factory=dict)
    last_view: dict = None
    seen: dict = field(default_factory=dict)       # landmark -> last time seen by keyframe trigger
    last_keyframe: float = -10.0

    @property
    def target(self):
        return self.waypoints[self.wp_i] if self.wp_i < len(self.waypoints) else None


class World:
    def __init__(self, seed=0, brain="real", drones=("scout-1", "scout-2", "courier"), wind_scale=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.brain_type = brain
        self.eye = shared_eye()
        self.t = 0.0
        self.events = []                       # drained by the caller each step
        self.drones = {}
        starts = {"scout-1": (-1.5, -16.5), "scout-2": (1.5, -16.5), "courier": (0.0, -15.0)}
        for i, n in enumerate(drones):
            x, y = starts.get(n, (0.0, -16.0))
            jitter = self.rng.normal(0, 0.3, 2)
            self.drones[n] = Drone(n, "courier" if n == "courier" else "scout",
                                   np.array([x + jitter[0], y + jitter[1], 0.4]), math.pi / 2 + self.rng.normal(0, 0.15))
        self.order = list(self.drones)
        self.signals = {n: EyeSignals(self.eye) for n in self.order}
        ws = wind_scale if wind_scale is not None else float(self.rng.uniform(0.85, 1.25))
        self.wind = np.array(arena.GUST["wind"]) * ws
        self.turb = arena.GUST["turb"] * ws
        self.fly = FlyBrains(brain, len(self.order)) if brain in ("real", "rewired", "rewired_class") else None
        self.reactive = Reactive()
        self.last_brain = {}

    # ----- missions -----
    def assign(self, name, waypoints):
        d = self.drones[name]
        d.waypoints, d.wp_i, d.active, d.done, d.delivered, d.t_start, d.t_done = list(waypoints), 0, True, False, False, self.t, None

    # ----- control -----
    def step(self):
        views, amps, stripes = {}, [], {}
        for n in self.order:
            d = self.drones[n]
            view = self.eye.cast(d.pos, d.yaw)
            sig = self.signals[n].update(view, CTRL_DT)
            views[n] = (view, sig)
            d.last_view = view
            tgt = d.target
            sl = sr = 0.0
            if tgt is not None and d.active:
                bearing = math.atan2(tgt.y - d.pos[1], tgt.x - d.pos[0])
                m = self.eye.stripe_mask(d.yaw, bearing)
                nl = float(m[self.eye.left].sum()); nr = float(m[~self.eye.left].sum())
                sl, sr = min(1.0, nl / 25.0), min(1.0, nr / 25.0)
                stripes[n] = m
            if self.fly is not None:
                amps.append(self.fly.encode(sl, sr, sig, d.active and not d.done))
        brain_out = self.fly.step(amps, int(CTRL_DT * 1000), CTRL_DT) if self.fly is not None else None
        for bi, n in enumerate(self.order):
            d = self.drones[n]
            view, sig = views[n]
            tgt = d.target
            if not d.active or d.done or tgt is None:
                cmd = {"yaw_rate": 0.0, "v": 0.0, "escape": False, "escape_started": False, "escape_dir": 0.0}
            elif brain_out is not None:
                cmd = brain_out[bi]
            else:
                bearing = math.atan2(tgt.y - d.pos[1], tgt.x - d.pos[0])
                err = (bearing - d.yaw + math.pi) % (2 * math.pi) - math.pi
                cmd = self.reactive.command(err, view, self.eye)
            if cmd.get("escape_started"):
                self.events.append({"type": "escape", "drone": n, "t": self.t})
            d.last_cmd = cmd
            self.last_brain[n] = (cmd, sig, stripes.get(n))
            zt = tgt.z if (tgt is not None and d.active and not d.done) else (CRUISE_Z if d.active else 0.0)
            if d.done:
                zt = 0.0
            for _ in range(PHYS_SUB):
                self._physics(d, cmd, zt, CTRL_DT / PHYS_SUB)
            self._bookkeep(d)
        self.t += CTRL_DT

    def _physics(self, d, cmd, zt, dt):
        vz = float(np.clip(1.5 * (zt - d.pos[2]) - 0.6 * d.vel[2], -2.0, 2.0))       # altitude PID (not the fly)
        side = 0.0
        if cmd.get("escape"):
            vz = 2.5; side = 3.0 * cmd.get("escape_dir", 0.0)
        d.yaw = (d.yaw + cmd["yaw_rate"] * dt + math.pi) % (2 * math.pi) - math.pi
        c, s = math.cos(d.yaw), math.sin(d.yaw)
        v_air = np.array([cmd["v"] * c - side * s, cmd["v"] * s + side * c, vz])
        wind = np.zeros(3)
        if arena.in_gust(d.pos[0], d.pos[1]) and d.pos[2] > 0.5:
            wind = self.wind + self.rng.normal(0, self.turb, 3) * np.array([1, 1, 0.3])
        d.vel += (v_air + wind - d.vel) * min(1.0, dt / 0.35)
        d.pos = d.pos + d.vel * dt
        d.pos[0] = np.clip(d.pos[0], -arena.HALF, arena.HALF); d.pos[1] = np.clip(d.pos[1], -arena.HALF, arena.HALF)
        d.pos[2] = np.clip(d.pos[2], 0.15, arena.CEILING)
        # disturbance estimate: how far the measured velocity departs from the commanded air velocity
        d.disturb += (float(np.linalg.norm((d.vel - v_air)[:2])) - d.disturb) * min(1.0, dt / 0.3)

    def _bookkeep(self, d):
        hit = arena.collide(d.pos)
        if hit and hit != d.in_contact and not (hit == "ground" and (not d.active or d.done or d.wp_i == 0 and d.pos[2] < 0.6)):
            d.collisions += 1
            self.events.append({"type": "collision", "drone": d.name, "with": hit, "t": self.t})
        d.in_contact = hit
        if hit and hit != "ground":            # push out of the obstacle
            d.vel *= -0.3
        g = arena.in_gust(d.pos[0], d.pos[1])
        if g and not d.in_gust:
            d.gust_entries += 1
        d.in_gust = g
        p = arena.in_pond(d.pos[0], d.pos[1])
        if p and not d.in_pond and d.active:
            d.pond_entries += 1
            self.events.append({"type": "pond", "drone": d.name, "t": self.t})
        d.in_pond = p
        # gust sensing: sustained disturbance (the drone feels it; it never reads the arena's gust box)
        if d.active and d.disturb > 1.6:
            d.gust_detect_t += CTRL_DT
            if d.gust_detect_t >= 0.5 and (not d.gust_reports or np.hypot(*(d.pos[:2] - d.gust_reports[-1][:2])) > 1.5):
                d.gust_reports.append(d.pos.copy())
                self.events.append({"type": "gust_felt", "drone": d.name, "pos": d.pos.round(2).tolist(), "t": self.t,
                                    "disturbance": round(d.disturb, 2)})
        else:
            d.gust_detect_t = 0.0
        if len(d.trail) == 0 or np.hypot(*(d.pos[:2] - d.trail[-1][:2])) > 0.4:
            d.trail.append(d.pos.copy())
        tgt = d.target
        if d.active and not d.done and tgt is not None:
            dist = math.hypot(tgt.x - d.pos[0], tgt.y - d.pos[1])
            radius = {"deliver": DELIVER_R, "home": arena.HOME_RADIUS, "goal": GOAL_R}.get(tgt.kind, REACH_R)
            if dist < radius:
                if tgt.kind == "deliver":
                    d.delivered = True
                    self.events.append({"type": "delivered", "drone": d.name, "t": self.t, "landmark": tgt.label})
                self.events.append({"type": "waypoint", "drone": d.name, "label": tgt.label, "kind": tgt.kind, "t": self.t})
                d.wp_i += 1
                if d.wp_i >= len(d.waypoints):
                    d.done = True; d.t_done = self.t
                    self.events.append({"type": "mission_done", "drone": d.name, "t": self.t})

    # ----- observation for keyframes and UI -----
    def visible_landmarks(self, name, min_px=6):
        view = self.drones[name].last_view
        if view is None:
            return {}
        ids, cnt = np.unique(view["lm"][view["lm"] >= 0], return_counts=True)
        return {arena.LANDMARKS[i].name: int(c) for i, c in zip(ids, cnt) if c >= min_px}

    def snapshot(self, detail_drone=None):
        out = {"t": round(self.t, 2), "brain": self.brain_type, "drones": []}
        for n in self.order:
            d = self.drones[n]
            tgt = d.target
            out["drones"].append({
                "name": n, "role": d.role, "pos": d.pos.round(2).tolist(), "yaw": round(d.yaw, 3),
                "speed": round(float(np.linalg.norm(d.vel)), 2), "active": d.active, "done": d.done, "delivered": d.delivered,
                "target": None if tgt is None else {"x": tgt.x, "y": tgt.y, "label": tgt.label, "kind": tgt.kind},
                "waypoints": [[w.x, w.y, w.kind, w.label] for w in d.waypoints], "wp_i": d.wp_i,
                "collisions": d.collisions, "gust_entries": d.gust_entries, "pond_entries": d.pond_entries,
                "in_gust": d.in_gust, "trail": [p[:3].round(2).tolist() for p in d.trail[-200:]],
            })
        return out
