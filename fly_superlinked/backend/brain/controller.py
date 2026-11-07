"""Eye -> connectome -> descending neurons -> drone command, plus the reactive baseline.

What the fly circuit controls: yaw rate (DNa02/DNa01 left-right difference), forward speed (DNp09
rate) and the escape reflex (DNp01, the giant fiber). What it does not control: altitude (a plain
PID), the 'go' current that tells DNp09 to walk while a mission is active (an engineered input,
as in FlyWorld's DNp09 button), and which target is shown (the planner chooses it).

Input mapping (retinotopic by eye; measured in bench/results/probe_circuit.json):
  target stripe in left/right eye  -> LC10 (all LC10 subtypes present in the subset) left/right
  near-object expansion per eye     -> LC4, LC6 (expansion rate); LPLC2, LC16 (size while expanding)
  mean temporal contrast per eye    -> HS left/right (optic-flow cells; small effect, measured)
Output mapping reuses FlyWorld's MotorDecoder rules and its measured calibration
(turn_bias_per_dnp09_hz removes the walk-coupled DNa02 asymmetry).
"""
import json
import math
import re

import numpy as np

from .circuit import DATA, circuit_edges, load_circuit
from .lif import BatchedLIF

GO_AMP = 0.15            # FlyWorld panel amplitude for DNp09 stimulation
LC10_MAX = 0.30
LOOM_VEL_MAX = 0.20
LOOM_SIZE_MAX = 0.20
HS_MAX = 0.10
V_MAX = 4.0              # m/s at DNp09 saturation
YAW_MAX = math.radians(120)
TURN_SCALE = 20.0        # Hz scale inside tanh (FlyWorld used 50; chosen on tuning seeds 100-102, then frozen)
TURN_DEAD = 4.0          # Hz dead zone (FlyWorld used 10; chosen on tuning seeds 100-102, then frozen)
WALK_THR, WALK_SAT = 10.0, 100.0

# ~64 neurons shown in the UI raster, by type (indices are chosen from these groups, never invented)
RASTER_GROUPS = [("LC10_L", 3), ("LC10_R", 8), ("LC4_L", 6), ("LC4_R", 6), ("LPLC2_L", 4), ("LPLC2_R", 4),
                 ("HS_L", 3), ("HS_R", 3), ("AOTU_L", 5), ("AOTU_R", 5), ("DNa02_L", 1), ("DNa02_R", 1),
                 ("DNa01_L", 1), ("DNa01_R", 1), ("DNp09_L", 1), ("DNp09_R", 1), ("DNp01_L", 1), ("DNp01_R", 1),
                 ("MDN", 4), ("DNg02", 5)]
EXTRA_TYPE_GROUPS = [("LC10", r"^LC10"), ("AOTU", r"^AOTU")]


def build_groups():
    c = load_circuit()
    g = dict(c["groups"])
    for name, rx in EXTRA_TYPE_GROUPS:
        idx = np.flatnonzero([re.match(rx, t) is not None for t in c["type"]])
        g[name] = idx
        for s in "LR":
            g[f"{name}_{s}"] = idx[c["side"][idx] == s]
    return g


def raster_neurons(groups):
    c = load_circuit()
    out = []
    for name, k in RASTER_GROUPS:
        for i in groups.get(name, [])[:k]:
            out.append((int(i), str(c["type"][i]), name.split("_")[-1] if name[-2:] in ("_L", "_R") else str(c["side"][i])))
    return out


def load_calibration():
    with open(DATA / "calibration.json") as f:
        return json.load(f)


class FlyBrains:
    """One batched connectome simulation for every fly-brained drone in a sim (real or rewired)."""

    def __init__(self, brain_type, n_drones, rewire_seed=1001):
        c = load_circuit()
        self.brain_type = brain_type
        self.groups = build_groups()
        pre, post, w = circuit_edges(brain_type, rewire_seed)
        self.raster = raster_neurons(self.groups)
        self.lif = BatchedLIF(c["N"], pre, post, w, self.groups, B=n_drones, watch=[r[0] for r in self.raster])
        self.turn_bias = self.calibrate_turn_bias(c, pre, post, w)
        self.ema = [dict() for _ in range(n_drones)]
        self.escape_timer = np.zeros(n_drones)
        self.escape_cool = np.zeros(n_drones)
        self.escape_dir = np.zeros(n_drones)

    def calibrate_turn_bias(self, c, pre, post, w, ms=300):
        """FlyWorld's calibration procedure, re-measured for this wiring: the DNa left-right asymmetry
        produced by the 'go' current alone, per Hz of DNp09, is subtracted by the decoder."""
        lif = BatchedLIF(c["N"], pre, post, w, self.groups, B=1)
        lif.set_amps([{"DNp09": GO_AMP}]); lif.step(100)
        counts, _ = lif.step(ms)
        r = {n: float(lif.rates_hz(counts, ms)[0, i]) for n, i in lif.index.items()}
        left = r["DNa02_L"] + 0.5 * r["DNa01_L"]; right = r["DNa02_R"] + 0.5 * r["DNa01_R"]
        self.calibration = {"go_dnp09_hz": r["DNp09"], "go_left_hz": left, "go_right_hz": right,
                            "flyworld_published_bias": float(load_calibration().get("turn_bias_per_dnp09_hz", 0.0))}
        return (left - right) / r["DNp09"] if r["DNp09"] > 0 else 0.0

    def encode(self, stripe_l, stripe_r, sig, active):
        """Currents on named input groups for one drone."""
        if not active:
            return {}
        a = {"DNp09": GO_AMP,
             "LC10_L": LC10_MAX * stripe_l, "LC10_R": LC10_MAX * stripe_r,
             "HS_L": HS_MAX * min(1.0, sig["motion_l"] / 2.0), "HS_R": HS_MAX * min(1.0, sig["motion_r"] / 2.0)}
        for side, exp_, cov in (("L", sig["expand_l"], sig["cover_l"]), ("R", sig["expand_r"], sig["cover_r"])):
            vel = LOOM_VEL_MAX * min(max(exp_ / 0.5, 0.0), 1.0)
            size = LOOM_SIZE_MAX * min(max((cov - 0.02) / 0.15, 0.0), 1.0) * (1.0 if exp_ > 0 else 0.0)
            a["LC4_" + side] = vel; a["LC6_" + side] = vel
            a["LPLC2_" + side] = size; a["LC16_" + side] = size
        return {k: v for k, v in a.items() if v > 0}

    def step(self, amps, ms, dt):
        """Run ms of biology. Returns per-drone readouts and the watched-neuron raster."""
        self.lif.set_amps(amps)
        counts, raster = self.lif.step(ms)
        rates = self.lif.rates_hz(counts, ms)
        k = 1.0 - math.exp(-ms / 100.0)              # FlyWorld EMA, 100 ms
        out = []
        for b in range(self.lif.B):
            r = {n: float(rates[b, i]) for n, i in self.lif.index.items()}
            ema = self.ema[b]
            for n in ("DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "DNp01_L", "DNp01_R", "LC10_L", "LC10_R", "AOTU_L", "AOTU_R"):
                ema[n] = ema.get(n, 0.0) + k * (r.get(n, 0.0) - ema.get(n, 0.0))
            left = ema["DNa02_L"] + 0.5 * ema["DNa01_L"]
            right = ema["DNa02_R"] + 0.5 * ema["DNa01_R"]
            delta = (right - left) + self.turn_bias * ema["DNp09"]
            if abs(delta) < TURN_DEAD:
                delta = 0.0
            yaw_rate = -YAW_MAX * math.tanh(delta / TURN_SCALE)      # + delta = turn right = negative yaw rate
            v = V_MAX * min(max((ema["DNp09"] - WALK_THR) / (WALK_SAT - WALK_THR), 0.0), 1.0)
            gf = counts[b, self.lif.index["DNp01"]] if "DNp01" in self.lif.index else 0
            self.escape_cool[b] = max(0.0, self.escape_cool[b] - dt)
            self.escape_timer[b] = max(0.0, self.escape_timer[b] - dt)
            escape_started = False
            if gf > 0 and self.escape_cool[b] <= 0 and self.escape_timer[b] <= 0:
                self.escape_timer[b] = 0.4; self.escape_cool[b] = 1.5
                self.escape_dir[b] = -1.0 if r.get("DNp01_L", 0) > r.get("DNp01_R", 0) else 1.0   # threat on left -> sidestep right
                escape_started = True
            out.append({"yaw_rate": yaw_rate, "v": v, "escape": self.escape_timer[b] > 0, "escape_started": escape_started,
                        "escape_dir": float(self.escape_dir[b]), "dn_left": left, "dn_right": right, "delta": delta,
                        "dnp09": ema["DNp09"], "giant_fiber_spikes": int(gf),
                        "rates": {n: round(r.get(n, 0.0), 1) for n in ("LC10_L", "LC10_R", "AOTU_L", "AOTU_R", "DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "DNp01_L", "DNp01_R", "LC4_L", "LC4_R")},
                        "raster": raster[b]})
        return out


class Reactive:
    """Hand-written baseline: steer toward the target bearing, repel from near surfaces seen by the eye."""

    def command(self, bearing_err, view, eye):
        near = (view["lm"] >= 0) & (view["depth"] < 6.0) & (np.abs(eye.el) < np.radians(35))
        push = 0.0
        if near.any():
            w = (1.0 - view["depth"][near] / 6.0) * np.cos(eye.az[near])
            w = np.clip(w, 0, None)
            push = float(-np.sum(w * np.sign(eye.az[near])) / 25.0)    # obstacle on left (az>0) -> turn right
        yaw_rate = float(np.clip(2.0 * bearing_err + 2.5 * push, -YAW_MAX, YAW_MAX))
        front = near & (np.abs(eye.az) < np.radians(25))
        slow = 0.4 if front.sum() > 8 else 1.0
        v = V_MAX * max(0.25, math.cos(min(abs(bearing_err), math.pi / 2))) * slow
        return {"yaw_rate": yaw_rate, "v": v, "escape": False, "escape_started": False, "escape_dir": 0.0}
