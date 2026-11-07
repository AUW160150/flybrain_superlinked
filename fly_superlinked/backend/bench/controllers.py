"""Controller experiment: real connectome vs degree-preserving rewired vs reactive baseline.

Same courier mission (home -> red tower -> home, pond avoided by the route, gust zone unknown),
same 20 evaluation seeds (0-19) for every arm; seeds 100-102 were used for decoder tuning only.
Writes backend/bench/results/controllers.json. Every number in the UI is read from that file.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from ..agents.route import plan_route
from ..sim import arena
from ..sim.world import World

SEEDS = list(range(20))
T_MAX = 120.0
OUT = Path(__file__).resolve().parent / "results/controllers.json"


def run_one(brain, seed):
    w = World(seed=seed, brain=brain, drones=("courier",))
    d = w.drones["courier"]
    lm = arena.LANDMARK_BY_NAME["red_tower"]
    w.assign("courier", plan_route(d.pos[:2], [(lm.goal[0], lm.goal[1], "red_tower", "deliver"), (*arena.HOME, "home", "home")]))
    t_deliver, escapes, path = None, 0, 0.0
    last = d.pos.copy()
    while w.t < T_MAX and not d.done:
        w.step()
        for ev in w.events:
            if ev["type"] == "delivered":
                t_deliver = ev["t"]
            if ev["type"] == "escape":
                escapes += 1
        w.events.clear()
        path += float(np.linalg.norm(d.pos[:2] - last[:2])); last = d.pos.copy()
    return {"seed": seed, "success": bool(d.done and d.delivered), "delivered": bool(d.delivered),
            "time_to_goal_s": None if t_deliver is None else round(t_deliver, 2),
            "mission_time_s": round(d.t_done, 2) if d.done else None,
            "collisions": d.collisions, "gust_entries": d.gust_entries, "pond_entries": d.pond_entries,
            "escapes": escapes, "path_m": round(path, 1)}


def summarize(rows):
    def mean(k):
        v = [r[k] for r in rows if r[k] is not None]
        return None if not v else round(float(np.mean(v)), 3)
    return {"n": len(rows), "success_rate": round(float(np.mean([r["success"] for r in rows])), 3),
            "delivery_rate": round(float(np.mean([r["delivered"] for r in rows])), 3),
            "mean_time_to_goal_s": mean("time_to_goal_s"), "mean_mission_time_s": mean("mission_time_s"),
            "mean_collisions": mean("collisions"), "mean_gust_entries": mean("gust_entries"),
            "mean_pond_entries": mean("pond_entries"), "mean_escapes": mean("escapes")}


def paired_bootstrap(a, b, key, n=10000, seed=0):
    """95% interval for mean(a[key]) - mean(b[key]) over seeds (paired by seed)."""
    x = np.array([float(r[key]) for r in a]); y = np.array([float(r[key]) for r in b])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    diffs = (x[idx] - y[idx]).mean(axis=1)
    return {"diff": round(float((x - y).mean()), 3), "ci95": [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]}


def main(arms=("real", "rewired", "rewired_class", "reactive")):
    prev = json.loads(OUT.read_text()) if OUT.exists() else {"arms": {}}
    out = {"task": "courier: home -> red tower -> home; pond avoided by route; hidden gust zone; 120 s limit",
           "seeds": SEEDS, "tuning_seeds_excluded": [100, 101, 102], "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "precomputed": True, "arms": {}}
    for arm in arms:
        t0 = time.time()
        rows = [run_one(arm, s) for s in SEEDS]
        out["arms"][arm] = {"summary": summarize(rows), "runs": rows, "wall_s": round(time.time() - t0, 1)}
        print(arm, out["arms"][arm]["summary"], flush=True)
    for arm, v in prev.get("arms", {}).items():      # keep arms run earlier with the same seeds
        out["arms"].setdefault(arm, v)
    out["arm_notes"] = {"real": "MaleCNS FlyWorld sub-circuit as wired", "rewired": "targets shuffled across the whole graph; in/out degree and outgoing weights kept",
                        "rewired_class": "targets shuffled within superclass x side block pairs (Project 08 style); degrees kept",
                        "reactive": "hand-written steer-to-target with obstacle repulsion"}
    if "real" in out["arms"]:
        cmp = {}
        for other in [a for a in out["arms"] if a != "real"]:
            cmp[f"real_minus_{other}"] = {k: paired_bootstrap(out["arms"]["real"]["runs"], out["arms"][other]["runs"], k)
                                          for k in ("success", "delivered", "collisions", "gust_entries")}
        out["comparisons"] = cmp
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("real", "rewired", "rewired_class", "reactive"))
