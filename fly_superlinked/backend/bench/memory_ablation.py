"""Memory ablation: does shared swarm memory help the courier?

Arms (each in a fresh memory store, same seeds 0-19 flown in order):
  none     courier recalls nothing
  private  courier recalls only private_courier, which fills with its own notes as the 20 missions go by
  shared   scouts survey first (once, seed 1000), writing to swarm_map; courier recalls swarm_map + outcomes + private
After every mission the operator writes the outcome, corrections, feedback and improve (the learning loop).
Perception (SIE vision) is off here to bound cost; hazards come from what drones physically feel.

Run one arm per process (cognee's configuration is process-wide):
  python -m backend.bench.memory_ablation <arm> <brain>      -> results/memory_ablation_<brain>_<arm>.json
  python -m backend.bench.memory_ablation merge <brain>       -> results/memory_ablation.json
"""
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = list(range(20))
T_MAX = 120.0
MISSION = "Courier, take the package to the red tower and come home, avoid the pond."
SCOUT_PLAN = {"drones": [
    {"drone": "scout-1", "goals": [{"landmark": "blue_arch", "action": "survey"}, {"landmark": "green_tree_2", "action": "survey"},
                                   {"landmark": "red_tower", "action": "survey"}], "constraints": [], "return_home": True},
    {"drone": "scout-2", "goals": [{"landmark": "green_tree_4", "action": "survey"}, {"landmark": "green_tree_5", "action": "survey"},
                                   {"landmark": "red_tower", "action": "survey"}], "constraints": [], "return_home": True}]}
COURIER_PLAN = {"drones": [{"drone": "courier", "goals": [{"landmark": "red_tower", "action": "deliver"}],
                            "constraints": [{"type": "avoid", "zone": "pond"}], "return_home": True}]}


async def fly(sw, plan_dict, t_max):
    from ..agents.jobs import MissionPlan
    sw.mission_id = f"{sw.seed}-{int(time.time() * 1000) % 100000}"
    sw.session_id = f"courier-{sw.mission_id}"
    sw.plan = MissionPlan.model_validate(plan_dict)
    sw.plan_meta = {"source": "fixed plan (ablation)"}
    for dp in sw.plan.drones:
        await sw._assign(dp)
    w = sw.world
    while w.t < t_max and not all(w.drones[dp.drone].done for dp in sw.plan.drones):
        sw.step()
        if sw.tasks:                      # memory reads/writes complete before the sim moves on
            await sw.drain()
    if not sw.outcome_written:
        sw.outcome_written = True
        await sw.finish_mission()
    await sw.drain()


async def run_arm(arm, brain):
    root = HERE.parent.parent / "var" / f"ablation_{brain}_{arm}"
    shutil.rmtree(root, ignore_errors=True)
    os.environ["DATA_DIR"] = str(root)
    from ..agents.swarm import Swarm
    from ..memory.store import SwarmMemory, cognee_configured
    mem = SwarmMemory()
    t0 = time.time()
    await mem.setup()
    rows = []
    if arm == "shared":
        sw = Swarm(mem, seed=1000, brain="reactive", memory_mode="none", perception=False, place_recognition=False)
        await fly(sw, SCOUT_PLAN, 150.0)
    for seed in SEEDS:
        sw = Swarm(mem, seed=seed, brain=brain, memory_mode=arm, perception=False, place_recognition=False)
        await fly(sw, COURIER_PLAN, T_MAX)
        d = sw.world.drones["courier"]
        rows.append({"seed": seed, "success": bool(d.done and d.delivered), "delivered": bool(d.delivered),
                     "mission_time_s": round(d.t_done - d.t_start, 2) if d.done else None,
                     "collisions": d.collisions, "gust_entries": d.gust_entries, "pond_entries": d.pond_entries,
                     "hazards_used": len(sw.leg_hazards.get("courier", [])),
                     "log": [l["msg"] for l in sw.log if "recalled" in l["msg"] or "replanned" in l["msg"]][:4]})
        print(arm, brain, rows[-1]["seed"], rows[-1]["success"], rows[-1]["gust_entries"], rows[-1]["hazards_used"], flush=True)
    ev = [e.as_dict() for e in mem.events]
    out = {"arm": arm, "brain": brain, "memory_backend": mem.backend.name, "runs": rows, "wall_s": round(time.time() - t0, 1),
           "memory_event_counts": {k: sum(e["kind"] == k for e in ev) for k in ("write", "read", "denied", "feedback", "improve", "error")},
           "denied_examples": [e for e in ev if e["kind"] == "denied"][:3], "errors": [e for e in ev if e["kind"] == "error"][:5]}
    (HERE / "results" / f"memory_ablation_{brain}_{arm}.json").write_text(json.dumps(out, indent=1, default=float))


def summarize(rows):
    m = lambda k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 3) if any(r[k] is not None for r in rows) else None
    return {"n": len(rows), "success_rate": round(float(np.mean([r["success"] for r in rows])), 3), "mean_mission_time_s": m("mission_time_s"),
            "mean_collisions": m("collisions"), "mean_gust_entries": m("gust_entries"),
            "gust_entry_rate": round(float(np.mean([r["gust_entries"] > 0 for r in rows])), 3), "mean_hazards_used": m("hazards_used"),
            "first5_gust_entry_rate": round(float(np.mean([r["gust_entries"] > 0 for r in rows[:5]])), 3)}


def merge(brain):
    out = {"protocol": __doc__.strip().split("\n\n")[1], "mission": MISSION, "seeds": SEEDS, "brain": brain,
           "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "precomputed": True, "arms": {}}
    for arm in ("none", "private", "shared"):
        p = HERE / "results" / f"memory_ablation_{brain}_{arm}.json"
        if p.exists():
            d = json.loads(p.read_text())
            out["arms"][arm] = {"summary": summarize(d["runs"]), **d}
    target = HERE / "results" / ("memory_ablation.json" if brain == "reactive" else f"memory_ablation_{brain}.json")
    target.write_text(json.dumps(out, indent=1, default=float))
    for a, v in out["arms"].items():
        print(a, v["summary"])


if __name__ == "__main__":
    import backend  # noqa: F401  (loads .env)
    if sys.argv[1] == "merge":
        merge(sys.argv[2] if len(sys.argv) > 2 else "reactive")
    else:
        asyncio.run(run_arm(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "reactive"))
