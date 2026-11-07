"""SIE open models vs a frontier model on the swarm's everyday jobs: quality, latency, cost.

Jobs
  mission planner   30 operator missions with hand-written gold plans (bench/missions.jsonl)
  perception        40 keyframes rendered from the simulator, labelled from simulator geometry (bench/keyframes/)
Providers
  sie       models from backend/models.yaml
  frontier  FRONTIER_PROVIDER (anthropic: claude-opus-5-5 at effort 'low' by default); 'not run' if no key
Cost per 1,000 calls = mean(measured input/output tokens per call x published price) x 1,000 (config/prices.yaml).
Writes backend/bench/results/sie_benchmark.json.
"""
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..agents import jobs, llm
from ..sim import arena
from ..sim.world import shared_eye

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/sie_benchmark.json"
KF = HERE / "keyframes"


# ---------------- keyframes ----------------

def label_view(eye, view):
    """Ground truth from simulator geometry (not from a model)."""
    lm, depth = view["lm"], view["depth"]
    ids, cnt = np.unique(lm[lm >= 0], return_counts=True)
    if len(ids) == 0 or cnt.max() < 15:
        dom = "none"; bearing = "ahead"; dist = "far"; near_ahead = False
    else:
        k = ids[np.argmax(cnt)]
        dom = arena.LANDMARKS[k].name
        m = lm == k
        az = np.degrees(np.arctan2(np.sin(eye.az[m]).mean(), np.cos(eye.az[m]).mean()))
        bearing = "ahead" if abs(az) < 45 else ("left" if 45 <= az <= 135 else ("right" if -135 <= az <= -45 else "behind"))
        dmin = float(depth[m].min())
        dist = "near" if dmin < 6 else ("medium" if dmin < 15 else "far")
        near_ahead = dist == "near" and bearing == "ahead"
    return dom, bearing, dist, near_ahead


def make_keyframes(n=40, seed=7):
    eye = shared_eye()
    rng = np.random.default_rng(seed)
    KF.mkdir(exist_ok=True)
    rows = []
    while len(rows) < n:
        x, y = rng.uniform(-18, 18, 2)
        if arena.collide((x, y, 4.0), r=1.5):
            continue
        yaw = rng.uniform(-math.pi, math.pi)
        z = float(rng.uniform(3, 6))
        view = eye.cast((x, y, z), yaw)
        dom, bearing, dist, near_ahead = label_view(eye, view)
        # pond visible below: ground hits inside the pond circle
        c, s = math.cos(yaw), math.sin(yaw)
        D = eye.dirs; Dw = np.stack([c * D[:, 0] - s * D[:, 1], s * D[:, 0] + c * D[:, 1], D[:, 2]], 1)
        down = (Dw[:, 2] < -0.05) & (view["lm"] < 0)
        t = np.where(down, -z / np.where(down, Dw[:, 2], -1), np.inf)
        gx, gy = x + t * Dw[:, 0], y + t * Dw[:, 1]
        pond_px = int((down & (np.hypot(gx - arena.POND["center"][0], gy - arena.POND["center"][1]) <= arena.POND["radius"]) & (t < 25)).sum())
        hazard = "obstacle" if near_ahead else ("water" if pond_px >= 10 else "none")
        # balance: at most 8 'none' frames
        if dom == "none" and sum(r["label"]["landmark"] == "none" for r in rows) >= 8:
            continue
        name = f"kf{len(rows):02d}.png"
        (KF / name).write_bytes(eye.equirect_png(view["rgb"]))
        rows.append({"id": name, "pose": {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "heading_deg": round(math.degrees(yaw), 1)},
                     "label": {"landmark": dom, "bearing": bearing, "approx_distance": dist, "hazard": hazard}})
    (KF / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


# ---------------- scoring ----------------

def norm_plan(plan, joint):
    """Comparable form. With joint scouts, the scouts' survey landmarks are compared as one set."""
    drones = {d["drone"]: d for d in plan["drones"]}
    out = {"drones": sorted(drones)}
    per = {}
    for name, d in drones.items():
        per[name] = {"goals": [(g["landmark"], g["action"]) for g in d["goals"]],
                     "constraints": sorted({c["zone"] for c in d["constraints"]}), "return_home": d["return_home"]}
    if joint:
        scouts = [per.pop(s) for s in ("scout-1", "scout-2") if s in per]
        per["scouts"] = {"goals": sorted({g for s in scouts for g in s["goals"]}),
                         "constraints": sorted({c for s in scouts for c in s["constraints"]}),
                         "return_home": sorted({s["return_home"] for s in scouts})}
    out["per"] = per
    return out


def score_plan(pred, gold, joint):
    g = norm_plan(gold, joint)
    if pred is None:
        return {"exact": 0, "fields": 0.0, "n_fields": 1 + 3 * len(g["per"])}
    p = norm_plan(pred, joint)
    hits, n = int(p["drones"] == g["drones"]), 1
    for k, gv in g["per"].items():
        pv = p["per"].get(k, {})
        for f in ("goals", "constraints", "return_home"):
            n += 1; hits += int(pv.get(f) == gv[f])
    return {"exact": int(hits == n), "fields": hits / n, "n_fields": n}


FIELDS = ("landmark", "bearing", "approx_distance", "hazard")


def tree_family(x):
    return "green_tree" if str(x).startswith("green_tree") else x


def score_obs(pred, gold):
    if pred is None:
        return {"exact": 0, "fields": 0.0, "landmark_family": 0}
    hits = {f: int(pred[f] == gold[f]) for f in FIELDS}
    return {"exact": int(all(hits.values())), "fields": sum(hits.values()) / len(FIELDS),
            "landmark_family": int(tree_family(pred["landmark"]) == tree_family(gold["landmark"])), **{f"{f}_ok": v for f, v in hits.items()}}


# ---------------- runner ----------------

def run_provider(provider, missions, frames, workers=4):
    def plan_job(m):
        r = jobs.plan_mission(m["text"], provider=provider)
        s = score_plan(r.data.model_dump() if r.ok else None, m["gold"], m["scouts_joint"])
        return {"id": m["id"], "ok": r.ok, "error": r.error, "latency_s": round(r.latency_s, 3), "in": r.input_tokens, "out": r.output_tokens,
                "cost_usd": r.cost_usd(), "attempts": r.attempts, **s, "pred": r.data.model_dump() if r.ok else None}

    def obs_job(f):
        png = (KF / f["id"]).read_bytes()
        p = f["pose"]
        r = jobs.perceive(png, f"x={p['x']} y={p['y']} z={p['z']} heading={p['heading_deg']} deg", provider=provider)
        s = score_obs(r.data.model_dump() if r.ok else None, f["label"])
        return {"id": f["id"], "ok": r.ok, "error": r.error, "latency_s": round(r.latency_s, 3), "in": r.input_tokens, "out": r.output_tokens,
                "cost_usd": r.cost_usd(), "attempts": r.attempts, **s, "pred": r.data.model_dump() if r.ok else None}

    with ThreadPoolExecutor(workers) as ex:
        plans = list(ex.map(plan_job, missions))
        obs = list(ex.map(obs_job, frames))
    model = (llm.models()["mission_planner"], llm.models()["perception"]) if provider == "sie" else (llm.frontier_model(),) * 2
    return {"mission_planner": summarize(plans, model[0]), "perception": summarize(obs, model[1], extra=("landmark_family",) + tuple(f"{f}_ok" for f in FIELDS))}


def summarize(rows, model, extra=()):
    lat = [r["latency_s"] for r in rows]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] == r["cost_usd"]]
    s = {"model": model, "n": len(rows), "valid_json_rate": round(np.mean([r["ok"] for r in rows]), 3),
         "exact_match": round(float(np.mean([r["exact"] for r in rows])), 3),
         "field_accuracy": round(float(np.mean([r["fields"] for r in rows])), 3),
         "median_latency_s": round(statistics.median(lat), 3), "p90_latency_s": round(float(np.percentile(lat, 90)), 3),
         "mean_input_tokens": round(float(np.mean([r["in"] for r in rows])), 1), "mean_output_tokens": round(float(np.mean([r["out"] for r in rows])), 1),
         "cost_per_1000_calls_usd": round(1000 * float(np.mean(costs)), 4) if costs else None, "items": rows}
    for k in extra:
        s[k] = round(float(np.mean([r.get(k, 0) for r in rows])), 3)
    return s


def place_recognition_cost(frames):
    """SIE image embedding job (no frontier equivalent is benchmarked)."""
    pngs = [(KF / f["id"]).read_bytes() for f in frames[:20]]
    vecs, r = llm.embed_images(pngs)
    per_call = llm.call_cost("sie", llm.models()["place_embedding"], 0, 0, images=1)
    return {"model": llm.models()["place_embedding"], "batch_size": len(pngs), "batch_latency_s": round(r.latency_s, 3),
            "dims": int(vecs.shape[1]), "cost_per_1000_images_usd": round(per_call * 1000, 4)}


def main():
    missions = [json.loads(l) for l in (HERE / "missions.jsonl").read_text().splitlines() if l.strip()]
    frames = make_keyframes()
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "precomputed": True,
           "notes": ["Mission gold plans are hand-written (bench/missions.jsonl). With 'scouts_joint', the two scouts' survey landmarks are scored as one set.",
                     "Keyframe labels come from simulator geometry (dominant landmark by ommatidium count, bearing from its mean azimuth, distance from nearest depth, pond pixels for 'water').",
                     "Costs are measured tokens from each API's usage field times published list prices in backend/config/prices.yaml.",
                     "Frontier calls use structured output and effort 'low'; thinking tokens, if any, are billed as output and included.",
                     "v2: perception prompt defines bearing/distance/hazard and the panorama is upscaled 4x (nearest) for both providers; v1 results kept in sie_benchmark_v1.json."],
           "prices": llm.prices(), "missions": len(missions), "keyframes": len(frames), "providers": {}}
    if not llm.sie_configured():
        sys.exit("SIE_API_KEY not set")
    t0 = time.time(); out["providers"]["sie"] = run_provider("sie", missions, frames); print("sie done", round(time.time() - t0), flush=True)
    try:
        out["place_recognition"] = place_recognition_cost(frames)
    except Exception as e:
        out["place_recognition"] = {"error": f"{type(e).__name__}: {e}"}
    if llm.frontier_provider():
        t0 = time.time(); out["providers"]["frontier"] = run_provider(llm.frontier_provider(), missions, frames, workers=3)
        out["providers"]["frontier"]["provider"] = llm.frontier_provider(); print("frontier done", round(time.time() - t0), flush=True)
    else:
        out["providers"]["frontier"] = {"status": "not run", "reason": "no FRONTIER_API_KEY / ANTHROPIC_API_KEY"}
    OUT.write_text(json.dumps(out, indent=1, default=float))
    for p, v in out["providers"].items():
        if "mission_planner" in v:
            for job in ("mission_planner", "perception"):
                j = v[job]
                print(p, job, {k: j[k] for k in ("model", "valid_json_rate", "exact_match", "field_accuracy", "median_latency_s", "cost_per_1000_calls_usd")})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
