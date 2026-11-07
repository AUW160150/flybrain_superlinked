"""The swarm: plans missions, flies them, writes and recalls shared memory, and learns from outcomes.

Memory modes for the courier's planning (the ablation's three arms):
  none     recalls nothing; knows only the static map (landmarks, pond)
  private  recalls only private_courier (its own past notes)
  shared   recalls swarm_map + outcomes + private_courier
"""
import asyncio
import math
import re
import time
import uuid

import numpy as np

from ..memory.store import SwarmMemory, make_tag, new_fact_id, parse_tags
from ..sim import arena
from ..sim.world import CRUISE_Z, World
from . import jobs, llm
from .route import plan_route

HAZARD_R = 3.0
KEYFRAME_S = 2.0


def fallback_plan(text):
    """Keyword planner used only when SIE is not configured. Labelled as such in the UI and LOG."""
    t = text.lower()
    drones = []
    north = [lm.name for lm in arena.LANDMARKS if lm.goal[1] > 0]
    if "scout" in t:
        half = (north[::2], north[1::2])
        for i, name in enumerate(("scout-1", "scout-2")):
            if "scouts" in t or name in t:
                drones.append({"drone": name, "goals": [{"landmark": g, "action": "survey"} for g in half[i]],
                               "constraints": [], "return_home": "home" in t or "return" in t or True})
    if "courier" in t:
        goals = [{"landmark": lm.name, "action": "deliver"} for lm in arena.LANDMARKS if lm.label in t or lm.name.replace("_", " ") in t][:1]
        cons = [{"type": "avoid", "zone": "pond"}] if "pond" in t else []
        drones.append({"drone": "courier", "goals": goals or [{"landmark": "red_tower", "action": "deliver"}],
                       "constraints": cons, "return_home": "home" in t or "return" in t})
    return jobs.MissionPlan.model_validate({"drones": drones or [{"drone": "courier", "goals": [{"landmark": "red_tower", "action": "deliver"}], "constraints": [], "return_home": True}]})


def place_xy(name):
    if name == "home":
        return arena.HOME
    return arena.LANDMARK_BY_NAME[name].goal


def hazards_from_recall(hits):
    """Avoid-circles from recalled facts. Corrections remove hazards; confirmations raise confidence."""
    hazards, removed = {}, set()
    for h in hits:
        for tag in parse_tags(h["text"]):
            if tag.get("kind") == "hazard" and tag.get("hazard") == "gust":
                key = (round(float(tag["x"]) / 2), round(float(tag["y"]) / 2))
                hazards[key] = (float(tag["x"]), float(tag["y"]), float(tag.get("r", HAZARD_R)), "gust")
            if tag.get("kind") == "correction" and tag.get("hazard") == "gust":
                removed.add((round(float(tag["x"]) / 2), round(float(tag["y"]) / 2)))
    return [v for k, v in hazards.items() if k not in removed]


class Swarm:
    def __init__(self, memory: SwarmMemory, seed=0, brain="real", memory_mode="shared", perception=True, place_recognition=True):
        self.memory = memory
        self.seed, self.brain, self.memory_mode = seed, brain, memory_mode
        self.world = World(seed=seed, brain=brain)
        self.perception_on = perception and llm.sie_configured()
        self.place_on = place_recognition and llm.sie_configured()
        self.plan = None
        self.plan_meta = {}
        self.log = []                     # mission log lines for the UI
        self.tasks = set()
        self.keyframes = {n: [] for n in self.world.order}   # (t, pos, vec)
        self.place_matches = []
        self.leg_hazards = {}
        self.session_id = None
        self.mission_id = None
        self.outcome_written = False
        self.sie_calls = []               # CallResult summaries for live cost display
        self.remaining = {}

    # ---------- planning ----------
    async def start_mission(self, text):
        self.mission_id = uuid.uuid4().hex[:8]
        self.session_id = f"courier-{self.mission_id}"
        self.outcome_written = False
        if llm.sie_configured():
            res = await asyncio.to_thread(jobs.plan_mission, text)
            self._record_call("mission_planner", res)
            if not res.ok:
                self.say(f"Planner failed after a retry: {res.error}")
                raise RuntimeError(f"planner failed: {res.error}")
            plan, source = res.data, f"SIE {res.model}"
        else:
            plan, source = fallback_plan(text), "keyword fallback (SIE not configured)"
        self.plan, self.plan_meta = plan, {"source": source, "text": text}
        self.say(f"Plan from {source}: " + "; ".join(f"{d.drone} -> {', '.join(g.landmark for g in d.goals)}"
                                                      + (" then home" if d.return_home else "") for d in plan.drones))
        for dp in plan.drones:
            await self._assign(dp)
        return plan

    async def _assign(self, dp):
        d = self.world.drones[dp.drone]
        goals = [(*place_xy(g.landmark), g.landmark, "deliver" if g.action == "deliver" else "goal") for g in dp.goals]
        if dp.return_home:
            goals.append((*arena.HOME, "home", "home"))
        avoid_pond = True              # the pond is a no-fly zone on every map
        hazards = []
        if dp.drone == "courier":
            hazards = await self.recall_hazards(goals)
        self.leg_hazards[dp.drone] = hazards
        self.remaining[dp.drone] = goals
        wps = plan_route(d.pos[:2], goals, hazards=hazards, avoid_pond=avoid_pond)
        self.world.assign(dp.drone, wps)
        if hazards:
            self.say(f"{dp.drone} routes around {len(hazards)} remembered gust hazard(s).")

    async def _replan_leg(self, n, reached):
        """Before each new leg the courier recalls again, so hazards found by scouts meanwhile are used."""
        goals = self.remaining.get(n, [])
        while goals and goals[0][2] != reached:
            goals = goals[1:]
        goals = goals[1:]
        self.remaining[n] = goals
        if not goals or self.memory_mode == "none":
            return
        hazards = await self.recall_hazards(goals)
        self.leg_hazards[n] = list({h[:2]: h for h in self.leg_hazards.get(n, []) + hazards}.values())
        d = self.world.drones[n]
        wps = plan_route(d.pos[:2], goals, hazards=hazards)
        d.waypoints = d.waypoints[:d.wp_i] + wps
        self.say(f"{n} replanned its next leg from memory ({len(hazards)} gust hazard(s)).")

    async def recall_hazards(self, goals):
        if self.memory_mode == "none":
            return []
        datasets = ["private_courier"] if self.memory_mode == "private" else ["swarm_map", "outcomes", "private_courier"]
        dest = ", ".join(g[2].replace("_", " ") for g in goals if g[3] != "home")
        q = f"Where is the {dest}? What hazards such as gusts lie between home and the {dest}? Which routes failed before?"
        hits = await self.memory.recall("courier", q, datasets, session_id=self.session_id)
        hz = hazards_from_recall(hits)
        self.say(f"courier recalled {len(hits)} memories from {', '.join(datasets)}; {len(hz)} gust hazard(s) to avoid.")
        return hz

    # ---------- flight ----------
    def step(self):
        w = self.world
        w.step()
        for ev in w.events:
            self._on_event(ev)
        w.events.clear()
        for n in w.order:
            d = w.drones[n]
            if d.active and not d.done and w.t - d.last_keyframe >= KEYFRAME_S:
                vis = w.visible_landmarks(n, min_px=8)
                new = [lm for lm in vis if w.t - d.seen.get(lm, -99) > 8.0]
                if new or w.t - d.last_keyframe >= 4 * KEYFRAME_S:
                    d.last_keyframe = w.t
                    for lm in vis:
                        d.seen[lm] = w.t
                    self._keyframe(n, vis)
        if not self.outcome_written and self.plan is not None and all(
                (not w.drones[dp.drone].active) or w.drones[dp.drone].done for dp in self.plan.drones):
            self.outcome_written = True
            self._spawn(self.finish_mission())

    def _spawn(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        t = loop.create_task(coro)
        self.tasks.add(t); t.add_done_callback(self.tasks.discard)
        return t

    async def drain(self):
        while self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)

    def _on_event(self, ev):
        n = ev.get("drone")
        if ev["type"] == "gust_felt":
            x, y = ev["pos"][0], ev["pos"][1]
            fid = new_fact_id()
            text = (f"{n} felt a strong gust (disturbance {ev['disturbance']:.1f} m/s) near ({x:.1f}, {y:.1f}) at t={ev['t']:.0f}s. "
                    + make_tag(kind="hazard", hazard="gust", x=round(x, 1), y=round(y, 1), r=HAZARD_R, by=n, id=fid))
            self.say(f"{n} felt a gust at ({x:.1f}, {y:.1f}).")
            self._spawn(self._share(n, text, ["hazard", "gust"]))
        elif ev["type"] == "waypoint" and n == "courier" and ev["kind"] in ("deliver", "goal"):
            self._spawn(self._replan_leg(n, ev["label"]))
        if ev["type"] in ("delivered", "collision", "escape", "mission_done", "pond"):
            self.say(f"{n}: {ev['type']}" + (f" ({ev.get('with')})" if ev.get("with") else ""))

    async def _share(self, drone, text, node_set):
        """Scouts share to swarm_map. The courier tries too: it lacks write permission, so cognee denies it
        and the note goes to its private dataset instead."""
        ok = await self.memory.write(drone, "swarm_map", text, node_set)
        if not ok:
            await self.memory.write(drone, f"private_{drone}", text, node_set)

    def _keyframe(self, n, visible):
        d = self.world.drones[n]
        png = self.world.eye.equirect_png(d.last_view["rgb"])
        pose = f"x={d.pos[0]:.1f} y={d.pos[1]:.1f} z={d.pos[2]:.1f} heading={math.degrees(d.yaw):.0f} deg"
        if self.perception_on:
            self._spawn(self._perceive(n, png, pose, d.pos.copy(), visible))
        if self.place_on:
            self._spawn(self._place(n, png, d.pos.copy()))

    async def _perceive(self, n, png, pose, pos, visible):
        res = await asyncio.to_thread(jobs.perceive, png, pose)
        self._record_call("perception", res)
        if not res.ok:
            self.say(f"{n} perception failed: {res.error[:80]}")
            return
        o = res.data
        if o.landmark == "none":
            return
        truth = "matches" if o.landmark in visible else "not in ground-truth view"
        text = (f"{n} saw the {o.landmark.replace('_', ' ')} {o.bearing}, {o.approx_distance}, from ({pos[0]:.1f}, {pos[1]:.1f}); hazard {o.hazard}. "
                + make_tag(kind="observation", landmark=o.landmark, bearing=o.bearing, dist=o.approx_distance, hazard=o.hazard,
                           x=round(float(pos[0]), 1), y=round(float(pos[1]), 1), by=n, id=new_fact_id(), check=truth.replace(" ", "_")))
        await self._share(n, text, ["observation", o.landmark])

    async def _place(self, n, png, pos):
        try:
            vecs, res = await asyncio.to_thread(llm.embed_images, [png])
        except Exception as e:
            self.say(f"place recognition unavailable: {type(e).__name__}")
            self.place_on = False
            return
        self._record_call("place_embedding", res)
        v = vecs[0]
        past = [(t, p, u) for (t, p, u) in self.keyframes[n] if self.world.t - t > 10.0]
        if past:
            sims = np.array([float(u @ v) for _, _, u in past])
            i = int(np.argmax(sims))
            if sims[i] > 0.93:
                err = float(np.linalg.norm(past[i][1][:2] - pos[:2]))
                m = {"drone": n, "t": round(self.world.t, 1), "similarity": round(float(sims[i]), 3), "then_t": round(past[i][0], 1),
                     "then_pos": past[i][1][:2].round(1).tolist(), "now_pos": pos[:2].round(1).tolist(), "distance_m": round(err, 2)}
                self.place_matches.append(m)
                self.say(f"{n}: I have seen this place before (t={past[i][0]:.0f}s, similarity {sims[i]:.2f}, {err:.1f} m apart).")
        self.keyframes[n].append((self.world.t, pos, v))

    # ---------- learning loop ----------
    async def finish_mission(self):
        w = self.world
        for dp in self.plan.drones:
            d = w.drones[dp.drone]
            ok = d.done and (d.delivered or not any(g.action == "deliver" for g in dp.goals))
            why = []
            if not d.done:
                why.append("did not finish in time")
            if d.gust_entries:
                why.append(f"entered the gust zone {d.gust_entries} time(s)")
            if d.collisions:
                why.append(f"{d.collisions} collision(s)")
            route = " -> ".join(f"({p.x:.0f},{p.y:.0f})" for p in d.waypoints)
            text = (f"Mission {self.mission_id}: {dp.drone} {'succeeded' if ok else 'failed'} in {((d.t_done or w.t) - d.t_start):.0f}s, "
                    f"brain={self.brain}, memory={self.memory_mode}, route {route}. " + ("Problems: " + "; ".join(why) + ". " if why else "")
                    + make_tag(kind="outcome", drone=dp.drone, success=int(ok), gust_entries=d.gust_entries, collisions=d.collisions,
                               id=new_fact_id()))
            await self.memory.write("operator", "outcomes", text, ["outcome", dp.drone])
            if dp.drone == "courier":
                await self._correct_hazards(d)
                score = 5 if ok and d.gust_entries == 0 else (3 if ok else 1)
                await self.memory.feedback("courier", self.session_id, score,
                                           f"route {'avoided' if d.gust_entries == 0 else 'crossed'} gusts; mission {'ok' if ok else 'failed'}")
        await self.memory.improve("operator", "outcomes", [self.session_id] if self.session_id else None)
        self.say("Outcome written to 'outcomes'; feedback and improve applied.")

    async def _correct_hazards(self, d):
        """Hazards the courier flew through without feeling anything get a correction from the operator."""
        felt = [p[:2] for p in d.gust_reports]
        for (x, y, r, _) in self.leg_hazards.get("courier", []):
            through = any(math.hypot(p[0] - x, p[1] - y) < r * 0.6 for p in d.trail)
            if through and not any(math.hypot(p[0] - x, p[1] - y) < r for p in felt):
                text = (f"Correction: the courier flew through ({x:.1f}, {y:.1f}) and felt no gust. "
                        + make_tag(kind="correction", hazard="gust", x=x, y=y, by="operator", id=new_fact_id()))
                await self.memory.write("operator", "outcomes", text, ["correction"])

    # ---------- misc ----------
    def _record_call(self, job, res):
        self.sie_calls.append({"job": job, "ok": res.ok, "provider": res.provider, "model": res.model,
                               "latency_s": round(res.latency_s, 2), "cost_usd": res.cost_usd(), "t": round(self.world.t, 1)})

    def say(self, msg):
        self.log.append({"t": round(self.world.t, 1), "msg": msg})
        self.log = self.log[-200:]
