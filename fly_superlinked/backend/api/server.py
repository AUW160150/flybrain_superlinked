"""FastAPI app: REST + a 20 Hz WebSocket stream of the live simulation, plus replays and bench results.

Serves the built frontend from frontend/dist when present (single Docker web service on Render).
"""
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..agents import llm
from ..agents.swarm import Swarm
from ..memory.store import SwarmMemory
from ..sim import arena
from ..sim.world import shared_eye

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "backend/bench/results"
REPLAYS = Path(os.environ.get("REPLAY_DIR", ROOT / "backend/replays"))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "var"))
DIST = ROOT / "frontend/dist"
WS_HZ = 20.0
RUN_LIMIT_S = 150.0


class State:
    def __init__(self):
        self.memory = SwarmMemory()
        self.memory_status = "starting"
        self.swarm = None
        self.seed, self.brain, self.memory_mode = 0, "real", "shared"
        self.clients = {}            # websocket -> selected drone
        self.paused = False
        self.recording = []
        self.mem_cursor = 0
        self.log_cursor = 0
        self.call_cursor = 0
        self.speed = 1.0
        self.loop = None


S = State()


def new_swarm(loop=None):
    S.swarm = Swarm(S.memory, seed=S.seed, brain=S.brain, memory_mode=S.memory_mode)
    S.swarm.loop = loop or S.loop
    S.recording = []
    S.log_cursor = 0; S.call_cursor = 0


async def setup_memory(attempts=3):
    """cognee's first cognify can fail on a malformed model reply; retry before falling back."""
    from ..memory.store import CogneeBackend, cognee_configured
    err = None
    for i in range(attempts):
        try:
            await S.memory.setup()
            S.memory_status = S.memory.backend.name
            return
        except Exception as e:
            err = e
            S.memory_status = f"starting (cognee setup attempt {i + 1} failed, retrying)"
            await asyncio.sleep(5)
            if cognee_configured():
                S.memory = SwarmMemory(CogneeBackend())
                if S.swarm:
                    S.swarm.memory = S.memory
    try:
        raise err
    except Exception as e:
        # cognee failed to start: fall back loudly rather than pretending
        from ..memory.store import LocalBackend
        S.memory = SwarmMemory(LocalBackend())
        await S.memory.setup()
        S.memory_status = f"local fallback (cognee setup failed: {type(e).__name__}: {str(e)[:120]})"
        if S.swarm:
            S.swarm.memory = S.memory


@asynccontextmanager
async def lifespan(app):
    S.loop = asyncio.get_running_loop()
    new_swarm()
    asyncio.create_task(setup_memory())
    task = asyncio.create_task(sim_loop())
    yield
    task.cancel()


app = FastAPI(title="Flybrain Swarm", lifespan=lifespan)


async def sim_loop():
    last_send = 0.0
    while True:
        t0 = time.perf_counter()
        sw = S.swarm
        active = sw is not None and any(d.active and not d.done for d in sw.world.drones.values())
        if sw is not None and active and not S.paused and sw.world.t < RUN_LIMIT_S:
            await asyncio.to_thread(sw.step)
            if sw.world.t >= RUN_LIMIT_S and not sw.outcome_written:
                sw.outcome_written = True
                sw._spawn(sw.finish_mission())
        now = time.perf_counter()
        if now - last_send >= 1.0 / WS_HZ and S.clients:
            last_send = now
            await broadcast()
        dt = (0.04 / S.speed) - (time.perf_counter() - t0)
        await asyncio.sleep(max(0.001, dt))


def drone_detail(sw, name):
    w = sw.world
    if name not in w.drones:
        return None
    d = w.drones[name]
    cmd, sig, stripe = w.last_brain.get(name, ({}, {}, None))
    out = {"name": name, "brain": w.brain_type}
    if d.last_view is not None:
        out["mosaic"] = w.eye.mosaic_bytes(d.last_view["rgb"])
        out["stripe"] = np.flatnonzero(stripe).tolist() if stripe is not None else []
    if sig:
        out["eye"] = {k: round(float(sig[k]), 3) for k in ("motion_l", "motion_r", "cover_l", "cover_r", "expand_l", "expand_r")}
    if cmd:
        out["cmd"] = {k: (round(float(v), 3) if isinstance(v, (int, float, np.floating)) else v)
                      for k, v in cmd.items() if k in ("yaw_rate", "v", "escape", "dn_left", "dn_right", "delta", "dnp09", "giant_fiber_spikes", "escape_dir")}
        out["rates"] = cmd.get("rates")
        if "raster" in cmd:
            r = cmd["raster"]
            nz = np.nonzero(r)
            out["spikes"] = [[int(i), int(t)] for i, t in zip(*nz)]
    return out


def frame(sw, selected):
    snap = sw.world.snapshot()
    snap.update({"seed": sw.seed, "memory_mode": sw.memory_mode, "mode": "live", "memory_backend": S.memory_status,
                 "sie": llm.sie_configured(), "frontier": llm.frontier_provider() or "not configured",
                 "plan": sw.plan.model_dump() if sw.plan else None, "plan_source": sw.plan_meta.get("source"),
                 "detail": drone_detail(sw, selected), "place_matches": sw.place_matches[-5:],
                 "gust_known": any(e.kind in ("write", "denied") and "gust" in e.text for e in S.memory.events)})
    return snap


async def broadcast():
    sw = S.swarm
    new_mem = [e.as_dict() for e in S.memory.events[S.mem_cursor:]]
    S.mem_cursor = len(S.memory.events)
    new_log = sw.log[S.log_cursor:] if S.log_cursor <= len(sw.log) else sw.log
    S.log_cursor = len(sw.log)
    new_calls = sw.sie_calls[S.call_cursor:]
    S.call_cursor = len(sw.sie_calls)
    dead = []
    rec = None
    for ws, sel in list(S.clients.items()):
        f = frame(sw, sel)
        f["memory_events"], f["log"], f["calls"] = new_mem, new_log, new_calls
        if rec is None and any(d["active"] and not d["done"] for d in f["drones"]):
            rec = f
        try:
            await ws.send_text(json.dumps(f, default=float))
        except Exception:
            dead.append(ws)
    for ws in dead:
        S.clients.pop(ws, None)
    if rec is not None and len(S.recording) < 4000:
        S.recording.append(rec)


@app.websocket("/ws/sim")
async def ws_sim(ws: WebSocket):
    await ws.accept()
    S.clients[ws] = "courier"
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if "select" in msg:
                S.clients[ws] = msg["select"]
    except WebSocketDisconnect:
        S.clients.pop(ws, None)


class MissionIn(BaseModel):
    text: str


class ResetIn(BaseModel):
    seed: int = 0
    brain: str = "real"
    memory_mode: str = "shared"


@app.post("/api/missions")
async def post_mission(m: MissionIn):
    if len(m.text) > 500:
        raise HTTPException(400, "mission text too long")
    sw = S.swarm
    if any(d.active and not d.done for d in sw.world.drones.values()) or sw.world.t > 0:
        new_swarm(); sw = S.swarm
    try:
        plan = await sw.start_mission(m.text)
    except Exception as e:
        raise HTTPException(502, f"planner failed: {e}")
    return {"plan": plan.model_dump(), "source": sw.plan_meta.get("source")}


@app.post("/api/sim/reset")
async def reset(r: ResetIn):
    if r.brain not in ("real", "rewired", "rewired_class", "reactive") or r.memory_mode not in ("none", "private", "shared"):
        raise HTTPException(400, "bad brain or memory mode")
    S.seed, S.brain, S.memory_mode = r.seed, r.brain, r.memory_mode
    new_swarm()
    return {"ok": True, "seed": S.seed, "brain": S.brain, "memory_mode": S.memory_mode}


@app.post("/api/sim/save_replay")
async def save_replay(name: str = "latest"):
    if not S.recording:
        raise HTTPException(400, "nothing recorded yet")
    REPLAYS.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in name if c.isalnum() or c in "-_")[:40] or "latest"
    path = REPLAYS / f"{safe}.json"
    path.write_text(json.dumps({"name": safe, "recorded": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": S.recording[::2]}, default=float))
    return {"saved": path.name, "frames": len(S.recording[::2])}


@app.get("/api/replays")
def list_replays():
    files = sorted(REPLAYS.glob("*.json")) if REPLAYS.exists() else []
    return [{"name": f.stem, "bytes": f.stat().st_size} for f in files]


@app.get("/api/replays/{name}")
def get_replay(name: str):
    path = REPLAYS / f"{''.join(c for c in name if c.isalnum() or c in '-_')}.json"
    if not path.exists():
        raise HTTPException(404, "no such replay")
    return FileResponse(path, media_type="application/json")


def bench_file(name):
    p = BENCH / name
    if not p.exists():
        return JSONResponse({"status": "not run", "file": name}, status_code=200)
    return FileResponse(p, media_type="application/json")


@app.get("/api/bench/sie")
def bench_sie():
    return bench_file("sie_benchmark.json")


@app.get("/api/bench/controllers")
def bench_controllers():
    return bench_file("controllers.json")


@app.get("/api/bench/memory")
def bench_memory():
    return bench_file("memory_ablation.json")


@app.get("/api/bench/probe")
def bench_probe():
    return bench_file("probe_circuit.json")


@app.get("/api/memory/graph")
async def memory_graph():
    """Nodes and edges from swarm_map and outcomes only: the datasets the operator can read."""
    return await S.memory.graph()


@app.get("/api/memory/events")
def memory_events():
    return [e.as_dict() for e in S.memory.events[-200:]]


@app.get("/api/arena")
def get_arena():
    return arena.describe()


@app.get("/api/eye/layout")
def eye_layout():
    return shared_eye().layout()


@app.get("/api/status")
def status():
    sw = S.swarm
    return {"memory_backend": S.memory_status, "sie": llm.sie_configured(), "frontier": llm.frontier_provider() or "not configured",
            "models": llm.models(), "seed": S.seed, "brain": S.brain, "memory_mode": S.memory_mode,
            "t": sw.world.t if sw else 0, "raster_neurons": [{"type": t, "side": s, "group": g} for _, t, s, g in (sw.world.fly.raster if sw and sw.world.fly else [])],
            "calibration": getattr(sw.world.fly, "calibration", None) if sw and sw.world.fly else None}


@app.get("/healthz")
def healthz():
    return {"ok": True}


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
