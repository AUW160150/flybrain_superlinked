"""Completion checks for GATES.md. Each subcommand prints '<NAME>_OK' only after every assertion passes.

  python -m backend.scripts.verify bench     bench JSON files are complete and internally consistent
  python -m backend.scripts.verify api       a fresh server serves health, arena, bench, plans a mission and streams moving frames
  python -m backend.scripts.verify docs      README (generated tables), DEMO, NOTES, LOG, Dockerfile, render.yaml
  python -m backend.scripts.verify replay    at least one recorded replay with moving drones
  python -m backend.scripts.verify pushed    working tree clean and HEAD equals origin/main
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "backend/bench/results"


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def bench():
    c = json.loads((R / "controllers.json").read_text())
    for arm in ("real", "rewired", "rewired_class", "reactive"):
        runs = c["arms"][arm]["runs"]
        if [r["seed"] for r in runs] != list(range(20)):
            fail(f"controllers {arm} seeds")
        sr = sum(r["success"] for r in runs) / 20
        if abs(sr - c["arms"][arm]["summary"]["success_rate"]) > 1e-9:
            fail(f"controllers {arm} summary does not match runs")
    s = json.loads((R / "sie_benchmark.json").read_text())
    if s["missions"] != 30 or s["keyframes"] != 40:
        fail("sie benchmark sizes")
    for p in ("sie", "frontier"):
        for job, n in (("mission_planner", 30), ("perception", 40)):
            j = s["providers"][p][job]
            if len(j["items"]) != n:
                fail(f"{p} {job} item count")
            em = sum(it["exact"] for it in j["items"]) / n
            if abs(em - j["exact_match"]) > 1e-3 or not j["cost_per_1000_calls_usd"] or j["cost_per_1000_calls_usd"] <= 0:
                fail(f"{p} {job} metrics inconsistent")
    m = json.loads((R / "memory_ablation.json").read_text())
    for arm in ("none", "private", "shared"):
        a = m["arms"][arm]
        if [r["seed"] for r in a["runs"]] != list(range(20)):
            fail(f"ablation {arm} seeds")
        if a["memory_backend"] != "cognee":
            fail(f"ablation {arm} not on cognee: {a['memory_backend']}")
        gr = sum(r["gust_entries"] > 0 for r in a["runs"]) / 20
        if abs(gr - a["summary"]["gust_entry_rate"]) > 1e-9:
            fail(f"ablation {arm} summary mismatch")
    if m["arms"]["shared"]["memory_event_counts"].get("denied", 0) < 1:
        fail("no cognee permission denial recorded in the shared arm")
    print("BENCH_FILES_OK")


def api():
    import httpx
    import websockets  # noqa: F401
    port = 8799
    env = dict(os.environ, MEMORY_BACKEND="local", DATA_DIR=str(ROOT / "var/verify"))
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.api.server:app", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                if httpx.get(base + "/healthz", timeout=2).json().get("ok"):
                    break
            except Exception:
                time.sleep(1)
        else:
            fail("server did not become healthy")
        if len(httpx.get(base + "/api/arena").json()["landmarks"]) < 8:
            fail("arena")
        for b in ("sie", "controllers", "memory", "probe"):
            if httpx.get(base + f"/api/bench/{b}").json().get("status") == "not run":
                fail(f"bench {b} not served")
        if len(httpx.get(base + "/api/eye/layout").json()["az"]) != 1709:
            fail("eye layout")
        r = httpx.post(base + "/api/missions", json={"text": "Courier, take the package to the red tower and come home, avoid the pond."}, timeout=120)
        if r.status_code != 200 or not r.json()["plan"]["drones"]:
            fail(f"mission: {r.status_code} {r.text[:200]}")
        import asyncio
        import websockets as ws_lib

        async def frames():
            async with ws_lib.connect(f"ws://127.0.0.1:{port}/ws/sim", max_size=2 ** 24) as ws:
                first = json.loads(await ws.recv())
                t0 = time.time()
                last = first
                while time.time() - t0 < 6:
                    last = json.loads(await ws.recv())
                return first, last
        first, last = asyncio.run(frames())
        c0 = next(d for d in first["drones"] if d["name"] == "courier")
        c1 = next(d for d in last["drones"] if d["name"] == "courier")
        moved = sum((a - b) ** 2 for a, b in zip(c0["pos"], c1["pos"])) ** 0.5
        if last["t"] <= first["t"] or moved < 2.0:
            fail(f"sim not advancing (t {first['t']} -> {last['t']}, moved {moved:.2f} m)")
        if not last.get("detail") or "mosaic" not in last["detail"]:
            fail("no eye mosaic in frames")
        print("API_OK")
    finally:
        proc.terminate()


def docs():
    readme = (ROOT / "README.md").read_text()
    seg = readme.split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0]
    if "not run" in seg or "| real connectome |" not in seg or "| shared swarm memory |" not in seg or "claude-opus" not in seg:
        fail("README results tables incomplete")
    for f in ("DEMO.md", "NOTES.md", "LOG.md", "Dockerfile", "requirements.txt", "../render.yaml"):
        if not (ROOT / f).exists():
            fail(f"missing {f}")
    if "```mermaid" not in readme:
        fail("README has no Mermaid diagram")
    rb = (ROOT / "../render.yaml").read_text()
    for k in ("SIE_API_KEY", "FRONTIER_API_KEY", "/var/data", "/healthz"):
        if k not in rb:
            fail(f"render.yaml missing {k}")
    print("DOCS_OK")


def replay():
    files = sorted((ROOT / "backend/replays").glob("*.json"))
    if not files:
        fail("no replays")
    d = json.loads(files[0].read_text())
    fr = d["frames"]
    if len(fr) < 50:
        fail("replay too short")
    p0 = fr[0]["drones"][0]["pos"]; p1 = fr[-1]["drones"][0]["pos"]
    if sum((a - b) ** 2 for a, b in zip(p0, p1)) ** 0.5 < 2:
        fail("replay drones do not move")
    print("REPLAY_OK")


def pushed():
    run = lambda *a: subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=ROOT)
    if run("git", "status", "--porcelain", "--", "."):
        fail("uncommitted changes in fly_superlinked")
    if run("git", "rev-parse", "HEAD") != run("git", "rev-parse", "origin/main"):
        fail("HEAD differs from origin/main")
    print("PUSHED_OK")


if __name__ == "__main__":
    import backend  # noqa: F401  (loads .env)
    {"bench": bench, "api": api, "docs": docs, "replay": replay, "pushed": pushed}[sys.argv[1]]()
