"""Turn an ordered goal list plus known hazards into waypoints (visibility graph + Dijkstra).

Hazards are circles (x, y, r). The pond is always known (it is on every map); landmarks are
obstacles; gust hazards come only from memory recall.
"""
import heapq
import math

from ..sim import arena
from ..sim.world import Waypoint

MARGIN = 1.6


def static_obstacles():
    obs = []
    for lm in arena.LANDMARKS:
        for p in lm.prims:
            if p.kind == "cyl":
                obs.append((p.center[0], p.center[1], p.size[0] + MARGIN, lm.name))
            else:
                obs.append((p.center[0], p.center[1], math.hypot(p.size[0], p.size[1]) + MARGIN, lm.name))
    return obs


def _seg_hits(a, b, c, skip_r=0.0):
    (ax, ay), (bx, by), (cx, cy, r) = a, b, c[:3]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / L2))
    px, py = ax + t * dx, ay + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2 < (r - 1e-6) ** 2


def _inside(p, c):
    return (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 < c[2] ** 2


def shortest_path(a, b, circles):
    """Shortest collision-free polyline a -> b around circles. A circle that contains an endpoint is
    shrunk to just exclude it (an obstacle the drone starts beside, or a hazard reaching over the goal),
    so the rest of it is still avoided."""
    active = []
    for c in circles:
        r = c[2]
        for p in (a, b):
            if _inside(p, (c[0], c[1], r)):
                r = math.dist(p, c[:2]) - 0.3
        if r > 0.5:
            active.append((c[0], c[1], r, c[3]))
    nodes = [a, b]
    for c in active:
        for k in range(12):
            th = 2 * math.pi * k / 12
            p = (c[0] + 1.12 * c[2] * math.cos(th), c[1] + 1.12 * c[2] * math.sin(th))
            if abs(p[0]) < arena.HALF - 0.5 and abs(p[1]) < arena.HALF - 0.5 and not any(_inside(p, o) for o in active):
                nodes.append(p)
    n = len(nodes)
    dist = [math.inf] * n; prev = [-1] * n; dist[0] = 0.0
    pq = [(0.0, 0)]
    while pq:
        dcur, i = heapq.heappop(pq)
        if i == 1:
            break
        if dcur > dist[i]:
            continue
        for j in range(n):
            if j == i:
                continue
            if any(_seg_hits(nodes[i], nodes[j], c) for c in active):
                continue
            nd = dcur + math.dist(nodes[i], nodes[j])
            if nd < dist[j]:
                dist[j] = nd; prev[j] = i; heapq.heappush(pq, (nd, j))
    if dist[1] == math.inf:
        return [b]
    path, k = [], 1
    while k != 0:
        path.append(nodes[k]); k = prev[k]
    return path[::-1]


def plan_route(start, goals, hazards=(), avoid_pond=True):
    """goals: list of (x, y, label, kind). hazards: list of (x, y, r, label). Returns Waypoint list."""
    circles = static_obstacles() + [tuple(h) for h in hazards]
    if avoid_pond:
        circles.append((arena.POND["center"][0], arena.POND["center"][1], arena.POND["radius"] + 1.2, "pond"))
    wps, cur = [], tuple(start)
    for gx, gy, label, kind in goals:
        # the goal point itself may sit beside its landmark; let the drone approach it
        own = [c for c in circles if c[3] != label]
        pts = shortest_path(cur, (gx, gy), own)
        for p in pts[:-1]:
            wps.append(Waypoint(round(p[0], 2), round(p[1], 2), label=f"detour to {label}", kind="route"))
        wps.append(Waypoint(gx, gy, label=label, kind=kind))
        cur = (gx, gy)
    return wps
