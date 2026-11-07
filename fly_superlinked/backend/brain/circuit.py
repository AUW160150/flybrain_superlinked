"""Load the FlyWorld sensory-to-descending sub-circuit of MaleCNS v1.0 and build controls.

Data files are copied unchanged from flybrain/projects/flyworld-game/processed/
(circuit_npc.npz: 10,203 neurons, 401,680 edges with >= 5 synapses; circuit_npc_groups.json:
named groups built by flyworld/groups.py). MaleCNS v1.0, Berg et al. 2026, CC BY 4.0.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=1)
def load_circuit():
    z = np.load(DATA / "circuit_npc.npz", allow_pickle=False)
    c = {k: z[k] for k in z.files}
    c["N"] = len(c["orig_index"])
    with open(DATA / "circuit_npc_groups.json") as f:
        c["groups"] = {k: np.asarray(v, dtype=np.int64) for k, v in json.load(f).items()}
    return c


def degree_preserving_rewire(pre, post, weight, seed, max_rounds=200):
    """Shuffle targets among edges, keeping every neuron's in-degree and out-degree exactly.

    Each edge keeps its source neuron and its signed weight (so a neuron's outgoing transmitter
    sign and synapse counts stay with it); only the target changes. Targets are permuted globally,
    which preserves the in-degree multiset per neuron; self-loops and duplicate edges created by
    the permutation are repaired by repeated swaps of targets between conflicting and random edges.
    """
    rng = np.random.default_rng(seed)
    pre = np.asarray(pre, dtype=np.int64)
    post = np.asarray(post, dtype=np.int64)[rng.permutation(len(post))]
    N = int(max(pre.max(), post.max())) + 1
    for _ in range(max_rounds):
        key = pre * N + post
        _, first = np.unique(key, return_index=True)
        dup = np.ones(len(key), bool); dup[first] = False
        bad = np.flatnonzero(dup | (pre == post))
        if len(bad) == 0:
            break
        # distinct partners outside the bad set, so each swap is a true transposition (multiset preserved)
        pool = np.setdiff1d(np.arange(len(post)), bad, assume_unique=True)
        partners = rng.choice(pool, size=len(bad), replace=False)
        post[bad], post[partners] = post[partners].copy(), post[bad].copy()
    else:
        raise RuntimeError("rewiring did not converge")
    return pre.astype(np.int32), post.astype(np.int32), np.asarray(weight, dtype=np.float32).copy()


@lru_cache(maxsize=4)
def circuit_edges(brain_type, seed=1001):
    c = load_circuit()
    if brain_type == "real":
        return c["pre"], c["post"], c["weight"]
    if brain_type == "rewired":
        return degree_preserving_rewire(c["pre"], c["post"], c["weight"], seed)
    raise ValueError(brain_type)
