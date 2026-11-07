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


def class_preserving_rewire(pre, post, weight, block, seed, max_rounds=200):
    """Like degree_preserving_rewire, but targets are shuffled only among edges that join the same
    (source block, target block) pair, block = superclass x side. This is the Project 08 style control
    (role x side blocks, degrees kept): it keeps how many synapses each class sends to each class.
    In a few small, dense block pairs a duplicate-free shuffle does not exist; those leftover parallel
    edges (about 0.03% here) are kept and simply add their weights."""
    rng = np.random.default_rng(seed)
    pre = np.asarray(pre, dtype=np.int64); post = np.asarray(post, dtype=np.int64).copy()
    _, binv = np.unique(np.asarray(block), return_inverse=True)
    pair = binv[pre] * (binv.max() + 1) + binv[post]
    groups = [np.flatnonzero(pair == g) for g in np.unique(pair)]
    for idx in groups:
        post[idx] = post[idx][rng.permutation(len(idx))]
    N = int(max(pre.max(), post.max())) + 1
    group_of = np.empty(len(post), dtype=np.int64)
    for gi, idx in enumerate(groups):
        group_of[idx] = gi
    for _ in range(max_rounds):
        key = pre * N + post
        _, first = np.unique(key, return_index=True)
        dup = np.ones(len(key), bool); dup[first] = False
        bad = np.flatnonzero(dup | (pre == post))
        if len(bad) == 0:
            break
        for b in bad:                          # swap with a random edge of the same block pair
            idx = groups[group_of[b]]
            if len(idx) < 2:
                continue
            j = idx[rng.integers(len(idx))]
            post[b], post[j] = post[j], post[b]
    return pre.astype(np.int32), post.astype(np.int32), np.asarray(weight, dtype=np.float32).copy()


@lru_cache(maxsize=4)
def circuit_edges(brain_type, seed=1001):
    c = load_circuit()
    if brain_type == "real":
        return c["pre"], c["post"], c["weight"]
    if brain_type == "rewired":
        return degree_preserving_rewire(c["pre"], c["post"], c["weight"], seed)
    if brain_type == "rewired_class":
        block = np.char.add(np.char.add(c["superclass"].astype(str), ":"), c["side"].astype(str))
        return class_preserving_rewire(c["pre"], c["post"], c["weight"], block, seed)
    raise ValueError(brain_type)
