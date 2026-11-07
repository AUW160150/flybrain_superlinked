"""Measure which lateralised visual input groups produce a left/right asymmetry in steering DNs.

Writes bench/results/probe_circuit.json. Used to decide (and document in NOTES.md) the
eye -> circuit mapping; nothing here is tuned to make the real circuit look good.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from backend.brain.circuit import circuit_edges, load_circuit
from backend.brain.lif import BatchedLIF

READ = ["DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNa_all_L", "DNa_all_R", "DNp09", "DNp01_L", "DNp01_R", "DN_all"]
VIS = ["LC11", "HS", "LC4", "LC6", "LPLC2", "LC16"]


def run(brain_type, amps_list, ms=300):
    c = load_circuit()
    pre, post, w = circuit_edges(brain_type)
    B = len(amps_list)
    brain = BatchedLIF(c["N"], pre, post, w, c["groups"], B=B)
    brain.set_amps(amps_list)
    brain.step(100)                      # transient
    counts, _ = brain.step(ms)
    r = brain.rates_hz(counts, ms)
    return [{k: round(float(r[b, brain.index[k]]), 1) for k in READ if k in brain.index} for b in range(B)], brain


def main(amp=0.15, go=0.15):
    out = {"amp": amp, "dnp09_go": go, "window_ms": 300, "brains": {}}
    for bt in ("real", "rewired"):
        conds = [("rest", {}), ("go", {"DNp09": go})]
        for g in VIS:
            for side in ("L", "R"):
                conds.append((f"go+{g}_{side}", {"DNp09": go, f"{g}_{side}": amp}))
        t0 = time.time()
        res, brain = run(bt, [d for _, d in conds])
        out["brains"][bt] = {name: r for (name, _), r in zip(conds, res)}
        print(f"{bt}: {len(conds)} conditions in {time.time()-t0:.1f}s, E={brain.E}")
        for name, r in out["brains"][bt].items():
            print(f"  {name:14s} " + " ".join(f"{k}={v}" for k, v in r.items()))
    p = Path(__file__).resolve().parent / "results/probe_circuit.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*(float(x) for x in sys.argv[1:]))
