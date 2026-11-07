"""Batched leaky integrate-and-fire simulation of a connectome sub-graph (CPU torch).

Reused from flybrain/projects/flyworld-game/flyworld/brain.py (BatchedLIF): same parameters
(1 ms steps, tau 20 ms, threshold 1, 2 ms refractory, weight = signed synapse count x gain 0.005,
event-driven propagation, runaway governor). Changes: CPU only, per-neuron spike readout for the
UI raster, and per-group spike counts accumulated per call instead of a ring buffer.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


@dataclass
class LIFParams:
    dt: float = 1.0          # biological ms per step
    tau: float = 20.0        # membrane time constant, ms
    thr: float = 1.0         # spike threshold (dimensionless v)
    reset: float = 0.0
    refr_ms: int = 2         # absolute refractory period, steps
    gain: float = 0.005      # synaptic gain multiplying signed synapse counts
    governor_frac: float = 0.05   # activity fraction that counts as runaway
    governor_steps: int = 20      # consecutive runaway steps before damping


class BatchedLIF:
    """B independent brains sharing one signed edge list. Inputs are currents on named groups."""

    def __init__(self, N, pre, post, weight, groups, B=1, params=None, watch=None):
        self.p = params or LIFParams()
        self.N, self.B = int(N), int(B)
        pre = np.asarray(pre, dtype=np.int64); post = np.asarray(post, dtype=np.int64)
        weight = np.asarray(weight, dtype=np.float32)
        order = np.argsort(pre, kind="stable")
        counts = np.bincount(pre, minlength=self.N)
        self.indptr = torch.as_tensor(np.concatenate([[0], np.cumsum(counts)]), dtype=torch.long)
        self.post_csr = torch.as_tensor(post[order], dtype=torch.long)
        self.wt_csr = torch.as_tensor(weight[order], dtype=torch.float32)
        self.E = int(len(pre))
        self.decay = float(math.exp(-self.p.dt / self.p.tau))
        self.groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items() if len(v) > 0}
        self.names = sorted(self.groups)
        self.index = {n: i for i, n in enumerate(self.names)}
        M = np.zeros((len(self.names), self.N), dtype=np.float32)
        for i, n in enumerate(self.names):
            M[i, self.groups[n]] = 1.0
        self.M = torch.as_tensor(M)                      # (G, N) membership, for drive and readout
        self.sizes = M.sum(axis=1).astype(np.float64)
        self.watch = torch.as_tensor(np.asarray(watch if watch is not None else [], dtype=np.int64))
        self.amp = torch.zeros(self.B, len(self.names))
        self.reset_all()

    def reset_all(self):
        self.v = torch.zeros(self.B, self.N)
        self.spk = torch.zeros(self.B, self.N)
        self.refr = torch.zeros(self.B, self.N)
        self.run = torch.zeros(self.B)
        self.t_ms = 0
        self.runaway_total = 0

    def set_amps(self, amps):
        """amps: list (len B) of {group: current}; unknown names are ignored."""
        a = np.zeros((self.B, len(self.names)), dtype=np.float32)
        for b, d in enumerate(amps):
            for name, val in d.items():
                i = self.index.get(name)
                if i is not None:
                    a[b, i] += val
        self.amp = torch.as_tensor(a)

    def step(self, n_steps):
        """Advance every brain n_steps ms. Returns (group spike counts (B,G), watched raster (B,W,n_steps))."""
        drive = self.amp @ self.M
        p, decay, gain, thr = self.p, self.decay, self.p.gain, self.p.thr
        inp = torch.zeros(self.B, self.N)
        flat = inp.view(-1)
        counts = torch.zeros(self.B, self.N)
        W = len(self.watch)
        raster = torch.zeros(self.B, W, int(n_steps))
        for s in range(int(n_steps)):
            inp.zero_()
            self._propagate(flat)
            v = decay * self.v + gain * inp + drive
            v = v * (self.refr <= 0)
            spk = (v >= thr).float()
            v = v * (1.0 - spk) + p.reset * spk
            self.refr = torch.clamp(self.refr - 1.0, min=0.0) + spk * p.refr_ms
            frac = spk.mean(dim=1)
            self.run = torch.where(frac > p.governor_frac, self.run + 1.0, torch.zeros_like(self.run))
            damp = (self.run >= p.governor_steps).float()
            self.runaway_total += int(damp.sum().item()) if damp.any() else 0
            self.v, self.spk = v * (1.0 - 0.5 * damp)[:, None], spk
            counts += spk
            if W:
                raster[:, :, s] = spk[:, self.watch]
            self.t_ms += p.dt
        return (counts @ self.M.T).numpy(), raster.numpy()

    def _propagate(self, flat):
        active = torch.nonzero(self.spk.view(-1)).squeeze(1)
        if active.numel() == 0:
            return
        neuron = active % self.N
        starts = self.indptr[neuron]
        cnt = self.indptr[neuron + 1] - starts
        total = int(cnt.sum())
        if total == 0:
            return
        excl = torch.cumsum(cnt, 0) - cnt
        eidx = torch.repeat_interleave(starts - excl, cnt, output_size=total) + torch.arange(total)
        boff = torch.repeat_interleave((active // self.N) * self.N, cnt, output_size=total)
        flat.index_add_(0, self.post_csr[eidx] + boff, self.wt_csr[eidx])

    def rates_hz(self, counts, window_ms):
        """Per-neuron mean rate per group (B, G) from group spike counts over window_ms."""
        return counts / (self.sizes[None, :] * (window_ms / 1000.0))
