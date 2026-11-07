"""Compound eye: ray-cast the arena along the measured fly eye directions.

Directions: 1,709 microCT ommatidial axes (857 left, 852 right) from reiserlab/eyemap_T4, the
same pinned file FlyVision imports (see backend/scripts/prepare_eye.py). Falls back to a
700-direction Fibonacci lattice only if that file is missing, and says so in `source`.
Body frame: +x forward, +y left, +z up.
"""
import base64
import io
from pathlib import Path

import numpy as np

from . import arena

DATA = Path(__file__).resolve().parent / "data/eye_directions.npz"
FAR = 60.0
SKY = np.array([0.62, 0.76, 0.96])
GRASS = np.array([0.36, 0.55, 0.27])
GRASS2 = np.array([0.30, 0.47, 0.22])
POND_COL = np.array([0.16, 0.30, 0.52])
PAD = np.array([0.78, 0.78, 0.74])


def fibonacci_dirs(n=700):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def load_dirs():
    if DATA.exists():
        z = np.load(DATA, allow_pickle=False)
        return z["rays"].astype(float), z["left"].astype(bool), "microCT reference eye (reiserlab/eyemap_T4, 1,709 directions)"
    d = fibonacci_dirs()
    return d, d[:, 1] > 0, "Fibonacci lattice (700 directions; reference eye file missing)"


class CompoundEye:
    def __init__(self):
        self.dirs, self.left, self.source = load_dirs()
        self.n = len(self.dirs)
        self.az = np.arctan2(self.dirs[:, 1], self.dirs[:, 0])            # + = left
        self.el = np.arcsin(np.clip(self.dirs[:, 2], -1, 1))
        self.P = arena.prim_arrays()
        # equirectangular lookup: nearest ommatidium for each pixel of a 128 x 64 image
        W, H = 128, 64
        az = np.linspace(np.pi, -np.pi, W, endpoint=False) - np.pi / W     # left edge = behind-left
        el = np.linspace(np.pi / 2, -np.pi / 2, H, endpoint=False) - np.pi / (2 * H)
        A, E = np.meshgrid(az, el)
        px = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        best = np.argmax(px @ self.dirs.T, axis=1)
        cover = np.max(px @ self.dirs.T, axis=1) > np.cos(np.radians(8))    # outside the eye's field: black
        self.eq_idx, self.eq_cover, self.eq_shape = best, cover, (H, W)

    def cast(self, pos, yaw):
        """Return dict with rgb (n,3), lum (n,), depth (n,), lm (n,) landmark index or -1."""
        c, s = np.cos(yaw), np.sin(yaw)
        d = self.dirs
        D = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], 1)
        o = np.asarray(pos, float)
        n = self.n
        t = np.full(n, FAR)
        col = SKY[None, :] * (0.8 + 0.2 * np.clip(D[:, 2], 0, 1))[:, None]
        lm = np.full(n, -1)
        # ground
        down = D[:, 2] < -1e-6
        tg = np.where(down, -o[2] / np.where(down, D[:, 2], -1), FAR)
        hit = down & (tg < t)
        if hit.any():
            gx = o[0] + tg * D[:, 0]; gy = o[1] + tg * D[:, 1]
            checker = ((np.floor(gx / 2) + np.floor(gy / 2)) % 2 == 0)
            gcol = np.where(checker[:, None], GRASS, GRASS2)
            pond = (gx - arena.POND["center"][0]) ** 2 + (gy - arena.POND["center"][1]) ** 2 <= arena.POND["radius"] ** 2
            pad = (gx - arena.HOME[0]) ** 2 + (gy - arena.HOME[1]) ** 2 <= arena.HOME_RADIUS ** 2
            gcol = np.where(pond[:, None], POND_COL, gcol)
            gcol = np.where(pad[:, None], PAD, gcol)
            fade = np.exp(-tg / 45.0)[:, None]
            gcol = gcol * fade + SKY * 0.85 * (1 - fade)
            col = np.where(hit[:, None], gcol, col); t = np.where(hit, tg, t)
        # vertical cylinders (side surface only)
        P = self.P
        for k in range(len(P["cyl_c"])):
            cx, cy, z0 = P["cyl_c"][k]; r, h = P["cyl_s"][k]
            ox, oy = o[0] - cx, o[1] - cy
            a = D[:, 0] ** 2 + D[:, 1] ** 2
            b = 2 * (ox * D[:, 0] + oy * D[:, 1])
            cc = ox * ox + oy * oy - r * r
            disc = b * b - 4 * a * cc
            ok = (disc >= 0) & (a > 1e-9)
            sq = np.sqrt(np.where(ok, disc, 0))
            t0 = (-b - sq) / np.where(ok, 2 * a, 1)
            z = o[2] + t0 * D[:, 2]
            hit = ok & (t0 > 1e-3) & (t0 < t) & (z >= z0) & (z <= z0 + h)
            if hit.any():
                nx = (ox + t0 * D[:, 0]) / r; ny = (oy + t0 * D[:, 1]) / r
                shade = 0.65 + 0.35 * np.clip(nx * 0.5 + ny * 0.7, -1, 1) * 0.5 + 0.175
                col = np.where(hit[:, None], P["cyl_col"][k] * shade[:, None], col)
                t = np.where(hit, t0, t); lm = np.where(hit, P["cyl_lm"][k], lm)
        # boxes (slab method)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / np.where(np.abs(D) < 1e-9, 1e-9, D)
        for k in range(len(P["box_c"])):
            lo = P["box_c"][k] - P["box_s"][k]; hi = P["box_c"][k] + P["box_s"][k]
            t1 = (lo - o) * inv; t2 = (hi - o) * inv
            tmin = np.max(np.minimum(t1, t2), axis=1); tmax = np.min(np.maximum(t1, t2), axis=1)
            hit = (tmax >= tmin) & (tmin > 1e-3) & (tmin < t)
            if hit.any():
                face = np.argmax(np.minimum(t1, t2), axis=1)
                shade = np.array([0.85, 0.7, 1.0])[face]
                col = np.where(hit[:, None], P["box_col"][k] * shade[:, None], col)
                t = np.where(hit, tmin, t); lm = np.where(hit, P["box_lm"][k], lm)
        lum = col @ np.array([0.299, 0.587, 0.114])
        return {"rgb": np.clip(col, 0, 1), "lum": lum, "depth": t, "lm": lm}

    def stripe_mask(self, yaw, bearing_world, half_width_deg=7.0):
        """Ommatidia inside the vertical target stripe at world bearing (elevation -30..+45 deg)."""
        rel = (bearing_world - yaw + np.pi) % (2 * np.pi) - np.pi
        daz = np.abs((self.az - rel + np.pi) % (2 * np.pi) - np.pi)
        return (daz <= np.radians(half_width_deg)) & (self.el > np.radians(-30)) & (self.el < np.radians(45))

    def equirect(self, rgb):
        H, W = self.eq_shape
        img = rgb[self.eq_idx] * self.eq_cover[:, None]
        return (img.reshape(H, W, 3) * 255).astype(np.uint8)

    def equirect_png(self, rgb, fmt="PNG", scale=4):
        """128 x 64 panorama of the eye's samples, upscaled (nearest) so vision models see blocks, not blur."""
        from PIL import Image
        buf = io.BytesIO()
        img = Image.fromarray(self.equirect(rgb))
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        img.save(buf, format=fmt)
        return buf.getvalue()

    def mosaic_bytes(self, rgb):
        """Compact per-ommatidium colours for the UI's hexagonal mosaic: base64 of n*3 uint8."""
        return base64.b64encode((np.clip(rgb, 0, 1) * 255).astype(np.uint8).tobytes()).decode()

    def layout(self):
        """Azimuth/elevation (degrees) per ommatidium and eye side, sent once to the UI."""
        return {"az": np.degrees(self.az).round(2).tolist(), "el": np.degrees(self.el).round(2).tolist(),
                "left": self.left.astype(int).tolist(), "source": self.source}


class EyeSignals:
    """Per-drone temporal state: motion (temporal contrast) and looming, split by eye."""

    def __init__(self, eye: CompoundEye):
        self.eye = eye
        self.prev_lum = None
        self.prev_near = None

    def update(self, view, dt, near_m=7.0):
        e = self.eye
        lum = view["lum"]
        motion = np.zeros_like(lum) if self.prev_lum is None else np.abs(lum - self.prev_lum) / max(dt, 1e-3)
        self.prev_lum = lum
        # looming: angular coverage of near landmark surfaces per eye, and its rate of expansion
        near = (view["lm"] >= 0) & (view["depth"] < near_m)
        # weight closer surfaces more (dark object filling the view)
        w = np.where(near, 1.0 - view["depth"] / near_m, 0.0)
        cov_l = float(w[e.left].sum() / e.left.sum()); cov_r = float(w[~e.left].sum() / (~e.left).sum())
        if self.prev_near is None:
            exp_l = exp_r = 0.0
        else:
            exp_l = max(0.0, (cov_l - self.prev_near[0]) / dt); exp_r = max(0.0, (cov_r - self.prev_near[1]) / dt)
        self.prev_near = (cov_l, cov_r)
        return {"motion": motion, "motion_l": float(motion[e.left].mean()), "motion_r": float(motion[~e.left].mean()),
                "cover_l": cov_l, "cover_r": cov_r, "expand_l": exp_l, "expand_r": exp_r}
