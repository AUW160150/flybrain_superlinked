"""Fetch the pinned microCT eye directions (reiserlab/eyemap_T4) and save them as a small npz.

Same source, pin and checks as flybrain/projects/flyvision/flyvision/eye_reference.py:
git-blob hash of the file, unit-vector norms, head basis (+x forward, +y left, +z up),
and lens/cone eye assignments. Output: backend/sim/data/eye_directions.npz
"""
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/reiserlab/eyemap_T4/99d2a43123db636cedb55af9ff31a59657e7d17e/data/microCT/20240701.RData"
BLOB_SHA1 = "78c05b5aac401cfd3d53893a98dd926aac637988"
NBYTES = 310361
OUT = Path(__file__).resolve().parents[1] / "sim/data/eye_directions.npz"


def main():
    import rdata
    data = urllib.request.urlopen(URL, timeout=60).read()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if len(data) != NBYTES or blob != BLOB_SHA1:
        sys.exit(f"pinned reference differs: bytes={len(data)} blob={blob}")
    with tempfile.NamedTemporaryFile(suffix=".RData") as f:
        f.write(data); f.flush()
        eye = rdata.read_rda(f.name)
    rays = np.asarray(eye["ucl_rot_sm"], dtype=float)
    left = np.asarray(eye["ind_left_cone"], dtype=bool)
    if rays.ndim != 2 or rays.shape[1] != 3 or left.shape != (len(rays),):
        sys.exit("unexpected shapes")
    if np.max(np.abs(np.linalg.norm(rays, axis=1) - 1)) > 1e-9:
        sys.exit("directions are not unit vectors")
    for key, expected in [("vf", [1, 0, 0]), ("vleft", [0, 1, 0]), ("vn", [0, 0, 1])]:
        if not np.allclose(eye[key], expected, atol=1e-12):
            sys.exit(f"unexpected head basis {key}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, rays=rays.astype(np.float32), left=left, source=np.array(URL), blob_sha1=np.array(BLOB_SHA1))
    print(f"saved {len(rays)} directions ({left.sum()} left, {(~left).sum()} right) -> {OUT}")


if __name__ == "__main__":
    main()
