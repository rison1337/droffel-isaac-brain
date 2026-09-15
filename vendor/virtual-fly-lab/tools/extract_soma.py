"""Extract soma positions for the whole-CNS point cloud ("x-ray" view).

Soma locations live in the annotations feather (bodyId -> [x, y, z] in raw EM
voxels). This maps them onto graph row indices, finds which axis runs
brain->VNC, normalizes into Godot world space (brain up), and writes:
  data/soma_positions.npz           ids + positions (float32, Godot scale)
  data/soma_positions_f32.bin       raw float32 xyz triplets for Godot
"""
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FLY_DATA_DIR", Path(__file__).resolve().parents[3] / "data" / "raw"))

ann = feather.read_table(
    DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    columns=["bodyId", "somaLocation", "superclass"],
).to_pandas()
ann = ann.dropna(subset=["somaLocation"])
ann = ann[ann["somaLocation"].map(lambda v: v is not None and len(v) == 3)]

z = np.load(ROOT / "data" / "graph_malecns_v1.npz")
ids = z["ids"]
pos_map = dict(zip(ann["bodyId"].astype(np.int64), ann["somaLocation"]))
P = np.full((len(ids), 3), np.nan, dtype=np.float64)
missing = 0
for i, b in enumerate(ids):
    v = pos_map.get(int(b))
    if v is None:
        missing += 1
    else:
        P[i] = v
print(f"somas found: {len(ids) - missing:,} / {len(ids):,} (missing {missing:,})")

valid = ~np.isnan(P[:, 0])
for ax, name in enumerate("xyz"):
    v = P[valid, ax]
    print(f"{name}: p1={np.percentile(v, 1):.0f} p50={np.percentile(v, 50):.0f} "
          f"p99={np.percentile(v, 99):.0f} extent={v.max() - v.min():.0f}")

# which axis separates brain from VNC? correlate position with VNC membership
is_vnc = ann.set_index("bodyId")["superclass"].fillna("").str.startswith("vnc")
vnc_flag = np.array([1.0 if is_vnc.get(int(b), False) else 0.0 for b in ids])
vnc_flag[~valid] = np.nan
print("\naxis vs VNC membership (|corr|):")
corrs = {}
for ax, name in enumerate("xyz"):
    m = valid & ~np.isnan(vnc_flag)
    c = np.corrcoef(P[m, ax], vnc_flag[m])[0, 1]
    corrs[name] = abs(c)
    print(f"  {name}: {c:+.3f}")
ap_axis = "xyz".index(max(corrs, key=corrs.get))
print(f"AP axis (brain->VNC): {'xyz'[ap_axis]}")

# orientation: brain (low VNC fraction) should be at +Y (up in Godot)
if np.corrcoef(P[valid, ap_axis], vnc_flag[valid])[0, 1] > 0:
    ap_sign = -1.0  # VNC at high coords -> flip so brain is up
else:
    ap_sign = 1.0

# bilateral axis: remaining axis with highest L/R split via somaSide? use PCA-free
# heuristic: among the two non-AP axes, the one with the larger extent is LR
other = [a for a in range(3) if a != ap_axis]
ext = {a: np.ptp(P[valid, a]) for a in other}
lr_axis = max(ext, key=ext.get)
third_axis = [a for a in other if a != lr_axis][0]
print(f"LR axis: {'xyz'[lr_axis]}, third axis: {'xyz'[third_axis]}")

Q = np.zeros((len(ids), 3), dtype=np.float32)
Q[:, 0] = P[:, lr_axis]          # X = left/right
Q[:, 1] = ap_sign * P[:, ap_axis]  # Y = up (brain up)
Q[:, 2] = P[:, third_axis]        # Z = depth
center = np.nanmedian(Q[valid], axis=0)
Q -= center
scale = 30.0 / np.nanmax(np.ptp(Q[valid], axis=0))
Q *= scale
Q[~valid] = 0.0
hidden = (~valid).astype(np.uint8)

print(f"\nGodot-space extents: {np.ptp(Q[valid], axis=0).round(1)} (x=LR, y=UD, z=depth)")
out = ROOT / "data"
np.savez_compressed(out / "soma_positions.npz", ids=ids, pos=Q, hidden=hidden)
Q.tofile(out / "soma_positions_f32.bin")
(out / "soma_positions_meta.json").write_text(json.dumps({
    "count": int(len(ids)),
    "valid": int(valid.sum()),
    "scale": float(scale),
    "center": [float(c) for c in center],
}))
print("saved data/soma_positions.{npz,_f32.bin,_meta.json}")
