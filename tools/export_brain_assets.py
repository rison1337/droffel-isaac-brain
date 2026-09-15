"""Export measured soma coordinates and row-aligned group metadata for Godot."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_groups import make_groups

source = ROOT / "vendor/virtual-fly-lab/data"
out = ROOT / "terrarium/data"
out.mkdir(parents=True, exist_ok=True)
with np.load(source / "graph_malecns_v1.npz") as graph:
    ids = graph["ids"]
ann = pd.read_parquet(source / "annotations_malecns_v1.parquet").set_index("bodyId").reindex(ids)
if "rootSide" not in ann:
    extra = feather.read_table(ROOT / "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                               columns=["bodyId", "rootSide"]).to_pandas().set_index("bodyId")
    ann = ann.join(extra)
    ann.reset_index().to_parquet(source / "annotations_malecns_v1.parquet", index=False)
groups = make_groups(ann)
shutil.copyfile(source / "soma_positions_f32.bin", out / "soma_positions_f32.bin")
positions = np.fromfile(out / "soma_positions_f32.bin", dtype=np.float32).reshape(-1,3)
valid = np.any(positions != 0, axis=1)
vnc = ann["superclass"].fillna("").str.startswith("vnc").to_numpy()
vnc.astype(np.uint8).tofile(out / "soma_is_vnc.bin")
brain_pos = positions[valid & ~vnc]
metadata = {"count": len(ids), "visible_somas": int(valid.sum()), "missing_somas": int((~valid).sum()),
            "brain_center": np.median(brain_pos, axis=0).tolist(),
            "groups": {name: idx.tolist() for name,idx in groups.items()},
            "gf": [{"index":int(i),"body_id":str(ids[i]),"type":"DNp01","position":positions[i].tolist()} for i in groups["GF"]]}
(out / "brain_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
print(json.dumps({"visible_somas":metadata["visible_somas"], "groups":{k:len(v) for k,v in groups.items()}}))
