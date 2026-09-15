"""Fetch selected real neuron arbors in the same MaleCNS coordinate space."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "vendor/virtual-fly-lab/data"
CACHE = ROOT / "data/morphology"
OUT = ROOT / "terrarium/data"
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc"
with np.load(DATA / "graph_malecns_v1.npz") as g:
    ids = g["ids"]
ann = pd.read_parquet(DATA / "annotations_malecns_v1.parquet").set_index("bodyId").reindex(ids)
selected = []
for cell_type, amount in [("DNp01",2),("LC4",4),("LPLC2",4),("DNa01",2),("DNg07",2),
                          ("ORN_DM1",2),("ORN_DA1",2),("MN9",2)]:
    candidates = np.flatnonzero((ann["type"].fillna("")==cell_type).to_numpy())
    # Deterministic spread in ID order; not a claim of representative biology.
    if len(candidates): selected.extend(candidates[np.linspace(0,len(candidates)-1,min(amount,len(candidates)),dtype=int)])
meta = json.loads((DATA / "soma_positions_meta.json").read_text())
CACHE.mkdir(exist_ok=True)

def fetch(idx):
    body_id = int(ids[idx])
    path = CACHE / f"{body_id}.swc"
    url = f"{BASE}/{body_id}.swc"
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        if not payload or payload.startswith(b"<"): raise ValueError(f"Not SWC: {url}")
        path.write_bytes(payload)
    payload = path.read_bytes()
    table = np.loadtxt(io.BytesIO(payload), ndmin=2)
    node_ids = table[:,0].astype(np.int64)
    parents = table[:,6].astype(np.int64)
    order = np.argsort(node_ids)
    lookup = np.searchsorted(node_ids[order],parents)
    found = (lookup<len(node_ids)) & (parents>=0)
    found &= node_ids[order[np.clip(lookup,0,len(node_ids)-1)]] == parents
    child = np.flatnonzero(found)
    parent = order[lookup[found]]
    p = table[:,[2,4,3]].astype(np.float32)
    p[:,1] *= -1
    p -= np.asarray(meta["center"],dtype=np.float32)
    p *= meta["scale"]
    segments = np.stack([p[child],p[parent]],axis=1)
    description = {"index":int(idx),"body_id":str(body_id),"type":str(ann.iloc[idx]["type"]),
                   "segments":len(segments),"url":url,"sha256":hashlib.sha256(payload).hexdigest()}
    print(f"{body_id} {description['type']}: {len(segments)} segments", flush=True)
    return description,segments

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(fetch,selected))
records = []
with (OUT / "neuron_arbors_f32.bin").open("wb") as stream:
    for record,segments in results:
        record["byte_offset"] = stream.tell()
        segments.astype("<f4").tofile(stream)
        records.append(record)
(OUT / "neuron_arbors.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
print(f"MORPHOLOGY READY: {len(records)} neurons, {sum(r['segments'] for r in records)} segments",flush=True)
