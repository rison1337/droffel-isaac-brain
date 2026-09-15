"""Build the compact graph artifact for the LIF sim from the MaleCNS v1.0 flat connectome.

The flat export contains fragment-level contacts for every segmentation piece; the
curated connectome is the subset where both endpoints are bodies with status 'Traced'
in the annotations (165,122 neurons — the published count).

Input : D:\\fly-data\\connectome-weights-male-cns-v1.0-minconf-0.5.feather
        D:\\fly-data\\body-neurotransmitters-male-cns-v1.0.feather
        D:\\fly-data\\body-annotations-male-cns-v1.0-minconf-0.5.feather
Output: data\\graph_malecns_v1.npz            CSR adjacency + ids + neurotransmitter codes
        data\\annotations_malecns_v1.parquet   id -> type/class/superclass/somaSide
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scipy.sparse as sp

DATA_DIR = Path(os.environ.get("FLY_DATA_DIR", r"D:\fly-data"))
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

t0 = time.time()

print("loading annotations, keeping status == 'Traced' ...")
ann = feather.read_table(DATA_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather").to_pandas()
ann_traced = ann.loc[ann["status"] == "Traced"].copy()
valid = np.sort(ann_traced["bodyId"].to_numpy())
n = len(valid)
print(f"traced neurons: {n:,} ({time.time() - t0:.0f}s)")

print("reading weights feather ...")
tab = feather.read_table(DATA_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
df = tab.select(["body_pre", "body_post", "weight"]).to_pandas()
print(f"rows: {len(df):,} ({time.time() - t0:.0f}s)")
del tab

print("filtering to traced endpoints ...")
mask = np.isin(df["body_pre"].to_numpy(), valid, assume_unique=False) & np.isin(
    df["body_post"].to_numpy(), valid, assume_unique=False
)
df = df.loc[mask]
print(f"curated connections: {len(df):,} ({time.time() - t0:.0f}s)")

pre_idx = np.searchsorted(valid, df["body_pre"].to_numpy())
post_idx = np.searchsorted(valid, df["body_post"].to_numpy())
graph = sp.csr_matrix(
    (df["weight"].to_numpy(dtype=np.float32), (pre_idx, post_idx)), shape=(n, n)
)
graph.sum_duplicates()
graph = graph.tocsr()
print(f"neurons: {n:,}   edges (nnz): {graph.nnz:,} ({time.time() - t0:.0f}s)")
del df, mask, pre_idx, post_idx

print("attaching neurotransmitter consensus ...")
nt = feather.read_table(DATA_DIR / "body-neurotransmitters-male-cns-v1.0.feather").to_pandas()
nt = nt.dropna(subset=["consensus_nt"]).drop_duplicates("body")
labels = sorted(nt["consensus_nt"].unique())
lab_code = {name: code for code, name in enumerate(labels)}
nt_arr = np.full(n, -1, dtype=np.int8)
body_pos = np.searchsorted(valid, nt["body"].to_numpy())
found = (body_pos < n) & (valid[np.clip(body_pos, 0, n - 1)] == nt["body"].to_numpy())
nt_arr[body_pos[found]] = [lab_code[x] for x in nt.loc[found, "consensus_nt"]]
print(f"nt labels attached for {int(found.sum()):,} / {n:,} neurons: {labels}")

print("saving graph npz ...")
OUT_DIR.mkdir(exist_ok=True)
np.savez_compressed(
    OUT_DIR / "graph_malecns_v1.npz",
    ids=valid,
    indptr=graph.indptr.astype(np.int64),
    indices=graph.indices.astype(np.int32),
    data=graph.data,
    nt=nt_arr,
    nt_labels=np.array(labels),
)
print(f"graph saved ({time.time() - t0:.0f}s)")

print("saving annotations parquet ...")
cols = ["bodyId", "type", "class", "superclass", "subclass", "somaSide", "rootSide", "group", "status"]
ann_traced[cols].to_parquet(OUT_DIR / "annotations_malecns_v1.parquet", index=False)
print("superclass counts (top 20):")
print(ann_traced["superclass"].value_counts().head(20).to_string())
print(f"done in {time.time() - t0:.0f}s")
