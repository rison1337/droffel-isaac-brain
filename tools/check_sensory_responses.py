"""Compare each probe to a seed-matched baseline in the complete Traced graph."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from droffel_sim import FlyNervousSystem

records = []
for name in ("control", "odor_a", "odor_b", "touch", "vibration", "heat", "light_L", "light_R"):
    fly = FlyNervousSystem(gpu="off")
    for _ in range(500):
        fly.step()
    baseline_spikes = fly.brain.spikes.copy()
    fly.sense({name: 0.65} if name != "control" else {})
    counts = np.zeros(fly.brain.n, dtype=np.int32)
    for _ in range(500):
        fly.step()
        counts += fly.brain.spikes
    rates = {key: float(counts[idx].mean()) for key, idx in fly.groups.items() if len(idx)}
    driven = fly.groups.get(name, np.array([], dtype=np.int64))
    indirect = np.ones(fly.brain.n, dtype=bool)
    indirect[driven] = False
    record = {"condition": name, "mean_rates_hz": rates, "driven_count": len(driven),
              "indirect_spikes": int(counts[indirect].sum()), "state": fly.state()}
    records.append(record)
    print(json.dumps({k:v for k,v in record.items() if k != "state"}), flush=True)
baseline = records[0]["mean_rates_hz"]
for record in records[1:]:
    name = record["condition"]
    assert record["driven_count"] > 0
    assert record["mean_rates_hz"][name] > baseline[name]+10, name
assert not np.intersect1d(fly.groups["odor_a"], fly.groups["odor_b"]).size
assert not np.intersect1d(fly.groups["light_L"], fly.groups["light_R"]).size
(ROOT / "data/sensory_responses.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print("PASS: all seven sensory probes respond; odor and eye populations are distinct", flush=True)
