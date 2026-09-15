"""MaleCNS neural service for the terrarium and future game adapters.

This is an approximate LIF model. Flight mechanics and action decoding are
engineered. A scare is sensory stimulation of LC4/LPLC2, never a forced GF spike.
"""
import argparse
from collections import deque
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "virtual-fly-lab"
sys.path.insert(0, str(VENDOR))
from sim.brain import Brain
from sim.server import SimServer
from neural_groups import make_groups, INPUT_HZ


class FlyNervousSystem:
    def __init__(self, gpu="auto", seed=7, norm=5.0, profile="interactive"):
        self.profile = profile
        if profile == "research":
            from research_brain import ResearchBrain
            self.brain = ResearchBrain(VENDOR / "data/graph_malecns_v1.npz", gpu_mode=gpu, seed=seed)
        else:
            self.brain = Brain(VENDOR / "data/graph_malecns_v1.npz", gpu_mode=gpu,
                               seed=seed, norm_const=norm, background_hz=0.3)
        ann = pd.read_parquet(VENDOR / "data/annotations_malecns_v1.parquet").set_index("bodyId")
        ann = ann.reindex(self.brain.ids)
        self.groups = make_groups(ann)
        self.arbor_indices = np.array([],dtype=np.int64)
        arbor_metadata = ROOT / "terrarium/data/neuron_arbors.json"
        if arbor_metadata.exists():
            self.arbor_indices = np.array([r["index"] for r in json.loads(arbor_metadata.read_text())],dtype=np.int64)
        for required in ("loom_L", "loom_R", "GF"):
            if self.groups[required].size == 0:
                raise RuntimeError(f"Required annotated circuit missing: {required}")
        self.rng = np.random.default_rng(seed + 1)
        self.sensory = {name: 0.0 for name in INPUT_HZ}
        self.elapsed = 0.0
        self.escape = 0.0
        self.escape_turn = 0.0
        self.escape_count = 0
        self.gf_spikes = 0
        self._last_escape = -10.0
        self.turn = 0.0
        self.gf_history = deque()
        self.gf_burst = 0

    def sense(self, values):
        for name in self.sensory:
            value = float(values.get(name, 0.0))
            self.sensory[name] = float(np.clip(value, 0, 1)) if math.isfinite(value) else 0.0
        if self.profile == "research":
            active = [self.groups[name] for name,value in self.sensory.items() if value>0]
            idx = np.unique(np.concatenate(active)) if active else np.array([],dtype=np.int64)
            self.brain.set_sensory_indices(idx)

    def step(self):
        parts = []
        for name, value in self.sensory.items():
            idx = self.groups[name]
            hz = INPUT_HZ[name]
            probability = -math.expm1(-hz * value * self.brain.dt_s)
            if self.profile == "research":
                probability = min(1.,hz*value*self.brain.dt_s)
            if value and idx.size:
                parts.append(idx[self.rng.random(idx.size) < probability])
        injected = np.concatenate(parts) if parts else None
        self.brain.step(injected)
        self.elapsed += self.brain.dt_s
        gf_count = int(self.brain.spikes[self.groups["GF"]].sum())
        self.gf_spikes += gf_count
        self.gf_history.append(gf_count)
        self.gf_burst += gf_count
        if len(self.gf_history) > round(.04/self.brain.dt_s):
            self.gf_burst -= self.gf_history.popleft()
        looming = max(self.sensory["loom_L"], self.sensory["loom_R"])
        # The downstream GF must actually fire. Input only gates contextual decoding.
        if gf_count and self.gf_burst >= 3 and looming > 0.05 and self.elapsed - self._last_escape > 0.5:
            self.escape = 1.0
            self.escape_turn = 1.0 if self.sensory["loom_L"] >= self.sensory["loom_R"] else -1.0
            self.escape_count += 1
            self._last_escape = self.elapsed
        self.escape *= math.exp(-self.brain.dt_s / 0.45)
        alpha = 1.0-math.exp(-self.brain.dt_s/.05)
        self.turn = (1-alpha) * self.turn + alpha * np.tanh((self.rate("steer_R") - self.rate("steer_L")) / 10.0)

    def rate(self, name):
        idx = self.groups[name]
        return float(self.brain.rate_ema[idx].mean()) if idx.size else 0.0

    def state(self, cloud=False):
        b = self.brain
        result = {"neurons": b.n, "edges": b.w.nnz, "kernel": b.kernel_name,
                  "profile": self.profile, "dt_ms": b.dt_ms,
                  "sim_time": round(self.elapsed, 3), "active": float(b.spikes.mean()) * 100,
                  "escape": self.escape, "escape_turn": self.escape_turn,
                  "escape_count": self.escape_count, "gf_spikes": self.gf_spikes,
                  "turn": float(self.turn), "flight": min(1., self.rate("flight") / 30.),
                  "rates": {name: round(self.rate(name), 2) for name in self.groups},
                  "group_sizes": {name: int(idx.size) for name, idx in self.groups.items()},
                  "sensory": self.sensory.copy(), "mean_hz": float(b.rate_ema.mean()),
                  "responding": int(np.count_nonzero(b.rate_ema > 2.0)),
                  "arbor_rates": [[int(i),round(float(b.rate_ema[i]),2)] for i in self.arbor_indices]}
        if cloud:
            hot = np.flatnonzero(b.rate_ema > 2.0)
            if hot.size > 6000:
                hot = hot[np.argpartition(b.rate_ema[hot], -6000)[-6000:]]
            # Keep the named output circuit visible even during broad input.
            hot = np.union1d(hot, self.groups["GF"][b.rate_ema[self.groups["GF"]] > 2.0])
            result["cloud"] = [[int(i), round(float(b.rate_ema[i]), 1)] for i in hot]
        return result


def benchmark(args):
    records = []
    for name, gain in [("control", 0.0), ("loom", 1.0), ("loom_GF_inputs_blocked", 1.0)]:
        fly = FlyNervousSystem(args.gpu, norm=args.norm, profile=args.profile)
        if name.endswith("blocked"):
            # Causal control: remove incoming chemical connections to GF, while
            # keeping identical stimulation, background noise and all neurons.
            b = fly.brain
            b.w.data[np.isin(b.w.indices, fly.groups["GF"])] = 0
            b.w.eliminate_zeros()
            if b._gpu is not None:
                from sim.gpu import GpuProp
                b._gpu = GpuProp(b.w)
        t0 = time.perf_counter()
        for _ in range(round(1/fly.brain.dt_s)):
            fly.step()
        baseline = fly.gf_spikes
        fly.sense({"loom_L": gain, "loom_R": gain})
        for _ in range(round(1/fly.brain.dt_s)):
            fly.step()
        state = fly.state()
        state["condition"] = name
        state["stimulus_gf_spikes"] = fly.gf_spikes - baseline
        state["wall_seconds"] = time.perf_counter() - t0
        records.append(state)
        print(json.dumps(state), flush=True)
    output = ROOT / ("data/research_benchmark.json" if args.profile=="research" else "data/brain_benchmark.json")
    output.write_text(json.dumps(records, indent=2) + "\n")
    assert records[1]["stimulus_gf_spikes"] > records[0]["stimulus_gf_spikes"] + 10
    assert records[1]["escape_count"] > 0
    assert records[2]["escape_count"] == 0
    print("PASS: looming drives GF; blocking GF inputs removes decoded escape", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--norm", type=float, default=5.)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument("--profile", choices=["interactive","research"], default="interactive")
    ap.add_argument("--port-file", type=Path, default=ROOT / "terrarium/data/sim_port.txt")
    ap.add_argument("--status-file", type=Path, default=ROOT / "data/live_status.json")
    args = ap.parse_args()
    if args.benchmark:
        return benchmark(args)
    fly = FlyNervousSystem(args.gpu, norm=args.norm, profile=args.profile)
    server = SimServer(port=args.port, port_file=args.port_file)
    print("READY " + json.dumps(fly.state()), flush=True)
    paused = False
    running = True
    last_sense = time.perf_counter()
    next_step = last_report = time.perf_counter()
    steps = 0
    latest_pose = {}
    while running:
        now = time.perf_counter()
        while not server.stimuli_q.empty():
            cmd = server.stimuli_q.get()
            if not isinstance(cmd, dict):
                continue
            try:
                if cmd.get("cmd") == "sense":
                    fly.sense(cmd)
                    last_sense = now
                elif cmd.get("cmd") == "pause":
                    paused = bool(cmd.get("value", False))
                    next_step = now
                elif cmd.get("cmd") == "pose":
                    latest_pose = {key: cmd[key] for key in ("position", "mode", "threat", "fps") if key in cmd}
                elif cmd.get("cmd") == "profile" and cmd.get("value") in ("interactive","research"):
                    if cmd["value"] != fly.profile:
                        pending = fly.state()
                        pending.update(loading=True,paused=True)
                        server.set_state(pending)
                        fly = FlyNervousSystem(args.gpu,norm=args.norm,profile=cmd["value"])
                        now = next_step = last_report = time.perf_counter()
                        steps = 0
                        last_sense = now
                elif cmd.get("cmd") == "quit":
                    running = False
            except (TypeError, ValueError):
                continue
        if now - last_sense > 0.5:
            fly.sense({})
        if not paused:
            count = max(0, min(8, int((now - next_step) / fly.brain.dt_s) + 1))
            for _ in range(count):
                fly.step()
            steps += count
            next_step += count * fly.brain.dt_s
            if now - next_step > 0.15:
                next_step = now
        else:
            next_step = now
        if now - last_report >= 0.1:
            state = fly.state(cloud=True)
            sps = steps / max(now-last_report,1e-6)
            state.update(paused=paused,sps=sps,speed_ratio=min(1.,sps*fly.brain.dt_s),body=latest_pose)
            server.set_state(state)
            args.status_file.write_text(json.dumps({k: v for k, v in state.items() if k != "cloud"}), encoding="utf-8")
            last_report, steps = now, 0
        time.sleep(min(0.02 if paused else 0.002, max(0.0002, next_step - time.perf_counter())))
    print("STOPPED", flush=True)


if __name__ == "__main__":
    main()
