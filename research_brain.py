"""Current-based LIF adaptation of Shiu et al. (Nature 2024) to MaleCNS.

The numerical equations are testable against Brian2; using this dataset and
transmitter assumptions is not a biological validation of the adaptation.
Source: https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py
"""
from pathlib import Path
import numpy as np
import scipy.sparse as sp

NT_SIGN = {"acetylcholine": 1., "gaba": -1., "glutamate": -1., "histamine": -1.,
           "dopamine": .5, "serotonin": .5, "octopamine": .5, "unclear": .5}


class ResearchBrain:
    def __init__(self, graph_path: Path | None = None, *, weights=None, dt_ms=.2,
                 gpu_mode="off", seed=0, weight_mv=.275, background_hz=0.0):
        if weights is not None:
            self.w = sp.csr_matrix(weights, dtype=np.float32)
            self.ids = np.arange(self.w.shape[0])
        else:
            with np.load(graph_path) as z:
                self.ids = z["ids"]
                signs = np.full(len(self.ids), .5, dtype=np.float32)
                for code, label in enumerate(z["nt_labels"]):
                    signs[z["nt"] == code] = NT_SIGN.get(str(label), .5)
                data = z["data"].astype(np.float32)*weight_mv*np.repeat(signs,np.diff(z["indptr"]))
                self.w = sp.csr_matrix((data,z["indices"],z["indptr"].astype(np.int32)),shape=(len(self.ids),len(self.ids)))
            # Keep anatomical counts; no log compression or input normalization.
            self.w.setdiag(0)
            self.w.eliminate_zeros()
        self.n = len(self.ids)
        self.dt_ms = dt_ms
        self.dt_s = dt_ms / 1000.
        self.rest_mv = -52.
        self.threshold = -45.
        self.tau_ms = 20.
        self.syn_tau_ms = 5.
        self.delay_steps = int(round(1.8/dt_ms))
        self.refractory_steps = int(round(2.2/dt_ms))
        self.background_hz = background_hz
        self.rng = np.random.default_rng(seed)
        self.v = np.full(self.n,self.rest_mv,dtype=np.float32)
        self.g = np.zeros(self.n,dtype=np.float32)
        self.spikes = np.zeros(self.n,dtype=bool)
        self.rate_ema = np.zeros(self.n,dtype=np.float32)
        self.last_spike = np.full(self.n,-1000000,dtype=np.int64)
        self.sensory_mask = np.zeros(self.n,dtype=bool)
        self.tick = 0
        self._queue = [np.empty(0,dtype=np.int64) for _ in range(self.delay_steps+1)]
        self._a = np.float32(np.exp(-dt_ms/self.tau_ms))
        self._b = np.float32(np.exp(-dt_ms/self.syn_tau_ms))
        self._coupling = np.float32(self.syn_tau_ms/(self.tau_ms-self.syn_tau_ms)*(self._a-self._b))
        self.ema_alpha = np.float32(1.-np.exp(-self.dt_s/.05))
        self._gpu = None
        self.kernel_name = "research-cpu"
        self.gpu_threshold = int(.1*self.n)
        if gpu_mode != "off":
            try:
                from sim.gpu import GpuProp
                self._gpu = GpuProp(self.w)
                self.kernel_name = "research-hybrid(cpu+cuda:0)"
            except Exception:
                if gpu_mode == "on": raise

    def set_sensory_indices(self, indices):
        # Like the authors' Poisson targets, sensory targets have no refractory
        # interval. All other cells recover for the fixed 2.2 ms interval.
        self.sensory_mask[:] = False
        self.sensory_mask[indices] = True

    def step(self, inject_idx=None):
        # Brian2 schedule: exact state update, threshold, delayed synapses,
        # external Poisson current, reset. Refractory variables are read-only.
        ready = ((self.tick-self.last_spike) >= self.refractory_steps) | self.sensory_mask
        # Refractory cells are exactly at rest with g=0 and reject all input,
        # so the same update leaves them unchanged without expensive slicing.
        self.v[:] = self.rest_mv+(self.v-self.rest_mv)*self._a+self.g*self._coupling
        self.g *= self._b
        firing = (self.v > self.threshold) & ready
        self.last_spike[firing] = self.tick
        # threshold events become refractory before synaptic delivery.
        writable = ready & ~firing
        due = self._queue[self.tick % len(self._queue)]
        if due.size:
            if self._gpu is not None and due.size > self.gpu_threshold:
                incoming = self._gpu.propagate(due)
            else:
                incoming = np.asarray(self.w[due].sum(axis=0)).ravel()
            np.add(self.g,incoming,out=self.g,where=writable)
        self._queue[self.tick % len(self._queue)] = np.empty(0,dtype=np.int64)
        spiking_indices = np.flatnonzero(firing)
        self._queue[(self.tick+self.delay_steps) % len(self._queue)] = spiking_indices
        # Input event adds voltage as in authors' PoissonInput (0.275*250 mV).
        if inject_idx is not None and len(inject_idx):
            idx = np.unique(inject_idx)
            idx = idx[writable[idx]]
            self.v[idx] += 68.75
        if self.background_hz:
            count = self.rng.poisson(self.background_hz*self.dt_s*self.n)
            if count:
                idx = self.rng.choice(self.n,size=min(count,self.n),replace=False)
                idx = idx[writable[idx]]
                self.v[idx] += 68.75
        self.v[firing] = self.rest_mv
        self.g[firing] = 0.
        self.spikes = firing
        self.rate_ema *= 1.-self.ema_alpha
        self.rate_ema[firing] += self.ema_alpha/self.dt_s
        self.tick += 1

    def activity_fraction(self):
        return float(self.spikes.mean())
