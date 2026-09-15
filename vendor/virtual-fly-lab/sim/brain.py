"""LIF brain kernel over the full MaleCNS connectome, event-driven propagation.

Weight semantics: data = synapse count per directed pair. We compress with log1p
and sign them by the presynaptic neuron's consensus neurotransmitter.
"""
from pathlib import Path

import numpy as np
import scipy.sparse as sp

NT_SIGN = {
    "acetylcholine": 1.0,
    "gaba": -1.0,
    # Uniform transmitter signs are a modeling assumption, not receptor data.
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 0.5,
    "serotonin": 0.5,
    "octopamine": 0.5,
    "unclear": 0.5,
}
NO_NT_SIGN = 0.5


class Brain:
    def __init__(
        self,
        graph_path: Path,
        dt_ms: float = 2.0,
        tau_ms: float = 20.0,
        threshold: float = 1.0,
        gain: float = 1.0,
        norm_const: float = 1.6,
        norm_power: float = 1.0,
        background_hz: float = 0.5,
        homeostasis_target: float = 0.02,
        homeostasis_gain: float = 8.0,
        thr_target_hz: float = 25.0,
        thr_gain: float = 0.3,
        gpu_mode: str = "auto",
        seed: int = 0,
    ):
        z = np.load(graph_path)
        self.ids = z["ids"]
        self.nt_labels = [str(x) for x in z["nt_labels"]]
        self.nt = z["nt"]
        n = len(self.ids)
        self.n = n

        sign = np.full(n, NO_NT_SIGN, dtype=np.float32)
        for code, label in enumerate(self.nt_labels):
            sign[self.nt == code] = NT_SIGN.get(label, NO_NT_SIGN)
        w = np.log1p(z["data"].astype(np.float32))
        # CSR rows are presynaptic neurons; columns are their postsynaptic targets.
        w *= np.repeat(sign, np.diff(z["indptr"]))
        # forward pre->post CSR: row-select by spiking presynaptic neurons
        self.w = sp.csr_matrix(
            (w, z["indices"], z["indptr"].astype(np.int32)), shape=(n, n)
        )
        # partial divisive normalization: hub neurons are tamed (col_abs^-norm_power)
        # while relative weight structure survives; autapses would become
        # perpetual oscillators, drop them
        col_abs = np.asarray(np.abs(self.w).sum(axis=0)).ravel()
        scale = (
            norm_const / np.maximum(col_abs, 1.0) ** norm_power
        ).astype(np.float32)
        self.w.data *= scale[self.w.indices]
        self.w.setdiag(0)
        self.w.eliminate_zeros()

        self.dt_ms = dt_ms
        self.dt_s = dt_ms / 1000.0
        self.tau_ms = tau_ms
        self.leak = np.exp(-dt_ms / tau_ms, dtype=np.float32)
        self.threshold = threshold
        self.gain = gain
        self.background_hz = background_hz
        self.homeostasis_target = homeostasis_target
        self.homeostasis_gain = homeostasis_gain
        self.thr_target_hz = thr_target_hz
        self.thr_gain = thr_gain
        self.rng = np.random.default_rng(seed)

        self.kernel_name = "cpu"
        self._gpu = None
        if gpu_mode != "off":
            try:
                from .gpu import GpuProp

                self._gpu = GpuProp(self.w)
                self.kernel_name = f"hybrid(cpu+{self._gpu.dev})"
                # CPU event-driven wins below ~10% activity; GPU cost is constant
                self.gpu_threshold = int(0.10 * n)
            except Exception as e:  # torch missing, no GPU, sm_70 unsupported...
                if gpu_mode == "on":
                    raise
                print(f"GPU kernel unavailable ({type(e).__name__}: {e}); using CPU")

        self.v = np.zeros(n, dtype=np.float32)
        self.spikes = np.zeros(n, dtype=bool)
        self.rate_ema = np.zeros(n, dtype=np.float32)  # Hz, per neuron
        ema_alpha = 1.0 - np.exp(-50.0 * dt_ms / 1000.0)  # ~50 ms window
        self.ema_alpha = np.float32(ema_alpha)
        self.activity_ema = 0.0  # slow fraction-active EMA, drives homeostatic inhibition

    def step(self, inject_idx: np.ndarray | None = None) -> None:
        """One dt_ms of biology. inject_idx: neurons forced to spike (sensory drive)."""
        self.v *= self.leak
        spiked = np.flatnonzero(self.spikes)
        if spiked.size:
            if self._gpu is not None and spiked.size > self.gpu_threshold:
                inflow = self._gpu.propagate(spiked)
            else:
                inflow = np.asarray(self.w[spiked].sum(axis=0)).ravel()
            self.v += self.gain * inflow
        # membrane leak + homeostatic brake (proportional controller on slow activity)
        overshoot = max(0.0, self.activity_ema - self.homeostasis_target)
        self.v -= overshoot * self.homeostasis_gain * self.dt_ms / self.tau_ms
        # spontaneous background: a random subset fires directly
        k = int(self.background_hz * self.dt_s * self.n)
        bg = self.rng.choice(self.n, size=k, replace=False) if k else np.empty(0, dtype=np.int64)
        new_spikes = self.v > (
            self.threshold
            + self.thr_gain * np.maximum(self.rate_ema - self.thr_target_hz, 0.0)
        )
        if inject_idx is not None and inject_idx.size:
            new_spikes[inject_idx] = True
        new_spikes[bg] = True
        # post-spike hyperpolarization acts as a 1-2 step refractory
        self.v[new_spikes] = -0.3
        self.spikes = new_spikes
        # slow activity EMA for the homeostat (~500 ms)
        a = 1.0 - np.exp(-self.dt_ms / 500.0)
        self.activity_ema += a * (float(new_spikes.mean()) - self.activity_ema)
        # per-neuron rate EMA in Hz (spike this step = 1/dt_s Hz)
        self.rate_ema *= 1.0 - self.ema_alpha
        if new_spikes.any():
            self.rate_ema[new_spikes] += self.ema_alpha * (1000.0 / self.dt_ms)

    def activity_fraction(self) -> float:
        return float(self.spikes.mean())
