"""CUDA propagation with the same source-to-target direction as the CPU kernel."""
import numpy as np


class GpuProp:
    def __init__(self, w_csr, device_id: int = 0):
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix
        self.cp = cp
        self.n = w_csr.shape[0]
        self.dev = f"cuda:{device_id}"
        cp.cuda.Device(device_id).use()
        self.w = csr_matrix(w_csr.transpose().tocsr())
        self.scratch = cp.zeros(self.n, dtype=cp.float32)
        self.w.dot(self.scratch)
        cp.cuda.get_current_stream().synchronize()

    def propagate(self, spiked_idx: np.ndarray) -> np.ndarray:
        self.scratch.fill(0)
        self.scratch[self.cp.asarray(spiked_idx)] = 1.0
        return self.cp.asnumpy(self.w.dot(self.scratch)).astype(np.float32, copy=False)
