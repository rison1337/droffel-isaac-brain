import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "virtual-fly-lab"))
from sim.brain import Brain
from sim.gpu import GpuProp


def small_brain(tmp_path):
    w = sp.csr_matrix(([2., 3.], ([0, 1], [1, 2])), shape=(3, 3))
    path = tmp_path / "graph.npz"
    np.savez(path, ids=[10, 11, 12], nt=[0, 1, 0], nt_labels=["acetylcholine", "gaba"],
             data=w.data, indices=w.indices, indptr=w.indptr)
    return Brain(path, background_hz=0, gpu_mode="off", threshold=100, homeostasis_gain=0)


def test_transmitter_sign_belongs_to_source(tmp_path):
    brain = small_brain(tmp_path)
    assert brain.w[0, 1] > 0
    assert brain.w[1, 2] < 0


def test_unstimulated_membrane_decays(tmp_path):
    brain = small_brain(tmp_path)
    brain.v[:] = [0.8, 0.4, -0.3]
    before = brain.v.copy()
    brain.step()
    np.testing.assert_allclose(brain.v, before * brain.leak)


def test_cpu_signal_travels_forward(tmp_path):
    brain = small_brain(tmp_path)
    brain.spikes[0] = True
    brain.step()
    assert brain.v[1] > 0
    assert brain.v[0] == 0
    assert brain.v[2] == 0


def test_cuda_matches_directed_cpu_sum():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA device unavailable")
    w = sp.csr_matrix(([2., -3., 5.], ([0, 1, 3], [1, 2, 1])), shape=(4, 4), dtype=np.float32)
    gpu = GpuProp(w)
    for firing in (np.array([0]), np.array([1]), np.array([0, 3]), np.array([], dtype=int)):
        expected = np.asarray(w[firing].sum(axis=0)).ravel()
        np.testing.assert_allclose(gpu.propagate(firing), expected)
