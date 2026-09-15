"""Independent numerical comparison to Brian2's exact linear integration."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_brain import ResearchBrain


@pytest.mark.parametrize("dt_ms", [.2, .1])
def test_matches_brian2_delays_refractory_and_signed_network(dt_ms):
    b = pytest.importorskip("brian2")
    b.start_scope()
    b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = dt_ms*b.ms
    weights = np.zeros((4,4),dtype=np.float32)
    weights[0,1],weights[1,2],weights[2,0],weights[0,3],weights[3,1] = 180,140,100,30,-50
    kernel = ResearchBrain(weights=weights,dt_ms=dt_ms)
    kernel.v[0] = -40
    kernel.g[3] = 140
    group = b.NeuronGroup(4, """
        dv/dt = (-52*mV-v+g)/(20*ms) : volt (unless refractory)
        dg/dt = -g/(5*ms) : volt (unless refractory)
        """, threshold="v > -45*mV", reset="v=-52*mV; g=0*mV",
        refractory=2.2*b.ms, method="exact")
    group.v = kernel.v*b.mV
    group.g = kernel.g*b.mV
    syn = b.Synapses(group,group,"w:volt",on_pre="g_post += w",delay=1.8*b.ms)
    pre,post = np.nonzero(weights)
    syn.connect(i=pre,j=post)
    syn.w = weights[pre,post]*b.mV
    monitor = b.StateMonitor(group,["v","g"],record=True,when="end")
    spikes = b.SpikeMonitor(group)
    b.Network(group,syn,monitor,spikes).run(35*b.ms)
    voltages, currents, events = [],[],[]
    for tick in range(round(35/dt_ms)):
        kernel.step()
        voltages.append(kernel.v.copy())
        currents.append(kernel.g.copy())
        events += [(int(i),tick) for i in np.flatnonzero(kernel.spikes)]
    np.testing.assert_allclose(np.array(voltages).T, monitor.v/b.mV, atol=2e-4,rtol=0)
    np.testing.assert_allclose(np.array(currents).T, monitor.g/b.mV, atol=2e-4,rtol=0)
    expected = list(zip(spikes.i[:].astype(int),np.rint(spikes.t[:]/b.ms/dt_ms).astype(int)))
    assert events == expected
    assert len(events)>3


def test_sensory_voltage_events_and_zero_refractory_match_brian2():
    import brian2 as b
    b.start_scope()
    b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = .2*b.ms
    kernel = ResearchBrain(weights=np.array([[0,90],[0,0]],dtype=np.float32))
    kernel.set_sensory_indices([0])
    group = b.NeuronGroup(2,"""
        dv/dt = (-52*mV-v+g)/(20*ms) : volt (unless refractory)
        dg/dt = -g/(5*ms) : volt (unless refractory)
        rfc : second
        """,threshold="v > -45*mV",reset="v=-52*mV; g=0*mV",refractory="rfc",method="exact")
    group.v = -52*b.mV
    group.rfc = [0,2.2]*b.ms
    syn = b.Synapses(group,group,on_pre="g_post += 90*mV",delay=1.8*b.ms)
    syn.connect(i=0,j=1)
    ticks = np.array([0,2,9,11,16,25,32,36])
    driver = b.SpikeGeneratorGroup(1,np.zeros(len(ticks),dtype=int),ticks*.2*b.ms)
    external = b.Synapses(driver,group,on_pre="v_post += 68.75*mV")
    external.connect(i=0,j=0)
    monitor = b.StateMonitor(group,["v","g"],record=True,when="end")
    spike_monitor = b.SpikeMonitor(group)
    b.Network(group,syn,driver,external,monitor,spike_monitor).run(15*b.ms)
    values, events = [],[]
    for tick in range(75):
        kernel.step(np.array([0]) if tick in ticks else None)
        values.append(kernel.v.copy())
        events += [(int(i),tick) for i in np.flatnonzero(kernel.spikes)]
    np.testing.assert_allclose(np.array(values).T,monitor.v/b.mV,atol=2e-4,rtol=0)
    expected = list(zip(spike_monitor.i[:].astype(int),np.rint(spike_monitor.t[:]/b.ms/.2).astype(int)))
    assert events == expected
    assert len(events)>len(ticks)
