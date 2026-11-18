"""Idle noise on the data qubits of circuits that start from a noiseless encoder.

The builders of fold.py, vcat.py and nn_cat.py prepare the data in a noiseless block before the Builder body, so
the Builder never marks the data as live and adds no idle noise to them. add_data_idle() adds it afterwards with
the Builder's own rule: in every tick after the encoder tick and before the final noiseless MPP readout, every
data qubit that no operation touches gets DEPOLARIZE1 with the file's uniform strength p. noise.si1000() then
rescales these channels like any other idle channel (p/10, or 2p in ticks with a measurement or reset).
The data qubits are those of the final MPP readout."""
import stim

NOISE = ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2")
ANNOT = ("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK")


def _segments(c):
    segs = [[]]
    for inst in c.flattened():
        if inst.name == "TICK":
            segs.append([])
        else:
            segs[-1].append(inst)
    return segs


def data_qubits(c):
    qs = set()
    for inst in c.flattened():
        if inst.name == "MPP":
            qs.update(t.value for t in inst.targets_copy() if not t.is_combiner)
    return qs


def add_data_idle(c, p=None):
    """returns (new circuit, number of idle locations added); data qubits that already carry idle noise in a tick
    (for example a fold root that is reset inside the Builder, or data reset by the Builder itself) are left alone"""
    segs = _segments(c)
    D = data_qubits(c)
    if p is None:
        ps = {a for inst in c.flattened() if inst.name in ("DEPOLARIZE1", "DEPOLARIZE2") for a in inst.gate_args_copy()}
        assert len(ps) == 1, f"expected one uniform noise strength, found {ps}"
        p = ps.pop()
    assert not any(inst.name in NOISE for inst in segs[0]), "the first tick is expected to be the noiseless encoder"
    out, added = stim.Circuit(), 0
    for i, seg in enumerate(segs):
        touched, noisy_idle = set(), set()
        for inst in seg:
            if inst.name not in NOISE + ANNOT:
                touched.update(t.value for t in inst.targets_copy() if t.is_qubit_target or t.is_x_target
                               or t.is_y_target or t.is_z_target)
        for inst in seg:
            if inst.name == "DEPOLARIZE1":
                noisy_idle.update(t.value for t in inst.targets_copy() if t.value in D and t.value not in touched)
        has_mpp = any(inst.name == "MPP" for inst in seg)
        for inst in seg:
            out.append(inst)
        idle = sorted(D - touched - noisy_idle)
        if i > 0 and idle and not has_mpp:
            out.append("DEPOLARIZE1", idle, p)
            added += len(idle)
        if i < len(segs) - 1:
            out.append("TICK")
    return out, added
