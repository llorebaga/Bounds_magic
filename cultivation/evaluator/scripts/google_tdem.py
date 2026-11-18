"""Google's experimental cultivation circuits (Rosenfeld et al., arXiv:2512.13908 / Science 2026), made TDEM-ready.

Input: the Crumble circuits linked from the paper (external/google_circuits/*_crumble.stim, decoded by
google_crumble_to_stim.py). Google's stim files replace every non-Clifford gate by an identity tagged T_gate /
T_dagger_gate / injection, and tag the physical H_YZ gates that turn the Y rotations into virtual Z rotations
as `conjugate`. Here:
  * I[conjugate] -> H_YZ, I[T_gate] -> S, I[T_dagger_gate] -> S_DAG (the TDEM reads S / S_DAG as T / T^dag), so a
    physical T_y = H_YZ T H_YZ, as in the experiment;
  * the injection R_y(pi/4)|0> becomes SQRT_Y_DAG, then S^(+-1), then H_YZ: the same single-qubit state, written
    so that the S proxy is deterministic (sign chosen so that the fault-free real-T circuit accepts with prob. 1);
  * SI1000 noise (Gidney-Newman-McEwen, the model of Google's own simulations): CZ DEPOLARIZE2(p), physical 1q gates
    (H, H_YZ, echo X) DEPOLARIZE1(p/10), idle DEPOLARIZE1(p/10), reset X_ERROR(2p), measurement X_ERROR(5p),
    idle during a measurement / reset layer DEPOLARIZE1(2p). T gates are virtual phase updates, so a layer that
    holds only T gates takes no time and carries no noise;
  * the circuit is cut at layer CUT (default: before the final readout layer); the data are then read out by
    noiseless MPPs of all six stabilisers and Z_L, the convention of Gidney et al. for the cultivated state.

Usage: python google_tdem.py fig2|fig3 P SIGN [CUT] [OUT]   (SIGN = +1 or -1; writes the noisy circuit)"""
import itertools
import os
import sys

import stim

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {"fig2": "fig2_cultivation_crumble.stim", "fig3": "fig3_kickback_crumble.stim"}
VIRTUAL = {"T_gate", "T_dagger_gate", "injection"}


def ticks(c):
    out = [[]]
    for inst in c.flattened():
        if inst.name == "TICK":
            out.append([])
        else:
            out[-1].append(inst)
    return [t for t in out]


def qubits(inst):
    return [t.value for t in inst.targets_copy() if t.is_qubit_target]


def clifford_layers(c, sign):
    """list of layers, each a list of (name, targets) after the substitutions; plus data qubits and injection q"""
    layers = []
    data, inj = [], None
    T = ticks(c)
    for k, tick in enumerate(T):
        for inst in tick:
            if inst.tag == "injection" and inst.name == "S":
                inj = qubits(inst)[0]
            if inst.tag == "T_gate" and not data:
                data = qubits(inst)
    for k, tick in enumerate(T):
        layer = []
        for inst in tick:
            nm, tag, q = inst.name, inst.tag, qubits(inst)
            if nm in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
                continue
            if tag == "injection":
                if nm == "S":
                    layer.append(("S" if sign > 0 else "S_DAG", q))
                continue                                   # I[injection]: placeholder
            if nm == "H_YZ" and tag == "conjugate" and q == [inj] and k + 1 < len(T) and \
                    any(i.tag == "injection" and i.name == "S" for i in T[k + 1]):
                layer.append(("SQRT_Y_DAG", q))            # |0> -> |->
                continue
            if tag == "conjugate":
                layer.append(("H_YZ", q))
            elif tag == "T_gate":
                layer.append(("S", q))
            elif tag == "T_dagger_gate":
                layer.append(("S_DAG", q))
            elif tag == "echo":
                layer.append(("ECHO", q))
            elif nm == "I":
                continue
            else:
                tg = inst.targets_copy()
                layer.append((nm, [t.value for t in tg]))
        layers.append(layer)
    return layers, data, inj


def build(layers, nq, data, p, cut, faces_x, faces_z, noiseless=False, clean_before=0):
    c = stim.Circuit()
    for li, layer in enumerate(layers[:cut]):
        names = {nm for nm, _ in layer}
        if not layer:
            continue
        if names <= {"S", "S_DAG"}:
            for nm, q in layer:
                c.append(nm, q)
            c.append("TICK")
            continue
        busy, meas, reset = set(), set(), set()
        for nm, q in layer:
            busy.update(q)
            if nm in ("M", "MX"):
                meas.update(q)
            if nm in ("R", "RX"):
                reset.update(q)
        heavy = bool(meas or reset)
        quiet = noiseless or li < clean_before
        for nm, q in layer:
            if nm in ("M", "MX") and not quiet:
                c.append("X_ERROR" if nm == "M" else "Z_ERROR", q, 5 * p)
            c.append("I" if nm == "ECHO" else nm, q)
            if quiet:
                continue
            if nm in ("R", "RX"):
                c.append("X_ERROR" if nm == "R" else "Z_ERROR", q, 2 * p)
            elif nm in ("CZ", "CX"):
                c.append("DEPOLARIZE2", q, p)
            elif nm in ("H", "H_YZ", "SQRT_Y_DAG", "ECHO", "X", "Z", "S", "S_DAG"):
                c.append("DEPOLARIZE1", q, p / 10)
        idle = [q for q in range(nq) if q not in busy]
        if idle and not quiet:
            c.append("DEPOLARIZE1", idle, 2 * p if heavy else p / 10)
        c.append("TICK")
    # noiseless readout of the six stabilisers and Z_L
    groups = [("X", f) for f in faces_x] + [("Z", f) for f in faces_z] + [("Z", data)]
    tg = []
    for P, f in groups:
        for i, q in enumerate(f):
            if i:
                tg.append(stim.target_combiner())
            tg.append(stim.target_x(q) if P == "X" else stim.target_z(q))
    c.append("MPP", tg)
    ng = len(groups)
    for i in range(ng - 1):
        c.append("DETECTOR", [stim.target_rec(-ng + i)])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return c


def faces_of(state_circuit, data):
    """weight-4 X and Z stabilisers of the data block in the noiseless proxy state"""
    sim = stim.TableauSimulator()
    sim.do(state_circuit.without_noise())
    fx, fz = [], []
    for f in itertools.combinations(data, 4):
        for P, lst in (("X", fx), ("Z", fz)):
            Q = stim.PauliString(state_circuit.num_qubits)
            for q in f:
                Q[q] = P
            if sim.peek_observable_expectation(Q) != 0:
                lst.append(list(f))
    return fx, fz


def make(fig, p, sign, cut=None, ideal_input=False):
    """ideal_input: injection, encoding and everything before the first check layer are noiseless"""
    c0 = stim.Circuit.from_file(os.path.join(HERE, "..", "external", "google_circuits", SRC[fig]))
    layers, data, inj = clifford_layers(c0, sign)
    if cut is None:
        cut = max(k for k, L in enumerate(layers) if any(nm == "M" for nm, _ in L))   # final readout layer
    nq = c0.num_qubits
    # stabilisers of the data block in the proxy just before the cut
    probe = build(layers, nq, data, 0, cut, [], [], noiseless=True)
    probe = stim.Circuit(str(probe).rsplit("MPP", 1)[0])
    fx, fz = faces_of(probe, data)
    fx, fz = independent(fx), independent(fz)
    first = min(k for k, L in enumerate(layers) if any(nm == "S" for nm, q in L if q == data)) - 2         if ideal_input else 0                              # the H_YZ and echo layers before the first T layer
    return build(layers, nq, data, p, cut, fx, fz, clean_before=first),         dict(data=data, inj=inj, faces_x=fx, faces_z=fz, cut=cut, clean_before=first)


def independent(faces):
    import numpy as np
    from gf2 import rref
    if not faces:
        return faces
    qs = sorted({q for f in faces for q in f})
    M = np.zeros((len(faces), len(qs)), dtype=np.uint8)
    for i, f in enumerate(faces):
        for q in f:
            M[i, qs.index(q)] = 1
    _, piv = rref(M.T)
    return [faces[i] for i in piv]


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    fig, p, sign = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    cut = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(HERE, "..", "results", f"google_{fig}_p{p:g}_s{sign:+d}"
                                                               + (f"_cut{cut}" if cut else "") + ".stim")
    c, info = make(fig, p, sign, cut)
    c.to_file(out)
    print(out, info, c.num_qubits, "qubits", c.num_detectors, "detectors")
else:
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
