"""Verified-cat double check on the d=5 colour code (19 qubits), the real-T-native counterpart of Gidney's GHZ-fold
check.  Every data qubit interacts with the check exactly once (CX cat_q -> q), and the cat is verified (Z_i Z_{i+1}
parities) before it touches the data, so one fault leaves at most one X on the data: by the coherent-completion
lemma real T then costs no distance.

Layout-free circuit (all-to-all), built with the ASAP / uniform-noise Builder of the first study:
  ideal encoded input (noiseless injection T|+> on a carrier + CX-only CSS encoder)
  -> stabiliser round -> [check round -> stabiliser round] x rounds -> noiseless MPP readout.
A check round: U^dag layer (T or T^dag per qubit, pattern s), GHZ cat on 19 qubits (CX tree), verification of all
adjacent Z_i Z_{i+1}, CX cat_q -> data_q, MX cat (parity = check outcome), U layer.
Gates are written as S / S_DAG (the TDEM reads them as T / T^dag)."""
import sys
import numpy as np
import stim

sys.path.insert(0, __file__.rsplit("vcat.py", 1)[0])
from cult_circuits import Builder          # noqa: E402

from gf2 import rref                        # noqa: E402
from tdem import sublattice_signs           # noqa: E402


def encoder_plan(n, faces, logical):
    """pivots of the X stabilisers in RREF and a carrier qubit of the reduced logical X"""
    M = np.zeros((len(faces), n), dtype=np.uint8)
    for i, f in enumerate(faces):
        M[i, f] = 1
    R, piv = rref(M)
    R = R[:len(piv)]
    L = np.zeros(n, dtype=np.uint8)
    L[logical] = 1
    for r, c in zip(R, piv):
        if L[c]:
            L ^= r
    carrier = [q for q in range(n) if L[q] and q not in piv][0]
    return [list(np.nonzero(r)[0]) for r in R], piv, list(np.nonzero(L)[0]), carrier


def make_vcat(faces, n=19, p=1e-3, rounds=2, inject_dag=False, flip_all=False, idle=True, flagged=True, verify=True,
              model="uniform"):
    B = Builder(p, idle, model)
    d = B.alloc(n)
    mcount = [0]

    def key(tag):
        mcount[0] += 1
        return (tag, mcount[0])

    signs = sublattice_signs(n, faces)
    if flip_all:
        signs = [-s for s in signs]
    gens, piv, Lrow, carrier = encoder_plan(n, faces, list(range(n)))

    def U_layer(dag):
        for q in range(n):
            g = "S" if (signs[q] > 0) != dag else "S_DAG"
            B.g(g, d[q])

    # ---- stabiliser round with one flag per stabiliser (coupled after the first, before the last data CX) ----
    def stabiliser_round():
        for P in "XZ":
            for f in faces:
                a = B.alloc(1)[0]
                fl = B.alloc(1)[0] if flagged else None
                if P == "X":
                    B.g("RX", a)
                    if flagged:
                        B.g("R", fl)
                    for i, q in enumerate(f):
                        B.g("CX", a, d[q])
                        if flagged and (i == 0 or i == len(f) - 2):
                            B.g("CX", a, fl)
                    k = key("sx"); B.g("MX", a, key=k); B.detectors.append([k])
                    if flagged:
                        k = key("fx"); B.g("M", fl, key=k); B.detectors.append([k])
                else:
                    B.g("R", a)
                    if flagged:
                        B.g("RX", fl)
                    for i, q in enumerate(f):
                        B.g("CX", d[q], a)
                        if flagged and (i == 0 or i == len(f) - 2):
                            B.g("CX", fl, a)
                    k = key("sz"); B.g("M", a, key=k); B.detectors.append([k])
                    if flagged:
                        k = key("fz"); B.g("MX", fl, key=k); B.detectors.append([k])

    def check_round():
        U_layer(True)
        cat = B.alloc(n)
        B.g("RX", cat[0])
        for q in cat[1:]:
            B.g("R", q)
        # GHZ by a binary CX tree (depth ~ log2 n)
        have = [0]
        while len(have) < n:
            new = []
            for i in have:
                j = len(have) + len(new)
                if j < n:
                    B.g("CX", cat[i], cat[j])
                    new.append(j)
            have += new
        ver = B.alloc(n - 1) if verify else []
        for i in range(n - 1 if verify else 0):
            B.g("R", ver[i]); B.g("CX", cat[i], ver[i]); B.g("CX", cat[i + 1], ver[i])
            k = key("ver"); B.g("M", ver[i], key=k); B.detectors.append([k])
        for q in range(n):
            B.g("CX", cat[q], d[q])
        ks = []
        for q in range(n):
            k = key("cat"); B.g("MX", cat[q], key=k); ks.append(k)
        B.detectors.append(ks)
        U_layer(False)

    stabiliser_round()
    for _ in range(rounds):
        check_round()
        stabiliser_round()
    for f in faces:
        for P in "XZ":
            k = key("final")
            B.g("MPP", [stim.target_combined_paulis([stim.target_x(d[q]) if P == "X" else stim.target_z(d[q]) for q in f])], key=k)
            B.detectors.append([k])
    k = key("obs")
    B.g("MPP", [stim.target_combined_paulis([stim.target_y(q) for q in d])], key=k)
    B.observables.append([k])
    body = B.build()

    # ---- noiseless ideal input: carrier |+>, injection, CX-only encoder ----
    pre = stim.Circuit()
    pre.append("R", d)
    pre.append("H", [d[carrier]] + [d[q] for q in piv])
    pre.append("S_DAG" if inject_dag else "S", [d[carrier]])
    for q in Lrow:
        if q != carrier:
            pre.append("CX", [d[carrier], d[q]])
    for g, pv in zip(gens, piv):
        for q in g:
            if q != pv:
                pre.append("CX", [d[pv], d[q]])
    pre.append("TICK")
    # the builder starts with R on fresh qubits; drop resets of the data (already prepared)
    out = pre.copy()
    for inst in body.flattened():
        out.append(inst)
    return out
