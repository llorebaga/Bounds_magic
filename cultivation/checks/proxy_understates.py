"""A proxy can understate fault tolerance (paper/cultivation.tex, Appendix F, "What the proxies miss here").

The nine-T circuit measures its code directly on T states. This script runs the same circuit with T replaced by S and by Z and prints:
  1. the circuit of Proposition 19 with its ideal decoder: harmful single faults with T, S and Z;
  2. the same stabilizers, all flagged and measured twice, without the decoder: single faults that violate the fault-tolerance
     condition of Simsek et al. (arXiv:2611.04909), that is, whose accepted output is not in the span of {P tau: wt(P) <= 1};
  3. version 2 with T written as S T-dagger on qubits 0 and 3, for which the outputs are exactly Q|+> in one logical frame.

Stabilizers and stabilising Cliffords of the output leave P tau unchanged up to a phase, so the span test covers both.
Usage: python proxy_understates.py     Prints ALL OK.
"""
import collections
import itertools

import numpy as np
import stim

import cultiv as cv
import cultivation_t9 as t9

N, NQ = 9, 11
OUT = list(range(N))
POWER = {"T": 1, "S": 2, "Z": 4}


def exponents(ops, k):
    d, j = {}, 0
    for op in ops:
        if op[0] == "T":
            d[j] = op[2] * k
            j += 1
    return d


def output(ops, k, fault=None):
    return cv.out_vec(cv.simulate(ops, NQ, fault, texp=exponents(ops, k)), NQ, OUT)


def apply9(v, letters):
    v = v.reshape([2] * N)
    for q, c in enumerate(letters):
        if c != "I":
            v = np.moveaxis(np.tensordot(cv.P1[c], v, axes=([1], [q])), 0, q)
    return v.reshape(-1)


WEIGHT_ONE = ["I" * q + c + "I" * (N - q - 1) for q in range(N) for c in "XYZ"]


def harmful_singles(ops, ideal):
    """single faults whose accepted output differs from the fault-free output, after the ideal decoder"""
    res, table = {}, {}
    for name, k in POWER.items():
        ref = output(ops, k)
        for i, faults in cv.locations(ops, ideal):
            for f in faults:
                v = output(ops, k, {i: f})
                n = np.linalg.norm(v)
                st = "rejected" if n < 1e-9 else ("harmless" if cv.parallel(v, ref) else "harmful")
                table.setdefault((i, f), {})[name] = st
        res[name] = sum(s[name] == "harmful" for s in table.values())
    return res, table


def violations(ops):
    """single faults outside the span of the weight-one errors on the fault-free output; also the Bloch vectors of the outputs"""
    gens = [stim.PauliString(g) for g in t9.FACTORY]
    tab = stim.Tableau.from_stabilizers(gens, allow_underconstrained=True)
    mat = lambda p: p.to_unitary_matrix(endian="big")
    X, Z = mat(tab.x_output(N - 1)), mat(tab.z_output(N - 1))
    logical = [X, 1j * X @ Z, Z]
    bad, bloch = {}, {}
    for name, k in POWER.items():
        tau = output(ops, k)
        tau = tau / np.linalg.norm(tau)
        bloch[name] = np.array([np.vdot(tau, m @ tau).real for m in logical])
        u, sv, _ = np.linalg.svd(np.array([tau] + [apply9(tau, p) for p in WEIGHT_ONE]).T, full_matrices=False)
        B = u[:, sv > 1e-9]
        count = 0
        for i, faults in cv.locations(ops):
            for f in faults:
                v = output(ops, k, {i: f})
                n = np.linalg.norm(v)
                if n > 1e-9:
                    v = v / n
                    count += np.linalg.norm(v - B @ (B.conj().T @ v)) > 1e-7
        bad[name] = int(count)
    return bad, bloch


def two_rounds():
    ops, ideal = t9.circuit([(g, o, 1) for g, o, _ in t9.PLAN] * 2)
    return [op for i, op in enumerate(ops) if i not in ideal]


# 1. Proposition 21
ops, ideal = t9.circuit(t9.PLAN)
counts, table = harmful_singles(ops, ideal)
print("nine-T circuit, harmful single faults:", counts)
example = next((i, f) for (i, f), s in table.items() if ops[i][:2] == ("T", 8) and f == "Z")
print("Z fault on the T gate of qubit 8:", table[example])
ok = counts == {"T": 0, "S": 8, "Z": 74} and table[example] == {"T": "rejected", "S": "harmful", "Z": "harmful"}

# 2. Two flagged rounds
ops2 = two_rounds()
bad2, bloch2 = violations(ops2)
print("two flagged rounds, two-qubit gates", sum(op[0] == "U2" for op in ops2), "violations:", bad2)
ok &= bad2 == {"T": 0, "S": 8, "Z": 75}
a = bloch2
print("outputs up to a logical Pauli: a_T = (a_S + a_Z)/sqrt2:", bool(np.allclose(a["T"], (a["S"] + a["Z"]) / np.sqrt(2), atol=1e-9)))

# 3. Exact family
ops3 = []
for op in ops2:
    if op[0] == "T" and op[1] in (0, 3):
        ops3 += [("T", op[1], -1), ("U1", op[1], "S")]
    else:
        ops3.append(op)
bad3, a = violations(ops3)
x, y = -a["Z"], a["S"]
exact = bool(abs(x @ y) < 1e-9 and np.allclose(a["T"], (x + y) / np.sqrt(2), atol=1e-9))
print("S T-dagger on qubits 0 and 3, violations:", bad3, "| exact Q|+> family:", exact)
ok &= exact and bad3["T"] == 0 and bad3["S"] > 0 and bad3["Z"] > 0
print("ALL OK" if ok else "FAILED")
