"""Tests of cultiv.py on random small Clifford+T circuits.

(i)  harmlessness from the fault map (D, U, E) agrees with direct simulation for all single faults and random
     pairs of faults;
(ii) for circuits whose skeleton map M has full row rank (Theorem 1), the dense skeleton map M_F, computed by
     feeding basis states to the injection gadgets, is zero exactly when D(F) != 0, and otherwise U(F) = 0 and
     M_F is a multiple of M E(F).
Usage: python cultiv_test.py [n_circuits] [seed]
"""
import itertools
import sys

import numpy as np

import cultiv as cv


def random_circuit(rng, nq, length, n_t, n_meas):
    ops = [("R", q, "X" if rng.random() < .5 else "Z") for q in range(nq)]
    kinds = ["T"] * n_t + ["M"] * n_meas + ["C"] * (length - n_t - n_meas)
    rng.shuffle(kinds)
    for k in kinds:
        if k == "T":
            ops.append(("T", int(rng.integers(nq)), int(rng.choice([1, -1]))))
        elif k == "C":
            if rng.random() < .5:
                c, t = rng.choice(nq, 2, replace=False)
                ops.append(("U2", str(rng.choice(["CX", "CY", "CZ"])), int(c), int(t)))
            else:
                ops.append(("U1", int(rng.integers(nq)), str(rng.choice(["H", "S", "Sdg"]))))
        else:
            w = int(rng.integers(1, 3))
            qs = tuple(int(q) for q in rng.choice(nq, w, replace=False))
            ops.append(("M", "".join(rng.choice(list("XYZ"), w)), qs, 1, False))
    ops.append(("M", "Z", (nq - 1,), 1, True))
    for i, op in enumerate(ops):
        if op[0] == "M":
            best = max((1, -1), key=lambda s: np.linalg.norm(cv.simulate(ops[:i] + [op[:3] + (s,) + op[4:]], nq)))
            ops[i] = op[:3] + (best,) + op[4:]
    return ops


def skeleton_matrix(ops, nq, out, F, t):
    cols = []
    for b in itertools.product([0, 1], repeat=t):
        tin = {j: ((1, 0) if bj == 0 else (0, 1)) for j, bj in enumerate(b)}
        cols.append(cv.out_vec(cv.simulate(ops, nq, F, tinput=tin), nq, out))
    return np.array(cols).T


def pauli_matrix(E, t):
    U = np.array([[1.0 + 0j]])
    for j in range(t):
        xb, zb = (E >> j) & 1, (E >> (t + j)) & 1
        U = np.kron(U, np.linalg.matrix_power(cv.P1["X"], xb) @ np.linalg.matrix_power(cv.P1["Z"], zb))
    return U


if __name__ == "__main__":
    ncirc = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    tot = bad = full = dense_checked = dense_bad = 0
    done = 0
    while done < ncirc:
        nq = int(rng.integers(2, 4))
        ops = random_circuit(rng, nq, int(rng.integers(8, 14)), int(rng.integers(1, 4)), int(rng.integers(1, 3)))
        out = list(range(nq - 1))
        if np.linalg.norm(cv.out_vec(cv.simulate(ops, nq), nq, out)) < 1e-6:
            continue
        done += 1
        A = cv.Analysis(ops, nq, out)
        singles = [(i, f, k) for (li, i, f, k) in A.faults]
        configs = [({i: f}, k) for i, f, k in singles]
        idx = rng.choice(len(singles), min(300, len(singles)))
        for a, b in zip(idx[::2], idx[1::2]):
            (i1, f1, k1), (i2, f2, k2) = singles[a], singles[b]
            if i1 != i2:
                configs.append(({i1: f1, i2: f2}, tuple(x ^ y for x, y in zip(k1, k2))))
        for F, k in configs:
            tot += 1
            bad += A.harmless_key(k) != A.harmless_direct(F)
        t = A.sk.t
        M = skeleton_matrix(ops, nq, out, {}, t)
        if np.linalg.matrix_rank(M, tol=1e-8) < M.shape[0]:
            continue
        full += 1
        for F, (D, U, E) in configs[:200]:
            MF = skeleton_matrix(ops, nq, out, F, t)
            dense_checked += 1
            if D:
                dense_bad += np.linalg.norm(MF) > 1e-9
            else:
                pred = (M @ pauli_matrix(E, t)).reshape(-1)
                dense_bad += U != 0 or np.linalg.norm(MF) < 1e-9 or not cv.parallel(MF.reshape(-1), pred)
    print("circuits", done, "configurations", tot, "harmlessness mismatches", bad)
    print("full-rank circuits", full, "configurations checked against dense M_F", dense_checked, "failures", dense_bad)
    print("ALL OK" if bad == 0 and dense_bad == 0 else "SOME CHECK FAILED")
