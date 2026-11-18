"""Every face of the triangular 6.6.6 colour code meets a minimum-weight logical support in exactly two qubits.

Argument (Corollary "Face hooks"): if a minimum-weight Z logical support L contains a qubit of a face f, then
|L & f| is even (Z^L commutes with the X stabilizer on f), nonzero, and at most |f|/2, since otherwise L + f would be
a lighter logical operator; for faces of weight 4 or 6 this leaves |L & f| = 2. It therefore suffices that every face
contains a qubit that lies in some minimum-weight logical support. Not every qubit does: at distance 2m+1,
m(m-1)/2 qubits lie in none (at distance five, the qubit shared by the three hexagons).

For each qubit q we solve min |x| over x in F_2^n with x_q = 1, even overlap with every face and odd total weight
(a Z logical operator: it commutes with the X faces and anticommutes with X on all data), as an integer program with
HiGHS (scipy.optimize.milp). The optimal supports are verified directly over F_2, and their overlaps with all faces
are listed.
Usage: python cultivation_face_cover_check.py [DELTA_MAX]   (default 11; delta = 13 takes more than half an hour)"""
import os, sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluator", "src"))
from colorcode import color_code_faces


def min_logical_through(faces, n, q):
    m = len(faces)
    nv = n + m + 1                             # x (n binary), y (m integers), z (1 integer)
    A = np.zeros((m + 2, nv))
    lo = np.zeros(m + 2)
    hi = np.zeros(m + 2)
    for j, f in enumerate(faces):
        A[j, f] = 1
        A[j, n + j] = -2                       # sum_{i in f} x_i = 2 y_j
    A[m, :n] = 1
    A[m, n + m] = -2
    lo[m] = hi[m] = 1                          # sum_i x_i = 2 z + 1
    A[m + 1, q] = 1
    lo[m + 1] = hi[m + 1] = 1                  # x_q = 1
    c = np.r_[np.ones(n), np.zeros(m + 1)]
    ub = np.r_[np.ones(n), np.full(m, 3), [n]]
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nv),
               bounds=Bounds(np.zeros(nv), ub))
    assert res.status == 0, res.message
    L = set(np.nonzero(np.round(res.x[:n]).astype(int))[0].tolist())
    assert q in L and len(L) % 2 == 1 and all(len(L & set(f)) % 2 == 0 for f in faces)
    return L


def main():
    dmax = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    ok = True
    for delta in range(3, dmax + 1, 2):
        faces, n = color_code_faces(delta)
        assert n == (3 * delta ** 2 + 1) // 4, (delta, n)
        weight, supports = {}, []
        for q in range(n):
            if any(q in L for L in supports):
                weight[q] = delta
                continue
            L = min_logical_through(faces, n, q)
            weight[q] = len(L)
            if len(L) == delta:
                supports.append(L)
        m = (delta - 1) // 2
        uncovered = [q for q in range(n) if weight[q] > delta]
        assert min(weight.values()) == delta
        overlaps = sorted({len(L & set(f)) for L in supports for f in faces})
        face_ok = all(any(len(L & set(f)) == 2 for L in supports) for f in faces)
        print(f"delta={delta:2d} n={n:3d} faces={len(faces):3d} qubits in no minimum-weight support={len(uncovered)}"
              f" (m(m-1)/2={m * (m - 1) // 2}) face overlaps of these supports={overlaps}"
              f" every face meets one in two qubits={face_ok}", flush=True)
        ok &= face_ok and overlaps == [0, 2]
    print("ALL OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
