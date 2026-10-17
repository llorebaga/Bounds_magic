"""Checks for the analytic proof of n_min(T,d) >= 2d (d >= 2).

1. Lemma (exact, all 60 two-qubit stabilizer states): with g_{jb} = |<phi_j phi_b|psi>|^2 and
   s = 3 - 2 sqrt2,
       g01 - s^2 g10 >= s |g00 - g11|   and   g10 - s^2 g01 >= s |g00 - g11|.
2. The proof uses only: harmless weights (B_v = 0, v < d), the Galois zero pattern (A_v = 0, v >= d)
   at n = 2d-1, the witnesses singlet^{d-1-t} (x) |+>^{2t} and singlet^{d-1-t} (x) |->^{2t}
   (t = 0..d-1), and the two inequalities of the lemma. We confirm with an LP that these alone force
   A_0 = 0 (max A_0 subject to sum = 1 is 0), and replay the induction symbolically.
"""
import itertools
import numpy as np
import sympy as sp
import gurobipy as gp
from gurobipy import GRB

import exact_cert as EC

r2 = sp.sqrt(2)
s = 3 - 2 * r2


def lemma_exact():
    states = EC.stab_states_exact(2)
    assert len(states) == 60
    worst = []
    for st in states:
        g = EC.t_basis_dist(st, 2)          # order 00, 01, 10, 11 = (j, b)
        g00, g01, g10, g11 = [sp.radsimp(x) for x in g]
        for lhs in (g01 - s ** 2 * g10, g10 - s ** 2 * g01):
            for sign in (1, -1):
                val = sp.radsimp(lhs - sign * s * (g00 - g11))
                assert val >= 0, (st, g, val)
                worst.append(val)
    return len(states), min(worst)


def induction_symbolic(d):
    """Replay the induction with sympy: returns the forced relations."""
    n = 2 * d - 1
    A = sp.symbols(f"A0:{n + 1}", nonnegative=True)
    B = sp.symbols(f"B0:{n + 1}", nonnegative=True)
    known = {}
    for v in range(d):
        known[B[v]] = 0
    for v in range(d, n + 1):
        known[A[v]] = 0              # Galois at n = 2d-1
    def g(M):
        a0 = sum(M[v] * A[v] for v in range(len(M)))
        b0 = sum(M[v] * B[v] for v in range(len(M)))
        a1 = sum(M[v] * A[v + 1] for v in range(len(M)))
        b1 = sum(M[v] * B[v + 1] for v in range(len(M)))
        return [sp.expand(x.subs(known)) for x in (a0, b0, a1, b1)]
    for t in range(d):
        p = d - 1 - t
        plus = sp.Poly(sp.Symbol("x") ** p * (1 + s * sp.Symbol("x")) ** (2 * t), sp.Symbol("x")).all_coeffs()[::-1]
        minus = sp.Poly(sp.Symbol("x") ** p * (s + sp.Symbol("x")) ** (2 * t), sp.Symbol("x")).all_coeffs()[::-1]
        a0, b0, a1, b1 = g(plus)
        assert sp.simplify(b0 - s ** 2 * a1) == 0          # LHS of inequality (iii) vanishes
        rel1 = sp.expand(a0 - b1)                          # hence a0 = b1
        if t == 0:
            sol = sp.solve(rel1, A[d - 1], dict=True)[0]
            known.update(sol)
            continue
        a0m, b0m, a1m, b1m = g(minus)
        assert sp.simplify(a1m - s ** 2 * b0m) == 0        # LHS of inequality (iv) vanishes
        rel2 = sp.expand(a0m - b1m)                        # hence a0 = b1
        sol = sp.solve([rel1, rel2], [A[p], B[d + t]], dict=True)
        assert sol == [{A[p]: 0, B[d + t]: 0}], (t, sol)
        known[A[p]] = 0
        known[B[d + t]] = 0
    return known[A[0]] == 0


def lp_check(d):
    n = 2 * d - 1
    sf = float(s)
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.NumericFocus = 3
    A = m.addVars(n + 2, lb=0.0); B = m.addVars(n + 2, lb=0.0)
    for v in range(d): m.addConstr(B[v] == 0)
    for v in range(d, n + 2): m.addConstr(A[v] == 0)
    m.addConstr(B[n + 1] == 0)
    m.addConstr(gp.quicksum(A[v] + B[v] for v in range(n + 1)) == 1)
    for t in range(d):
        p = d - 1 - t
        for f in ([1, sf], [sf, 1]):
            M = np.zeros(1); M[0] = 1
            M = np.convolve(np.eye(p + 1)[p], np.poly1d([1])(0) * np.ones(1))
            P = np.zeros(p + 1); P[p] = 1
            for _ in range(2 * t): P = np.convolve(P, f)
            a0 = gp.quicksum(P[v] * A[v] for v in range(len(P)))
            b0 = gp.quicksum(P[v] * B[v] for v in range(len(P)))
            a1 = gp.quicksum(P[v] * A[v + 1] for v in range(len(P)))
            b1 = gp.quicksum(P[v] * B[v + 1] for v in range(len(P)))
            for sign in (1, -1):
                m.addConstr(b0 - sf ** 2 * a1 >= sign * sf * (a0 - b1))
                m.addConstr(a1 - sf ** 2 * b0 >= sign * sf * (a0 - b1))
    m.setObjective(A[0], GRB.MAXIMIZE); m.optimize()
    return m.ObjVal


if __name__ == "__main__":
    k, w = lemma_exact()
    print(f"Lemma verified exactly on all {k} two-qubit stabilizer states (min slack {sp.nsimplify(w)})")
    for d in range(2, 9):
        print(f"d={d}: symbolic induction forces A_0 = 0: {induction_symbolic(d)}", flush=True)
    print("LP with only the proof's witnesses and inequalities, max A_0:",
          [f"d={d}:{lp_check(d):.1g}" for d in range(2, 15)])
