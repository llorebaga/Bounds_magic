"""Extract the Farkas certificate of an infeasible level-k LP for one T state.

Prints the conditioning states (their T-basis weight distributions m) and facets that carry
positive multipliers, i.e. the stabilizer witnesses that prove the bound.
"""
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from t_cut import facets, contraction_matrix
from t_lp import conditioning_set


def certificate(n, d, k, n_random=600, rng=None):
    rng = rng or np.random.default_rng(0)
    F = facets(k)
    ms = conditioning_set(n - k, rng, n_random) if n > k else np.array([[1.0]])
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.InfUnbdInfo = 1
    x = m.addMVar((n + 1) * 2, lb=0.0)
    fix = [m.addConstr(x[v * 2 + 1] == 0) for v in range(min(d, n + 1))]
    norm = m.addConstr(x[0] == 1)
    cons = []
    for mi, mv in enumerate(ms):
        A = contraction_matrix(n, k, mv)
        for fi in range(len(F)):
            c = m.addConstr((F[fi] @ A) @ x >= 0)
            cons.append((mi, fi, c))
    m.optimize()
    if m.Status != GRB.INFEASIBLE:
        return None
    used = [(mi, fi, -c.FarkasDual) for mi, fi, c in cons if abs(c.FarkasDual) > 1e-9]
    return ms, F, used


if __name__ == "__main__":
    n, d, k = (int(a) for a in sys.argv[1:4])
    res = certificate(n, d, k)
    if res is None:
        print("feasible: no certificate")
        sys.exit()
    ms, F, used = res
    np.set_printoptions(precision=4, suppress=True)
    print(f"n={n} d={d} k={k}: {len(used)} constraints in the certificate")
    for mi, fi, y in sorted(used, key=lambda t: -abs(t[2])):
        print(f"  y={y:.4g}  m(s)={ms[mi]}  facet={F[fi]}")
