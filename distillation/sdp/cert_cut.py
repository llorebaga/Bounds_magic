"""Farkas certificate from the cutting-plane LP (levels 1-3) for one T state."""
import sys, math, pickle
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from t_cut import facets, contraction_matrix
from t_lp import conditioning_set


def run(n, d, levels, n_random=2000, seed=0, max_rounds=2000):
    rng = np.random.default_rng(seed)
    blocks = []
    for k in levels:
        R = n - k
        ms = conditioning_set(R, rng, n_random) if R > 0 else np.array([[1.0]])
        blocks.append((k, facets(k), ms, np.array([contraction_matrix(n, k, mv) for mv in ms])))
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.InfUnbdInfo = 1
    x = m.addMVar((n + 1) * 2, lb=0.0)
    for v in range(min(d, n + 1)):
        m.addConstr(x[v * 2 + 1] == 0)
    m.addConstr(x[0] == 1)
    m.setObjective(x.sum(), GRB.MINIMIZE)
    cuts = []
    for rnd in range(max_rounds):
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            break
        lam = x.X; new = 0
        for bi, (k, F, ms, As) in enumerate(blocks):
            V = np.einsum("fr,mrc,c->mf", F, As, lam)
            bad = np.argwhere(V < -1e-9)
            for i in np.argsort(V[bad[:, 0], bad[:, 1]])[:500]:
                mi, fi = bad[i]
                c = m.addConstr((F[fi] @ As[mi]) @ x >= 0)
                cuts.append((bi, mi, fi, c)); new += 1
        if new == 0:
            return None
    used = [(bi, mi, fi, -c.FarkasDual) for bi, mi, fi, c in cuts if abs(c.FarkasDual) > 1e-10]
    return blocks, used


if __name__ == "__main__":
    n, d = int(sys.argv[1]), int(sys.argv[2])
    res = run(n, d, (1, 2, 3))
    if res is None:
        print("feasible"); sys.exit()
    blocks, used = res
    np.set_printoptions(precision=4, suppress=True)
    print(f"n={n} d={d}: certificate uses {len(used)} constraints")
    for bi, mi, fi, y in sorted(used, key=lambda t: -t[3]):
        k, F, ms, _ = blocks[bi]
        print(f"  level {k}  y={y:.4g}  m(s)={ms[mi]}")
    pickle.dump({"used": [(blocks[bi][0], blocks[bi][2][mi], blocks[bi][1][fi], y) for bi, mi, fi, y in used]},
                open(f"cert_T_n{n}_d{d}.pkl", "wb"))
