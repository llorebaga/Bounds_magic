"""Robustness: products of exactly enumerated states only, and facets weakened by delta."""
import sys, math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import t_cut
from t_cut import contraction_matrix
from t_lp import weight_dists_exact


def product_set(R, rng, n_samples=4000):
    small = {r: weight_dists_exact(r) for r in range(1, 6)}
    if R == 0:
        return np.array([[1.0]])
    if R <= 5:
        return small[R]
    out = set()
    ms = []
    for _ in range(n_samples):
        left, parts = R, []
        while left > 0:
            r = int(rng.integers(1, min(5, left) + 1)); parts.append(small[r][rng.integers(len(small[r]))]); left -= r
        m = np.array([1.0])
        for p in parts:
            m = np.convolve(m, p)
        key = tuple(np.round(m, 10))
        if key not in out:
            out.add(key); ms.append(m)
    return np.array(ms)


def solve(n, d, levels, delta, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for k in levels:
        F = t_cut.facets(k) + delta          # f.g >= 0 becomes f.g + delta*sum(g) >= 0: weaker, still valid
        ms = product_set(n - k, rng)
        blocks.append((F, np.array([contraction_matrix(n, k, mv) for mv in ms])))
    m = gp.Model(); m.Params.OutputFlag = 0
    x = m.addMVar((n + 1) * 2, lb=0.0)
    for v in range(min(d, n + 1)):
        m.addConstr(x[v * 2 + 1] == 0)
    m.addConstr(x[0] == 1)
    m.setObjective(x.sum(), GRB.MINIMIZE)
    for _ in range(3000):
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return False
        lam = x.X; new = 0
        for F, As in blocks:
            V = np.einsum("fr,mrc,c->mf", F, As, lam)
            bad = np.argwhere(V < -1e-9)
            for i in np.argsort(V[bad[:, 0], bad[:, 1]])[:500]:
                mi, fi = bad[i]; m.addConstr((F[fi] @ As[mi]) @ x >= 0); new += 1
        if new == 0:
            return True
    raise RuntimeError


if __name__ == "__main__":
    for (n, d) in [(3, 2), (5, 3), (7, 4), (9, 5)]:
        for delta in [0.0, 1e-6, 1e-4]:
            f = solve(n, d, (1, 2, 3), delta)
            print(f"n={n} d={d} delta={delta:g}: {'feasible' if f else 'INFEASIBLE'}", flush=True)
