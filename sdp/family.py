"""Which witness families suffice?  Conditioning states = all products of stabilizer states on at most
`maxblock` qubits (exact small sets, all multisets of blocks), at chosen levels."""
import sys, itertools, math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import t_cut
from t_cut import contraction_matrix
from t_lp import weight_dists_exact


def products(R, maxblock):
    small = {r: [tuple(np.round(m, 12)) for m in weight_dists_exact(r)] for r in range(1, maxblock + 1)}
    out = {}
    def rec(left, minr, mini, m):
        if left == 0:
            out.setdefault(tuple(np.round(m, 10)), m); return
        for r in range(minr, min(maxblock, left) + 1):
            for i in range(mini if r == minr else 0, len(small[r])):
                rec(left - r, r, i, np.convolve(m, small[r][i]))
    rec(R, 1, 0, np.array([1.0]))
    return np.array(list(out.values()))


def solve(n, d, levels, maxblock):
    blocks = []
    for k in levels:
        if k > n: continue
        ms = products(n - k, maxblock) if n > k else np.array([[1.0]])
        blocks.append((t_cut.facets(k), np.array([contraction_matrix(n, k, mv) for mv in ms])))
    m = gp.Model(); m.Params.OutputFlag = 0
    x = m.addMVar((n + 1) * 2, lb=0.0)
    for v in range(min(d, n + 1)): m.addConstr(x[v * 2 + 1] == 0)
    m.addConstr(x[0] == 1); m.setObjective(x.sum(), GRB.MINIMIZE)
    for _ in range(5000):
        m.optimize()
        if m.Status != GRB.OPTIMAL: return False
        lam = x.X; new = 0
        for F, As in blocks:
            V = np.einsum("fr,mrc,c->mf", F, As, lam)
            bad = np.argwhere(V < -1e-9)
            for i in np.argsort(V[bad[:, 0], bad[:, 1]])[:500]:
                mi, fi = bad[i]; m.addConstr((F[fi] @ As[mi]) @ x >= 0); new += 1
        if new == 0: return True
    raise RuntimeError


if __name__ == "__main__":
    levels = tuple(int(a) for a in sys.argv[1].split(","))
    maxblock = int(sys.argv[2])
    ds = [int(a) for a in sys.argv[3].split(",")]
    for d in ds:
        n = d
        while not solve(n, d, levels, maxblock): n += 1
        print(f"levels={levels} blocks<={maxblock}: d={d} first n not excluded = {n}", flush=True)
