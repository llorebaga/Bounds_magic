"""Level-k LP for T/CS/CCZ with a separation oracle instead of facets (robust in higher dimension).

For current lam and each conditioning state s, solve  min f.g_s  s.t.  f.G_i >= 0 for all generators,
-1 <= f <= 1. If the minimum is negative, f.g >= 0 is a valid inequality violated by g_s; it is
weakened slightly (f + eps*1, valid since every generator has total mass 1) and added as a cut.
"""
import sys, time, math
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gen_lp import TARGETS, generators, contraction
from t_lp import conditioning_set

EPS_VALID = 1e-12


class Separator:
    def __init__(self, G):
        self.G = G
        self.m = gp.Model(); self.m.Params.OutputFlag = 0
        self.f = self.m.addMVar(G.shape[1], lb=-1.0, ub=1.0)
        self.m.addConstr(G @ self.f >= 0)

    def cut(self, g):
        self.m.setObjective(g @ self.f, GRB.MINIMIZE)
        self.m.optimize()
        if self.m.ObjVal < -1e-8:
            f = self.f.X + EPS_VALID
            if (self.G @ f).min() >= 0:
                return f
            f = f + (-(self.G @ f).min() + 1e-12)
            assert (self.G @ f).min() >= 0
            return f
        return None


def solve(target, n, d, levels, n_random=1500, seed=0, verbose=False):
    r, _, rational = TARGETS[target]
    rng = np.random.default_rng(seed)
    blocks = []
    for k in levels:
        if k > n:
            continue
        ms = conditioning_set(n - k, rng, n_random) if n > k else np.array([[1.0]])
        blocks.append((Separator(generators(target, k)), [contraction(n, k, r, mv) for mv in ms]))
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.NumericFocus = 3
    x = m.addMVar((n + 1) * (r + 1), lb=0.0)
    for v in range(n + 1):
        if v < d or (rational and v > n - d):
            for w in range(1, r + 1):
                m.addConstr(x[v * (r + 1) + w] == 0)
    m.addConstr(x.sum() == 1)
    m.setObjective(x[0], GRB.MAXIMIZE)
    for rnd in range(2000):
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return 0.0
        lam = x.X; new = 0
        for sep, As in blocks:
            for A in As:
                f = sep.cut(A @ lam)
                if f is not None:
                    m.addConstr((f @ A) @ x >= 0); new += 1
        if verbose:
            print(f"   round {rnd}: obj {m.ObjVal:.3g}, {new} cuts", flush=True)
        if new == 0:
            return m.ObjVal
    raise RuntimeError


if __name__ == "__main__":
    target = sys.argv[1]
    levels = tuple(int(a) for a in sys.argv[2].split(","))
    ds = [int(a) for a in sys.argv[3].split(",")]
    for k in levels:
        t0 = time.time(); G = generators(target, k)
        print(f"{target} k={k}: {len(G)} generators [{time.time()-t0:.0f}s]", flush=True)
    for d in ds:
        out = []
        for n in range(d, 4 * d + 3):
            t0 = time.time()
            val = solve(target, n, d, levels)
            out.append(f"{n}:{val:.2g}")
            print(f"  {target} d={d} n={n}: max lam00 = {val:.3g} [{time.time()-t0:.0f}s]", flush=True)
            if val > 1e-6:
                break
        print(f"{target} d={d} levels={levels}: first n not excluded = {n}", flush=True)
