"""Does the structured family x^p (1+sx)^{2t}, x^p (s+x)^{2t}, x^p (1+x^2)^t  (p+t = d-1) suffice?"""
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import t_cut
from t_cut import contraction_matrix

s = 3 - 2 * 2 ** 0.5
def polys(d, kinds):
    out = []
    for p in range(d):
        t = d - 1 - p
        base = np.zeros(p + 1); base[p] = 1.0
        for kind in kinds:
            f = {"plus": [1, s], "minus": [s, 1], "bell": [1, 0, 1]}[kind]
            P = base.copy()
            for _ in range(2 * t if kind != "bell" else t):
                P = np.convolve(P, f)
            P = np.pad(P, (0, 2 * d - 1 - len(P)))
            out.append(P / P.sum())
    return out

def margin(d, kinds):
    n = 2 * d - 1
    F = t_cut.facets(1)
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.NumericFocus = 3
    x = m.addMVar(2 * (n + 1), lb=0.0)
    for v in range(d): m.addConstr(x[v * 2 + 1] == 0)
    m.addConstr(x.sum() == 1)
    for M in polys(d, kinds):
        m.addConstr((F @ contraction_matrix(n, 1, M)) @ x >= 0)
    m.setObjective(x[0], GRB.MAXIMIZE); m.optimize()
    return m.ObjVal

for kinds in [("plus",), ("plus", "minus"), ("plus", "bell"), ("plus", "minus", "bell")]:
    print(kinds, [f"d={d}:{margin(d, kinds):.2g}" for d in range(2, 10)], flush=True)
