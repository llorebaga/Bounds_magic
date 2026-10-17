import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import t_cut
from t_cut import contraction_matrix
from family import products

def run(n, d):
    F = t_cut.facets(1)
    ms = products(n - 1, 2)
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.NumericFocus = 3
    x = m.addMVar((n + 1) * 2, lb=0.0)
    for v in range(d): m.addConstr(x[v * 2 + 1] == 0)
    m.addConstr(x.sum() == 1)
    cons = []
    for mi, mv in enumerate(ms):
        A = contraction_matrix(n, 1, mv)
        for fi in range(len(F)):
            cons.append((mi, fi, m.addConstr((F[fi] @ A) @ x >= 0)))
    m.setObjective(x[0], GRB.MAXIMIZE)
    m.optimize()
    print(f"n={n} d={d}: max lam00 = {m.ObjVal:.3g}")
    np.set_printoptions(precision=4, suppress=True)
    for mi, fi, c in sorted(cons, key=lambda t: t[2].Pi):
        if abs(c.Pi) > 1e-8:
            print(f"   dual={c.Pi:.4g} facet f{fi} m={ms[mi]}")
for n, d in [(3, 2), (5, 3), (7, 4)]:
    run(n, d)
