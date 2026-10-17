import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import t_cut
from t_cut import contraction_matrix
from family import products

n, d = int(sys.argv[1]), int(sys.argv[2])
F = t_cut.facets(1)
ms = products(n - 1, 2)
m = gp.Model(); m.Params.OutputFlag = 0; m.Params.InfUnbdInfo = 1
x = m.addMVar((n + 1) * 2, lb=0.0)
for v in range(d): m.addConstr(x[v * 2 + 1] == 0)
m.addConstr(x[0] == 1)
cons = []
for mi, mv in enumerate(ms):
    A = contraction_matrix(n, 1, mv)
    for fi in range(len(F)):
        cons.append((mi, fi, m.addConstr((F[fi] @ A) @ x >= 0)))
m.optimize()
print("status", m.Status, "(3 = infeasible)")
np.set_printoptions(precision=4, suppress=True)
for mi, fi, c in sorted(cons, key=lambda t: t[2].FarkasDual):
    if abs(c.FarkasDual) > 1e-9:
        print(f"  y={-c.FarkasDual:.4g} facet {fi} m={ms[mi]}")
