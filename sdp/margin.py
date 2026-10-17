"""max lam[0,0] s.t. sum(lam) = 1 and the level-1 constraints with product conditioning states."""
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import t_cut
from t_cut import contraction_matrix
from family import products

def margin(n, d, levels=(1,), maxblock=2, solver_tol=1e-9):
    m = gp.Model(); m.Params.OutputFlag = 0
    m.Params.FeasibilityTol = solver_tol; m.Params.OptimalityTol = solver_tol
    m.Params.NumericFocus = 3
    x = m.addMVar((n + 1) * 2, lb=0.0)
    for v in range(d): m.addConstr(x[v * 2 + 1] == 0)
    m.addConstr(x.sum() == 1)
    for k in levels:
        if k > n: continue
        F = t_cut.facets(k)
        for mv in (products(n - k, maxblock) if n > k else np.array([[1.0]])):
            A = contraction_matrix(n, k, mv)
            m.addConstr((F @ A) @ x >= 0)
    m.setObjective(x[0], GRB.MAXIMIZE)
    m.optimize()
    return m.ObjVal if m.Status == GRB.OPTIMAL else None, (x.X if m.Status == GRB.OPTIMAL else None)

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    for n, d in [(3, 2), (4, 2), (5, 3), (6, 3), (7, 4), (8, 4)]:
        val, x = margin(n, d)
        print(f"n={n} d={d}: max lam00 = {val}", flush=True)
