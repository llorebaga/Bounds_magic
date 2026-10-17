"""Leading-coefficient minimization for distance-3 phase-polynomial factories (extends tri_mip.py).

Same model as tri_mip.build (generalized triorthogonality, target trilinear form, distinct nonzero check
columns so that every fault of weight <= 2 is detected). Added objective: the number of harmful weight-3
fault patterns, i.e. triples {i,j,k} whose check parts cancel (G0 e = 0) while their logical parts do not
(G1 e != 0). Parities are linearized exactly with integer slacks: a+b+c = z + 2w.
Every returned design must still be certified with verify_tri2.py.
Usage: python tri_mip_opt.py TARGET n s timelimit [warmstart.json]
"""
import sys, os, json, itertools, time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from tri_mip import TARGETS, build

target, n, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tl = float(sys.argv[4]); warm = sys.argv[5] if len(sys.argv) > 5 else None
k = TARGETS[target][0]; N = k + s
m, G, k, N = build(target, n, s, 3, timelimit=tl, threads=int(os.environ.get('THREADS', 16)))
m.Params.OutputFlag = 1; m.Params.MIPFocus = 1

def parity(rowvars):
    z = m.addVar(vtype=GRB.BINARY); w = m.addVar(vtype=GRB.INTEGER, lb=0, ub=1)
    m.addConstr(gp.quicksum(rowvars) == z + 2 * w)
    return z

harm = []
for (i, j, l) in itertools.combinations(range(n), 3):
    zc = [parity([G[r, i], G[r, j], G[r, l]]) for r in range(k, N)]
    y = m.addVar(vtype=GRB.BINARY)            # y = 1 iff all check parities vanish
    m.addConstr(y >= 1 - gp.quicksum(zc))
    for z in zc: m.addConstr(y <= 1 - z)
    zl = [parity([G[r, i], G[r, j], G[r, l]]) for r in range(k)]
    h = m.addVar(vtype=GRB.BINARY)
    for z in zl: m.addConstr(h >= y + z - 1)
    harm.append(h)
m.setObjective(gp.quicksum(harm), GRB.MINIMIZE)

if warm:
    rows = json.load(open(warm))["matrix"]
    M0 = np.array([[int(c) for c in r] if isinstance(r, str) else r for r in rows])
    # warm start must respect the column ordering constraint of build(): sort columns by check value
    order = sorted(range(n), key=lambda j: sum(int(M0[r, j]) << (r - k) for r in range(k, N)))
    M0 = M0[:, order]
    for a in range(N):
        for j in range(n): G[a, j].Start = int(M0[a, j])

t0 = time.time(); m.optimize()
res = {"target": target, "n": n, "s": s, "status": int(m.Status), "sec": round(time.time() - t0, 1)}
if m.SolCount:
    M = np.array([[int(round(G[a, j].X)) for j in range(n)] for a in range(N)])
    res.update({"harmful_weight3": int(round(m.ObjVal)), "bound": m.ObjBound, "G": M.tolist()})
out = f"../results/mip/opt_{target}_n{n}_s{s}.json"
json.dump(res, open(out, "w"), indent=1)
print({k_: v for k_, v in res.items() if k_ != "G"})
