"""Gurobi MIP for generalized triorthogonal distillation protocols (Haah-Hastings / Wills-Jain-Singh form),
with an exact dense verifier of the resulting single-branch protocol.

Matrix G (N x n) = [G1 (k logical rows); G0 (s stabiliser rows)], columns = input T states.
Constraints (all mod 2): every overlap <g_a,g_b,g_c> (a<=b<=c, repetitions allowed) involving a stabiliser
row vanishes; logical overlaps equal the target trilinear form tau (tau_aaa: T_a, tau_aab: CS_ab, tau_abc: CCZ).
Distance d_Z = min{|v| : G0 v = 0, G1 v != 0} >= d is imposed for d <= 3 directly on columns.
Usage: python tri_mip.py TARGET n s d [timelimit] [threads]
TARGET: name in TARGETS below.
"""
import sys, json, time, itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# target trilinear forms: dict of index tuples (sorted, 0-based, with repetition) -> 1
TARGETS = {
    "ccz": (3, {(0, 1, 2)}),
    "cs12cs13": (3, {(0, 0, 1), (0, 0, 2)}),               # Clifford-equivalent to |CCZ> (cliff_bfs)
    "t": (1, {(0, 0, 0)}),
    "tt": (2, {(0, 0, 0), (1, 1, 1)}),
    "cs": (2, {(0, 0, 1)}),
    "ccz2": (6, {(0, 1, 2), (3, 4, 5)}),
    "ccz_cs12cs13": (6, {(0, 1, 2), (3, 3, 4), (3, 3, 5)}),
    "cs12cs13_2": (6, {(0, 0, 1), (0, 0, 2), (3, 3, 4), (3, 3, 5)}),
}

def build(target, n, s, d, timelimit=600, threads=16, extra_logical_free=False, seed=0):
    k, tau = TARGETS[target]; N = k + s
    m = gp.Model("tri"); m.Params.OutputFlag = 0; m.Params.TimeLimit = timelimit; m.Params.Threads = threads
    m.Params.Seed = seed
    G = m.addVars(N, n, vtype=GRB.BINARY, name="G")
    # products
    def prod(rows, j, cache={}):
        key = (rows, j)
        if key in cache: return cache[key]
        if len(rows) == 1: return G[rows[0], j]
        v = m.addVar(vtype=GRB.BINARY)
        for r in rows: m.addConstr(v <= G[r, j])
        m.addConstr(v >= gp.quicksum(G[r, j] for r in rows) - (len(rows) - 1))
        cache[key] = v; return v
    for size in (1, 2, 3):
        for rows in itertools.combinations(range(N), size):
            if any(r >= k for r in rows): val = 0
            elif size == 1: val = int((rows[0],) * 3 in tau)
            elif size == 2: val = int((rows[0], rows[0], rows[1]) in tau or (rows[0], rows[1], rows[1]) in tau)
            else: val = int(rows in tau)
            t = m.addVar(vtype=GRB.INTEGER, lb=0, ub=n)
            m.addConstr(gp.quicksum(prod(rows, j) for j in range(n)) == 2 * t + val)
    # no zero column
    for j in range(n): m.addConstr(gp.quicksum(G[a, j] for a in range(N)) >= 1)
    if d >= 2:  # a column with zero stabiliser part must be zero
        for j in range(n):
            for a in range(k): m.addConstr(G[a, j] <= gp.quicksum(G[r, j] for r in range(k, N)))
    if d >= 3:  # distinct nonzero stabiliser parts (no duplicate columns); then logical parts free
        val = [gp.quicksum((1 << (r - k)) * G[r, j] for r in range(k, N)) for j in range(n)]
        for j in range(n - 1): m.addConstr(val[j] + 1 <= val[j + 1])
    else:  # symmetry breaking: lexicographic by full column value
        val = [gp.quicksum((1 << a) * G[a, j] for a in range(N)) for j in range(n)]
        for j in range(n - 1): m.addConstr(val[j] <= val[j + 1])
    if d >= 4: raise NotImplementedError
    return m, G, k, N

def solve(target, n, s, d, timelimit=600, threads=int(__import__("os").environ.get("THREADS", 16))):
    m, G, k, N = build(target, n, s, d, timelimit, threads)
    t0 = time.time(); m.optimize()
    st = {GRB.OPTIMAL: "feasible", GRB.INFEASIBLE: "infeasible", GRB.TIME_LIMIT: "timeout"}.get(m.Status, str(m.Status))
    if m.SolCount > 0: st = "feasible"
    M = None
    if st == "feasible":
        M = np.array([[int(round(G[a, j].X)) for j in range(n)] for a in range(N)], dtype=np.int64)
    return st, M, time.time() - t0, k

# ---------------- exact checks of a matrix ----------------
def overlaps_ok(M, k, target):
    N, n = M.shape; tau = TARGETS[target][1]
    for size in (1, 2, 3):
        for rows in itertools.combinations(range(N), size):
            ov = int(np.prod(M[list(rows)], axis=0).sum()) % 2
            if any(r >= k for r in rows):
                if ov: return False
            else:
                key = (rows * 3)[:3] if size == 1 else ((rows[0], rows[0], rows[1]) if size == 2 else rows)
                want = 1 if (key in tau or (size == 2 and (rows[0], rows[1], rows[1]) in tau)) else 0
                if ov != want: return False
    return True

def distance(M, k, dmax=6):
    N, n = M.shape
    G0 = M[k:]; G1 = M[:k]
    cols0 = [int(sum(int(G0[r, j]) << r for r in range(N - k))) for j in range(n)]
    cols1 = [int(sum(int(G1[r, j]) << r for r in range(k))) for j in range(n)]
    for w in range(1, dmax + 1):
        for S in itertools.combinations(range(n), w):
            s0 = 0; s1 = 0
            for j in S: s0 ^= cols0[j]; s1 ^= cols1[j]
            if s0 == 0 and s1 != 0: return w
    return dmax + 1

if __name__ == "__main__":
    target, n, s, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    tl = float(sys.argv[5]) if len(sys.argv) > 5 else 600
    st, M, sec, k = solve(target, n, s, d, tl)
    rec = {"target": target, "n": n, "s": s, "d": d, "status": st, "seconds": round(sec, 1)}
    if M is not None:
        rec["overlaps_ok"] = overlaps_ok(M, k, target); rec["dZ"] = distance(M, k, max(d, 3))
        rec["matrix"] = ["".join(map(str, row)) for row in M]
    print(json.dumps(rec))
