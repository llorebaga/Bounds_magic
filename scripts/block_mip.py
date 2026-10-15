"""MIP for two-level factories: an outer phase-polynomial (generalized triorthogonal) code whose inputs are
raw T states and CCZ *blocks*, each block being a CCZ state prepared by an inner stabilizer protocol.

Registers v in F2^N (k logical rows, s check rows).
  T column g (cost 1 T, fault cost 1):   phase w^{[g.v]},  fault -> Z^g
  CCZ block (a,b,c) of type t:           phase (-1)^{[a.v][b.v][c.v]} via gate teleportation of a CCZ state,
                                          fault (after the CCZ-symmetry twirl) -> Z^{l1 a + l2 b + l3 c}, l != 0
     type 'J': Jones 4T -> CCZ, order 1: T cost 4, fault cost 1 (any single T fault spoils the block)
     type 'C': cube 8T -> CCZ, order 2:  T cost 8, fault cost 2 (a block is spoiled only by >= 2 T faults)
Phase polynomial mod 8: T columns contribute as in generalized triorthogonality (singles, pairs, triples mod 2);
blocks contribute 4*(a.v)(b.v)(c.v), i.e. only to the trilinear (cubic) form mod 2 through the symmetrized
tensor sum_{perm} a_i b_j c_k on distinct rows (lower-order parts are Z/CZ Cliffords).
Constraints: every overlap involving a check row even; logical cubic/pair/single forms = target.
Order D of the composite: every combination of faulty units with total fault cost < D must have zero check
syndrome only if it has zero logical part.  Objective: minimize total T cost.
Usage: python block_mip.py TARGET D s nT nJ nC [timelimit]
"""
import sys, itertools, json, time, os
import gurobipy as gp
from gurobipy import GRB

TARGETS = {"ccz": (3, {(0, 1, 2)}), "cs12cs13": (3, {(0, 0, 1), (0, 0, 2)})}

def build(target, D, s, nT, nJ, nC, tl=600, threads=8):
    k, tau = TARGETS[target]; N = k + s
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.TimeLimit = tl; m.Params.Threads = threads
    G = m.addVars(N, nT, vtype=GRB.BINARY)                       # T columns
    nB = nJ + nC; btype = ['J'] * nJ + ['C'] * nC
    A = m.addVars(nB, 3, N, vtype=GRB.BINARY)                    # block forms
    m.update()
    cache = {}
    def prod(lits):
        byidx = {v.index: v for v in lits}
        key = tuple(sorted(byidx)); lits = [byidx[i] for i in key]
        if len(lits) == 1: return lits[0]
        if key in cache: return cache[key]
        y = m.addVar(vtype=GRB.BINARY)
        for l in lits: m.addConstr(y <= l)
        m.addConstr(y >= gp.quicksum(lits) - (len(lits) - 1))
        cache[key] = y; return y
    # overlaps
    for size in (1, 2, 3):
        for rows in itertools.combinations(range(N), size):
            terms = [prod([G[r, j] for r in rows]) for j in range(nT)]
            if size == 3:
                for b in range(nB):
                    for perm in itertools.permutations(rows):
                        terms.append(prod([A[b, 0, perm[0]], A[b, 1, perm[1]], A[b, 2, perm[2]]]))
            if any(r >= k for r in rows): val = 0
            elif size == 1: val = int((rows[0],) * 3 in tau)
            elif size == 2: val = int((rows[0], rows[0], rows[1]) in tau or (rows[0], rows[1], rows[1]) in tau)
            else: val = int(rows in tau)
            if not terms:
                if val: return None
                continue
            t = m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(terms))
            m.addConstr(gp.quicksum(terms) == 2 * t + val)
    # fault units: (cost, list of candidate Z-vectors as lists of variable-vectors to XOR)
    units = [(1, [[('G', j)]]) for j in range(nT)]
    for b in range(nB):
        cost = 1 if btype[b] == 'J' else 2
        pats = []
        for l in range(1, 8):
            pats.append([('A', b, q) for q in range(3) if (l >> q) & 1])
        units.append((cost, pats))
    def vec(item, r):
        return G[r, item[1]] if item[0] == 'G' else A[item[1], item[2], r]
    ncons = 0
    for size in range(1, D):
        for combo in itertools.combinations(range(len(units)), size):
            if sum(units[u][0] for u in combo) >= D: continue
            for choice in itertools.product(*[units[u][1] for u in combo]):
                items = [it for c in choice for it in c]
                bits = []
                for r in range(N):
                    xs = [vec(it, r) for it in items]
                    if len(xs) == 1: bits.append(xs[0]); continue
                    z = m.addVar(vtype=GRB.BINARY); q = m.addVar(vtype=GRB.INTEGER, lb=0, ub=len(xs))
                    m.addConstr(gp.quicksum(xs) == 2 * q + z); bits.append(z)
                chk = gp.quicksum(bits[r] for r in range(k, N))
                for r in range(k): m.addConstr(bits[r] <= chk)
                ncons += 1
    # nondegeneracy: every T column and every block form has a nonzero check part (else wasteful/undetected)
    for j in range(nT): m.addConstr(gp.quicksum(G[r, j] for r in range(k, N)) >= 1)
    # symmetry breaking among T columns
    for j in range(nT - 1):
        m.addConstr(gp.quicksum((1 << r) * G[r, j] for r in range(N)) <= gp.quicksum((1 << r) * G[r, j + 1] for r in range(N)))
    return m, G, A, k, N, btype, ncons

if __name__ == "__main__":
    target, D, s, nT, nJ, nC = sys.argv[1], *map(int, sys.argv[2:7])
    tl = float(sys.argv[7]) if len(sys.argv) > 7 else 600
    t0 = time.time()
    out = build(target, D, s, nT, nJ, nC, tl, int(os.environ.get("THREADS", 8)))
    rec = {"target": target, "D": D, "s": s, "nT": nT, "nJ": nJ, "nC": nC, "Tcost": nT + 4 * nJ + 8 * nC}
    if out is None: rec["status"] = "trivially infeasible"; print(json.dumps(rec)); sys.exit()
    m, G, A, k, N, btype, ncons = out
    m.optimize()
    rec["fault_constraints"] = ncons; rec["sec"] = round(time.time() - t0, 1)
    rec["status"] = "feasible" if m.SolCount else {GRB.INFEASIBLE: "infeasible", GRB.TIME_LIMIT: "timeout"}.get(m.Status, str(m.Status))
    if m.SolCount:
        rec["Tcols"] = ["".join(str(int(round(G[r, j].X))) for r in range(N)) for j in range(nT)]
        rec["blocks"] = [[btype[b]] + ["".join(str(int(round(A[b, q, r].X))) for r in range(N)) for q in range(3)] for b in range(len(btype))]
    print(json.dumps(rec))
