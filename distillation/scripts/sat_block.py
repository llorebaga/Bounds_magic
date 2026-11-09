"""SAT encoding of two-level factories (same model as block_mip.py), solved with CaDiCaL via PySAT.

Outer generalized-triorthogonal phase polynomial on N = k + s registers with
  T columns g          : phase w^{[g.v]}, fault cost 1, fault -> Z^g
  CCZ blocks (a,b,c)   : phase (-1)^{[a.v][b.v][c.v]}, type 'J' (Jones 4T, fault cost 1) or 'C' (cube 8T,
                         fault cost 2); a faulty block applies Z^{l1 a + l2 b + l3 c} for some l != 0.
Constraints: for every set R of <= 3 rows, the parity of the overlap (T columns) plus, for |R| = 3, the
symmetrized block tensor, equals the target coefficient (0 if R contains a check row).  For every
combination of faulty units with total cost < D and every choice of block patterns, the combined Z-vector
must have a nonzero check part or a zero logical part.
Usage: python sat_block.py TARGET D s nT nJ nC [solver] [timeout_s]
Targets: ccz (k=3), cs12cs13 (k=3), ccz2 (k=6, two CCZ on rows 0-2 and 3-5).
"""
import sys, os, itertools, json, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "python"))
from pysat.solvers import Solver

TARGETS = {"ccz": (3, {(0, 1, 2)}), "cs12cs13": (3, {(0, 0, 1), (0, 0, 2)}),
           "ccz2": (6, {(0, 1, 2), (3, 4, 5)}), "cs": (2, {(0, 0, 1)})}

class CNF:
    def __init__(self): self.n = 0; self.cl = []; self.andc = {}
    def new(self): self.n += 1; return self.n
    def AND(self, xs):
        xs = tuple(sorted(set(xs)))
        if len(xs) == 1: return xs[0]
        if xs in self.andc: return self.andc[xs]
        y = self.new()
        for x in xs: self.cl.append([-y, x])
        self.cl.append([y] + [-x for x in xs])
        self.andc[xs] = y; return y
    def XOR2(self, a, b):
        z = self.new()
        self.cl += [[-z, a, b], [-z, -a, -b], [z, -a, b], [z, a, -b]]
        return z
    def XOR(self, xs):
        if not xs: return None
        acc = xs[0]
        for x in xs[1:]: acc = self.XOR2(acc, x)
        return acc

def build(target, D, s, nT, nJ, nC, symbreak=True):
    k, tau = TARGETS[target]; N = k + s
    f = CNF()
    G = [[f.new() for _ in range(nT)] for _ in range(N)]                     # G[r][j]
    nB = nJ + nC; btype = ['J'] * nJ + ['C'] * nC
    A = [[[f.new() for _ in range(N)] for _ in range(3)] for _ in range(nB)]  # A[b][q][r]
    # parity (overlap) constraints
    for size in (1, 2, 3):
        for rows in itertools.combinations(range(N), size):
            terms = [f.AND([G[r][j] for r in rows]) for j in range(nT)]
            if size == 3:
                for b in range(nB):
                    for perm in itertools.permutations(rows):
                        terms.append(f.AND([A[b][0][perm[0]], A[b][1][perm[1]], A[b][2][perm[2]]]))
            if any(r >= k for r in rows): val = 0
            elif size == 1: val = int((rows[0],) * 3 in tau)
            elif size == 2: val = int((rows[0], rows[0], rows[1]) in tau or (rows[0], rows[1], rows[1]) in tau)
            else: val = int(rows in tau)
            x = f.XOR(terms)
            if x is None:
                if val: return None
                continue
            f.cl.append([x] if val else [-x])
    # fault units
    units = [(1, [[('G', j)]]) for j in range(nT)]
    for b in range(nB):
        units.append((1 if btype[b] == 'J' else 2, [[('A', b, q) for q in range(3) if (l >> q) & 1] for l in range(1, 8)]))
    bit = lambda it, r: G[r][it[1]] if it[0] == 'G' else A[it[1]][it[2]][r]
    ncomb = 0
    for size in range(1, D):
        for combo in itertools.combinations(range(len(units)), size):
            if sum(units[u][0] for u in combo) >= D: continue
            for choice in itertools.product(*[units[u][1] for u in combo]):
                items = [it for c in choice for it in c]
                u = [f.XOR([bit(it, r) for it in items]) for r in range(N)]
                chk = [u[r] for r in range(k, N)]
                for r in range(k): f.cl.append([-u[r]] + chk)
                ncomb += 1
    # optional symmetry breaking for construction searches (env FIXB0=1): by GL(s) on check rows and adding
    # check rows to logical rows, block 0 can be brought to the unit vectors on the first three check rows
    # whenever its three check parts are independent (not exhaustive: UNSAT is then not a proof)
    if os.environ.get("FIXB0") == "1" and nB > 0 and s >= 3:
        for q in range(3):
            for r in range(N):
                f.cl.append([A[0][q][r]] if r == k + q else [-A[0][q][r]])
    # every T column nonzero (zero columns are wasteful)
    for j in range(nT): f.cl.append([G[r][j] for r in range(N)])
    if symbreak and nT > 1:  # T columns in strictly increasing order of their (row-0 most significant) values
        for j in range(nT - 1):
            lex_less(f, [G[r][j] for r in range(N)], [G[r][j + 1] for r in range(N)])
    return f, G, A, k, N, btype, ncomb

def lex_less(f, x, y):
    """encode x <_lex y (strict), x[0] most significant"""
    eq_prefix = None
    lt_terms = []
    for i in range(len(x)):
        lt_i = f.new(); f.cl += [[-lt_i, -x[i]], [-lt_i, y[i]]]           # lt_i -> (x_i=0, y_i=1)
        if eq_prefix is not None: f.cl.append([-lt_i, eq_prefix])
        lt_terms.append(lt_i)
        e = f.new()                                                        # e <-> x_i == y_i
        f.cl += [[-e, -x[i], y[i]], [-e, x[i], -y[i]], [e, x[i], y[i]], [e, -x[i], -y[i]]]
        if eq_prefix is None: eq_prefix = e
        else:
            ne = f.new(); f.cl += [[-ne, eq_prefix], [-ne, e], [ne, -eq_prefix, -e]]; eq_prefix = ne
    f.cl.append(lt_terms)

def solve(target, D, s, nT, nJ, nC, solver="cadical195", timeout=600):
    out = build(target, D, s, nT, nJ, nC)
    rec = {"target": target, "D": D, "s": s, "nT": nT, "nJ": nJ, "nC": nC, "Tcost": nT + 4 * nJ + 8 * nC}
    if out is None: rec["status"] = "UNSAT(trivial)"; return rec
    f, G, A, k, N, btype, ncomb = out
    rec.update({"vars": f.n, "clauses": len(f.cl), "fault_combos": ncomb})
    t0 = time.time()
    with Solver(name=solver, bootstrap_with=f.cl, use_timer=True) as S:
        timer = threading.Timer(timeout, S.interrupt); timer.start()
        res = S.solve_limited(expect_interrupt=True)
        timer.cancel()
        rec["sec"] = round(time.time() - t0, 1)
        if res is None: rec["status"] = "TIMEOUT"
        elif not res: rec["status"] = "UNSAT"
        else:
            rec["status"] = "SAT"; model = set(x for x in S.get_model() if x > 0)
            rec["Tcols"] = ["".join("1" if G[r][j] in model else "0" for r in range(N)) for j in range(nT)]
            rec["blocks"] = [[btype[b]] + ["".join("1" if A[b][q][r] in model else "0" for r in range(N)) for q in range(3)]
                             for b in range(len(btype))]
    return rec

if __name__ == "__main__":
    a = sys.argv
    rec = solve(a[1], *map(int, a[2:7]), solver=a[7] if len(a) > 7 else "cadical195",
                timeout=float(a[8]) if len(a) > 8 else 600)
    print(json.dumps(rec), flush=True)
