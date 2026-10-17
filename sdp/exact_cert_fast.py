"""Fast exact certificates (same mathematics as exact_cert.py, all arithmetic in QS2).

Exact inputs (from exact_cert.exact_data, computed once with sympy): weight polynomials of all
1- and 2-qubit stabilizer states, the 13 two-qubit generators, and the facet family (1, s, -s, -s^2).
"""
import sys, time, pickle
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from qs2 import QS2, simplex_feasible, ZERO, ONE
import exact_cert as EC


def key(p):
    return tuple((v.a, v.b) for v in p)


def pmul(p, q):
    out = [ZERO] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a.is_zero():
            continue
        for j, b in enumerate(q):
            out[i + j] = out[i + j] + a * b
    return out


def load():
    try:
        return pickle.load(open("exact_small.pkl", "rb"))
    except FileNotFoundError:
        W1, W2, G = EC.exact_data()
        F = EC.facets_exact()
        conv = lambda L: [[QS2.from_sympy(x) for x in t] for t in L]
        data = (conv(W1), conv(W2), conv(G), conv(F))
        pickle.dump(data, open("exact_small.pkl", "wb"))
        return data


def conditioning(R, W1, W2):
    res = {}
    def rec(left, i1, i2, P, phase):
        if left == 0:
            res.setdefault(key(P), P); return
        if phase == 1:
            for i in range(i1, len(W1)):
                rec(left - 1, i, i2, pmul(P, W1[i]), 1)
        if left >= 2:
            for j in range(i2, len(W2)):
                rec(left - 2, len(W1), j, pmul(P, W2[j]), 2)
    rec(R, 0, 0, [ONE], 1)
    return list(res.values())


def certify(d, verbose=True):
    t0 = time.time()
    n = 2 * d - 1
    W1, W2, G, F = load()
    F = list({key(f): f for f in F}.values())
    for f in F:
        for g in G:
            v = ZERO
            for a, b in zip(f, g):
                v = v + a * b
            assert v.sign() >= 0
    Ms = conditioning(n - 1, W1, W2)
    rows = []
    for Mi, M in enumerate(Ms):
        for fi, f in enumerate(F):
            a = [ZERO] * (2 * (n + 1))
            for j in range(2):
                for b in range(2):
                    c = f[j * 2 + b]
                    for v, mv in enumerate(M):
                        if not mv.is_zero():
                            a[(v + j) * 2 + b] = a[(v + j) * 2 + b] + c * mv
            rows.append((Mi, fi, a))
    num = np.array([[float(x) for x in r] for _, _, r in rows])
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.NumericFocus = 3
    x = m.addMVar(2 * (n + 1), lb=0.0)
    for v in range(d):
        m.addConstr(x[v * 2 + 1] == 0)
    m.addConstr(x.sum() == 1)
    cons = m.addConstr(num @ x >= 0)
    m.setObjective(x[0], GRB.MAXIMIZE)
    m.optimize()
    y_num = -np.array(cons.Pi)
    order = np.argsort(-y_num)
    free = {v * 2 + 1 for v in range(d)}
    coords = [c for c in range(2 * (n + 1)) if c not in free and c != 0]
    sol = None
    for size in (int((y_num > 1e-9).sum()), 4 * int((y_num > 1e-9).sum()), 200, len(rows)):
        cols = list(order[:max(size, 1)])
        A = [[rows[i][2][c] for i in cols] + [ONE if cj == ci else ZERO for cj in range(len(coords))]
             for ci, c in enumerate(coords)]
        A.append([rows[i][2][0] for i in cols] + [ZERO] * len(coords))
        rhs = [ZERO] * len(coords) + [QS2(-1)]
        sol = simplex_feasible(A, rhs)
        if sol is not None:
            break
    if sol is None:
        print(f"d={d}: no exact certificate"); return False
    used = [(cols[k], sol[k]) for k in range(len(cols)) if not sol[k].is_zero()]
    total = [ZERO] * (2 * (n + 1))
    for i, v in used:
        assert v.sign() > 0
        for c in range(2 * (n + 1)):
            total[c] = total[c] + v * rows[i][2][c]
    ok = all(total[c].sign() <= 0 for c in coords) and (total[0] - QS2(-1)).is_zero()
    print(f"d={d} n={n}: {len(Ms)} conditioning polynomials, {len(rows)} valid inequalities; "
          f"exact certificate with {len(used)} inequalities verified: {ok}  [{time.time()-t0:.0f}s]", flush=True)
    pickle.dump({"d": d, "n": n, "used": [(key(Ms[rows[i][0]]), key(F[rows[i][1]]), (v.a, v.b)) for i, v in used]},
                open(f"exact_cert_T_d{d}.pkl", "wb"))
    return ok


if __name__ == "__main__":
    for d in [int(a) for a in sys.argv[1].split(",")]:
        certify(d)
