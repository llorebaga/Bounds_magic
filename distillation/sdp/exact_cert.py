"""Exact certificate that no stabilizer protocol distills one T state with order d from n = 2d-1 inputs.

Relaxation (README, level 1): lam[v,b] >= 0, lam[v,1] = 0 for v < d, lam[0,0] = 1, and for every
conditioning stabilizer state on n-1 inputs with T-basis weight polynomial M(x) = sum_v m_v x^v,
    g = ( <M,A>, <M,B>, <xM,A>, <xM,B> )        (A = lam[.,0], B = lam[.,1])
must satisfy f.g >= 0 for every valid inequality f of the 2-qubit stabilizer cone (one input, output).

Everything is exact in Q(sqrt2):
  * the 60 two-qubit and 6 one-qubit stabilizer states are enumerated with exact amplitudes;
  * each inequality f is checked against all 13 aggregated two-qubit generators;
  * conditioning polynomials are products of exact one- and two-qubit weight polynomials;
  * the Farkas multipliers y >= 0 are solved exactly on the support suggested by a floating LP, and
    the identity  sum_i y_i a_i <= 0 (componentwise, free coordinates excepted), coefficient of
    lam[0,0] = -1, is verified exactly. Then 0 <= sum_i y_i a_i . lam = -lam[0,0] + (<= 0) < 0.
"""
import itertools
import sys
import numpy as np
import sympy as sp
import gurobipy as gp
from gurobipy import GRB

r2 = sp.sqrt(2)
w = (1 + sp.I) / r2
s = 3 - 2 * r2


def stab_states_exact(N):
    """All N-qubit stabilizer states (exact amplitude vectors), Dehaene-De Moor form."""
    states = []
    for t in range(N + 1):
        for piv in itertools.combinations(range(N), t):
            free = [[j for j in range(p + 1, N) if j not in piv] for p in piv]
            for bits in itertools.product((0, 1), repeat=sum(len(f) for f in free)):
                B = [[0] * N for _ in range(t)]
                it = iter(bits)
                for i, p in enumerate(piv):
                    B[i][p] = 1
                    for j in free[i]:
                        B[i][j] = next(it)
                nonpiv = [j for j in range(N) if j not in piv]
                for cb in itertools.product((0, 1), repeat=len(nonpiv)):
                    c = [0] * N
                    for j, b in zip(nonpiv, cb):
                        c[j] = b
                    for ph in itertools.product(range(4), repeat=t):
                        for q in itertools.product((0, 1), repeat=t * (t - 1) // 2):
                            amp = [0] * (2 ** N)
                            for u in itertools.product((0, 1), repeat=t):
                                x = [(sum(u[i] * B[i][j] for i in range(t)) + c[j]) % 2 for j in range(N)]
                                e = sum(ph[i] * u[i] for i in range(t))
                                k = 0
                                for a in range(t):
                                    for b2 in range(a + 1, t):
                                        e += 2 * q[k] * u[a] * u[b2]
                                        k += 1
                                idx = int("".join(map(str, x)), 2) if N else 0
                                amp[idx] += sp.I ** (e % 4)
                            states.append([sp.nsimplify(a) / sp.sqrt(2 ** t) for a in amp])
    return states


def t_basis_dist(state, N):
    """d_e = |<phi_e|s>|^2 with phi_0 = |T>, phi_1 = Z|T>; patterns in product order."""
    out = []
    for e in itertools.product((0, 1), repeat=N):
        a = 0
        for xi, x in enumerate(itertools.product((0, 1), repeat=N)):
            coef = 1
            for ej, xj in zip(e, x):
                coef *= sp.conjugate(w) ** xj * (-1) ** (ej * xj) / r2
            a += coef * state[xi]
        out.append(sp.simplify(sp.expand(a * sp.conjugate(a))))
    return out


def exact_data():
    one = [t_basis_dist(st, 1) for st in stab_states_exact(1)]
    two = [t_basis_dist(st, 2) for st in stab_states_exact(2)]
    assert len(two) == 60
    # weight polynomials (as coefficient tuples) of 1- and 2-qubit states
    W1 = {tuple(sp.nsimplify(sp.radsimp(v)) for v in d) for d in one}
    W2 = {tuple(sp.radsimp(v) for v in (d[0], d[1] + d[2], d[3])) for d in two}
    # level-1 generators: (j, b) = (bit of qubit 1, bit of qubit 2)
    G = {tuple(sp.radsimp(v) for v in d) for d in two}
    return sorted(W1, key=str), sorted(W2, key=str), sorted(G, key=str)


def facets_exact():
    base = [1, s, -s, -s ** 2]          # order (00, 01, 10, 11) = (j b)
    def flipj(f): return [f[2], f[3], f[0], f[1]]
    def flipb(f): return [f[1], f[0], f[3], f[2]]
    def swap(f): return [f[0], f[2], f[1], f[3]]
    out = set()
    frontier = [tuple(base)]
    while frontier:
        f = frontier.pop()
        if f in out:
            continue
        out.add(f)
        for g in (flipj, flipb, swap):
            frontier.append(tuple(sp.expand(x) for x in g(list(f))))
    return [list(f) for f in out]


def poly_mul(p, q):
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return [sp.expand(c) for c in out]


def conditioning_polys(R, W1, W2):
    """All products of 1- and 2-qubit weight polynomials using exactly R qubits."""
    res = {}
    def rec(left, k1, k2, P):
        if left == 0:
            res[tuple(sp.nsimplify(c) for c in P)] = P
            return
        for i in range(k1, len(W1)):
            rec(left - 1, i, 0 if False else k2, poly_mul(P, list(W1[i])))
        if left >= 2:
            for j in range(k2, len(W2)):
                rec(left - 2, len(W1), j, poly_mul(P, list(W2[j])))
    # canonical order: all 1-qubit factors first (nondecreasing index), then 2-qubit factors
    def rec2(left, i1, i2, P, phase):
        if left == 0:
            res[tuple(sp.radsimp(c) for c in P)] = P
            return
        if phase == 1:
            for i in range(i1, len(W1)):
                rec2(left - 1, i, i2, poly_mul(P, list(W1[i])), 1)
        if left >= 2:
            for j in range(i2, len(W2)):
                rec2(left - 2, len(W1), j, poly_mul(P, list(W2[j])), 2)
    rec2(R, 0, 0, [sp.Integer(1)], 1)
    return list(res.values())


def constraint_rows(n, M, f):
    """Row a (length 2(n+1), index v*2+b) with a.lam = f.g for conditioning polynomial M."""
    a = [sp.Integer(0)] * (2 * (n + 1))
    for j in range(2):
        for b in range(2):
            coef = f[j * 2 + b]
            for v, mv in enumerate(M):
                if mv != 0:
                    a[(v + j) * 2 + b] += coef * mv
    return [sp.expand(x) for x in a]


def certify(d, verbose=True):
    n = 2 * d - 1
    W1, W2, G = exact_data()
    F = facets_exact()
    # validity of every inequality against every exact generator
    for f in F:
        for g in G:
            val = sp.radsimp(sum(fi * gi for fi, gi in zip(f, g)))
            assert val >= 0, (f, g, val)
    Ms = conditioning_polys(n - 1, W1, W2)
    rows = [(Mi, fi, constraint_rows(n, M, f)) for Mi, M in enumerate(Ms) for fi, f in enumerate(F)]
    if verbose:
        print(f"d={d} n={n}: {len(Ms)} conditioning polynomials, {len(rows)} inequalities (all valid)")
    # floating LP: max lam00 s.t. sum lam = 1 -> dual support
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
    supp = [i for i in range(len(rows)) if y_num[i] > 1e-9]
    if verbose:
        print(f"  floating max lam00 = {m.ObjVal:.2e}; dual support {len(supp)}")
    # exact Farkas LP over Q(sqrt2): y >= 0, slack >= 0,
    #   sum_i y_i a_i[c] + slack_c = 0  for every coordinate c except lam[v,1] (v < d) and lam[0,0],
    #   sum_i y_i a_i[0] = -1.
    from qs2 import QS2, simplex_feasible, ZERO, ONE
    free = {v * 2 + 1 for v in range(d)}
    coords = [c for c in range(2 * (n + 1)) if c not in free and c != 0]
    for cols in (supp, list(range(len(rows)))):
        ex = [[QS2.from_sympy(rows[i][2][c]) for c in range(2 * (n + 1))] for i in cols]
        A, rhs = [], []
        for ci, c in enumerate(coords):
            A.append([ex[k][c] for k in range(len(cols))] + [ONE if cj == ci else ZERO for cj in range(len(coords))])
            rhs.append(ZERO)
        A.append([ex[k][0] for k in range(len(cols))] + [ZERO] * len(coords))
        rhs.append(QS2(-1))
        sol = simplex_feasible(A, rhs)
        if sol is not None:
            break
    if sol is None:
        print("  no exact certificate found")
        return False
    y = sol[:len(cols)]
    # independent verification of the certificate
    used = [(cols[k], y[k]) for k in range(len(cols)) if not y[k].is_zero()]
    assert all(v.sign() > 0 for _, v in used)
    total = [ZERO] * (2 * (n + 1))
    for i, v in used:
        for c in range(2 * (n + 1)):
            total[c] = total[c] + v * QS2.from_sympy(rows[i][2][c])
    ok = all(total[c].sign() <= 0 for c in coords) and (total[0] - QS2(-1)).is_zero()
    if verbose:
        print(f"  exact certificate with {len(used)} inequalities, verified: {ok}")
        for i, v in used:
            Mi, fi, _ = rows[i]
            print(f"    y = {float(v):.6g}   M = {[str(sp.nsimplify(c)) for c in Ms[Mi]]}   facet {fi}")
    return ok


if __name__ == "__main__":
    for d in [int(a) for a in sys.argv[1].split(",")]:
        print(f"d={d}: {'CERTIFIED: n_min(T,d) >= 2d' if certify(d) else 'not certified'}", flush=True)
