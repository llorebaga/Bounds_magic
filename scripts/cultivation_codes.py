"""Codes on the T gates of cultivation circuits (paper/cultivation.tex, Lemma 3, Theorem 5, Corollary 6, Sec. S2).

1. Lemma 3: for the nine-input T code, the 15-qubit Reed-Muller code and the eight-qubit CCZ code, every Pauli on the
   inputs whose support is smaller than the order of the code collapses.
2. Proposition 12 (combinatorial form of the collapse condition): the twisted signed weight distributions of the
   cosets of S_C in its normalizer decide collapse, compared with state vectors for every Pauli on random codes.
3. Theorem 5, construction: T gates on |+> data, a cat-state measurement of every generator, each cat state checked
   in rounds of parity measurements, and an ideal decoder; all single faults, and for the 15-qubit code all pairs
   of faults, are harmless.
4. Theorem 5, bounds: for the published distance-three circuit, the order of the code on its T gates and the
   weights modulo S_C of the Paulis of undetected pairs of faults.
5. Corollary 6: exact Farkas certificates for the linear-programming bound on [[n,k,d]] stabilizer codes, with the
   quantum MacWilliams identities and Rains' shadow enumerator (--lp; several minutes).

Usage: python cultivation_codes.py [--lp]      Writes results/cultivation/codes.json.
"""
import itertools
import json
import os
import sys
import time
from fractions import Fraction
from math import gcd

import numpy as np

import cultiv as cv

HERE = os.path.dirname(os.path.abspath(__file__))
W8 = np.exp(1j * np.pi / 4)
MATS = {"X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}

NINE = ["-XZIIZIIIX", "+XIZIZIIXI", "+ZXZZIIXZI", "+ZZXIZXIIZ", "-YXYIXIIZZ", "-YYXXIIIII", "+IZIZZIZII", "+IIZZZZIII"]
CUBE = ["+XXXXXXXX", "+ZZZZIIII", "+ZZIIZZII", "+ZIZIZIZI", "+ZZZZZZZZ"]
_PTS = list(range(1, 16))
RM15 = (["+" + "".join("X" if (x >> i) & 1 else "I" for x in _PTS) for i in range(4)]
        + ["+" + "".join("Z" if (x >> i) & 1 else "I" for x in _PTS) for i in range(4)]
        + ["+" + "".join("Z" if (x >> i) & 1 and (x >> j) & 1 else "I" for x in _PTS)
           for i in range(4) for j in range(i + 1, 4)])
CODES = {"[[9,1,3]] nine-input T code": NINE, "[[8,3,2]] CCZ code": CUBE, "[[15,1,3]] Reed-Muller code": RM15}


# ------------------------------------------------------------------ state vectors on the T inputs
def apply_letters(psi, s):
    t = len(s)
    psi = psi.reshape([2] * t)
    for q, c in enumerate(s):
        if c != "I":
            psi = np.moveaxis(np.tensordot(MATS[c], psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def project(psi, gens):
    for g in gens:
        psi = (psi + (-1 if g[0] == "-" else 1) * apply_letters(psi, g[1:])) / 2
    return psi


def t_states(t):
    psi = np.ones(1, dtype=complex)
    for _ in range(t):
        psi = np.kron(psi, np.array([1, W8]) / np.sqrt(2))
    return psi


def order(gens):
    """least weight of a Z pattern that does not collapse, and the number of such patterns of that weight"""
    t = len(gens[0]) - 1
    T, ref = t_states(t), project(t_states(t), gens)
    for w in range(1, t + 1):
        bad = sum(1 for e in itertools.combinations(range(t), w)
                  if not cv.parallel(project(apply_letters(T, "".join("Z" if q in e else "I" for q in range(t))), gens), ref))
        if bad:
            return w, bad
    return None, 0


def lemma3(gens):
    """number of Paulis of each support size below the order that do not collapse (Lemma 3 says none)"""
    t = len(gens[0]) - 1
    dC, _ = order(gens)
    T, ref = t_states(t), project(t_states(t), gens)
    out = {}
    for w in range(1, dC):
        bad = tot = 0
        for supp in itertools.combinations(range(t), w):
            for let in itertools.product("XYZ", repeat=w):
                s = ["I"] * t
                for q, c in zip(supp, let):
                    s[q] = c
                tot += 1
                bad += not cv.parallel(project(apply_letters(T, "".join(s)), gens), ref)
        out[w] = {"Paulis": tot, "not collapsing": bad}
    return dC, out


# ------------------------------------------------------------------ Proposition 12
def herm_pauli(x, z, t):
    m = np.array([[1]])
    for k in range(t):
        m = np.kron(m, [np.eye(2), MATS["X"], MATS["Z"], MATS["Y"]][((x >> k) & 1) + 2 * ((z >> k) & 1)])
    return m


def enumerator_test(seed=3, trials=25):
    """collapse of every Pauli E against A_E(Q) A_0(S) = A_E(S) A_0(Q) on random codes"""
    rng = np.random.default_rng(seed)
    H, S = np.array([[1, 1], [1, -1]]) / np.sqrt(2), np.diag([1, 1j])
    tested = bad = codes = 0
    for _ in range(trials):
        t, r = int(rng.integers(2, 6)), int(rng.integers(1, 3))
        r = min(r, t)
        d = 2 ** t
        op1 = lambda U, q: np.kron(np.kron(np.eye(2 ** q), U), np.eye(2 ** (t - q - 1)))
        U = np.eye(d, dtype=complex)
        for _ in range(40):
            k, q = rng.integers(3), int(rng.integers(t))
            if k == 0:
                U = op1(H, q) @ U
            elif k == 1:
                U = op1(S, q) @ U
            else:
                q2 = int(rng.integers(t))
                if q2 != q:
                    P = np.zeros((d, d))
                    for i in range(d):
                        j = i ^ (1 << (t - 1 - q2)) if (i >> (t - 1 - q)) & 1 else i
                        P[j, i] = 1
                    U = P @ U
        P0 = np.eye(d, dtype=complex)
        for q in range(t - r):
            P0 = P0 @ (np.eye(d) + (1 if rng.integers(2) == 0 else -1) * op1(MATS["Z"], q)) / 2
        Pi = U @ P0 @ U.conj().T
        T = t_states(t)
        psi = Pi @ T
        if np.linalg.norm(psi) < 1e-9:
            continue
        codes += 1
        allp = [(x, z) for x in range(d) for z in range(d)]
        anti = lambda a, b: (bin(a[0] & b[1]).count("1") + bin(a[1] & b[0]).count("1")) & 1
        stab = {}
        for p in allp:
            c = np.trace(herm_pauli(*p, t) @ Pi).real / np.trace(Pi).real
            if abs(abs(c) - 1) < 1e-9:
                stab[p] = int(round(c))
        norm = [p for p in allp if all(anti(p, s) == 0 for s in stab)]
        seen, reps = set(), []
        for p in norm:
            if p not in seen:
                seen |= {(p[0] ^ s[0], p[1] ^ s[1]) for s in stab}
                reps.append(p)

        def A(E, L):
            Lm = herm_pauli(*L, t)
            return sum(np.vdot(T, Lm @ (sg * herm_pauli(*s, t)) @ T) * (-1) ** anti(E, (L[0] ^ s[0], L[1] ^ s[1]))
                       for s, sg in stab.items())
        A0 = {L: A((0, 0), L) for L in reps}
        for E in allp:
            direct = cv.parallel(Pi @ (herm_pauli(*E, t) @ T), psi)
            AES = A(E, (0, 0))
            crit = all(abs(A(E, L) * A0[(0, 0)] - AES * A0[L]) < 1e-9 for L in reps)
            tested += 1
            bad += crit != direct
    return {"random codes": codes, "Paulis tested": tested, "disagreements": bad}


# ------------------------------------------------------------------ Theorem 5, construction
def construction(gens, rounds):
    """ops, ideal op indices, number of qubits, number of T gates"""
    t = len(gens[0]) - 1
    wmax = max(sum(c != "I" for c in g[1:]) for g in gens)
    cat0, anc = t, t + wmax
    ops = [("R", q, "X") for q in range(t)] + [("T", q, 1) for q in range(t)]
    for g in gens:
        supp = [q for q in range(t) if g[1 + q] != "I"]
        cat = list(range(cat0, cat0 + len(supp)))
        ops.append(("R", cat[0], "X"))
        for c in cat[1:]:
            ops += [("R", c, "Z"), ("U2", "CX", cat[0], c)]
        for _ in range(rounds):
            for c in cat[1:]:
                ops += [("R", anc, "Z"), ("U2", "CX", cat[0], anc), ("U2", "CX", c, anc), ("M", "Z", (anc,), 1, True)]
        for c, q in zip(cat, supp):
            ops.append(("U2", "C" + g[1 + q], c, q))
        for c in cat:
            ops.append(("M", "X", (c,), 1, True))
    ideal = set()
    for g in gens:
        supp = [q for q in range(t) if g[1 + q] != "I"]
        ideal.add(len(ops))
        ops.append(("M", "".join(g[1 + q] for q in supp), tuple(supp), 1, False))
    return ops, ideal, anc + 1, t


def check_construction(gens, rounds, pairs):
    ops, ideal, nq, t = construction(gens, rounds)
    sk = cv.Skeleton(ops, nq, list(range(t)))
    faults = [(li, sk.key(*sk.fault_effect(i, f))) for li, (i, fs) in enumerate(cv.locations(ops, ideal)) for f in fs]
    T, ref = t_states(t), project(t_states(t), gens)
    cache = {}

    def harmless(U, E):
        if U:
            return False
        if E not in cache:
            s = "".join("IXZY"[((E >> j) & 1) + 2 * ((E >> (t + j)) & 1)] for j in range(t))
            cache[E] = cv.parallel(project(apply_letters(T, s), gens), ref)
        return cache[E]
    res = {"T gates": t, "qubits": nq, "rounds of checks": rounds, "noisy operations": len(ops) - len(ideal),
           "two-qubit gates": sum(op[0] == "U2" for op in ops), "elementary faults": len(faults),
           "harmful single faults": sum(1 for li, (D, U, E) in faults if D == 0 and not harmless(U, E))}
    if pairs:
        per = {}
        for li, k in faults:
            per[li] = per.get(li, 0) + 1
        tot = sum(per.values())
        res["pairs at distinct locations"] = (tot * tot - sum(c * c for c in per.values())) // 2
        by = {}
        for li, k in faults:
            by.setdefault(k[0], []).append((li, k))
        und = bad = 0
        for lst in by.values():
            for (la, ka), (lb, kb) in itertools.combinations(lst, 2):
                if la != lb:
                    und += 1
                    bad += not harmless(ka[1] ^ kb[1], ka[2] ^ kb[2])
        res.update({"undetected pairs": und, "harmful pairs": bad})
    return res


# ------------------------------------------------------------------ Theorem 5, published circuit
def published():
    import cultivation_gsj as G
    ops, weights, ideal, data, nq = G.build("T")
    A, faults = G.analysis(ops, G.all_faults(ops, ideal), ideal, data, nq)
    sk, t = A.sk, A.sk.t
    group = [0]
    for v in sk.SC:
        group = group + [s ^ v for s in group]
    group = np.array(group, dtype=np.int64)

    def wmod(E):
        x = group ^ E
        return int(min(bin(int(s)).count("1") for s in (x & ((1 << t) - 1)) | ((x >> t) & ((1 << t) - 1))))
    zbad = {}
    for w in (1, 2, 3):
        zbad[w] = sum(1 for e in itertools.combinations(range(t), w)
                      if not A.harmless_UE(0, cv._reduce(sum(1 << (t + j) for j in e), sk.SC_rref)))
    by = {}
    for loc, i, f, w, k in faults:
        by.setdefault(k[0], []).append((loc, k))
    table = {}
    for lst in by.values():
        for (la, ka), (lb, kb) in itertools.combinations(lst, 2):
            if la != lb:
                k = tuple(x ^ y for x, y in zip(ka, kb))
                key = (wmod(k[2]), "harmless" if A.harmless_key(k) else "harmful")
                table[key] = table.get(key, 0) + 1
    single = max(wmod(k[2]) for loc, i, f, w, k in faults if k[0] == 0)
    return {"T gates": t, "harmful Z patterns by weight": zbad,
            "largest weight modulo S_C of an undetected single fault": single,
            "undetected pairs by weight modulo S_C": {"%d %s" % kk: v for kk, v in sorted(table.items())}}


# ------------------------------------------------------------------ Corollary 6, linear-programming bound
def binom(a, b):
    if b < 0 or b > a:
        return 0
    r = 1
    for i in range(b):
        r = r * (a - i) // (i + 1)
    return r


def kraw(n, i, j):
    """coefficient of y^j in (1+3y)^(n-i) (1-y)^i"""
    return sum((-1) ** s * 3 ** (j - s) * binom(i, s) * binom(n - i, j - s) for s in range(min(i, j) + 1))


def lp_rows(n, k, d):
    """the system over A_0..A_n, scaled by 2^(n-k) to integers: (coefficients, rhs, 'eq' or 'ge')"""
    M = 2 ** (n - k)
    rows = [([1] + [0] * n, 1, "eq"), ([1] * (n + 1), M, "eq")]
    for j in range(n + 1):
        rows.append(([kraw(n, i, j) - (M if i == j else 0) for i in range(n + 1)], 0, "eq" if 1 <= j < d else "ge"))
        rows.append(([(-1) ** i * kraw(n, i, j) for i in range(n + 1)], 0, "ge"))
    return rows


def farkas(n, k, d):
    """exact phase-one simplex (Bland's rule); a Farkas certificate if the system is infeasible, else None"""
    rows = lp_rows(n, k, d)
    m, nA = len(rows), n + 1
    ge = [r for r in range(m) if rows[r][2] == "ge"]
    nS = len(ge)
    ncol = nA + nS + m
    T = []
    for r, (coef, rhs, kind) in enumerate(rows):
        row = [Fraction(c) for c in coef] + [Fraction(0)] * (nS + m) + [Fraction(rhs)]
        if kind == "ge":
            row[nA + ge.index(r)] = Fraction(-1)
        if rhs < 0:
            row = [-x for x in row]
        row[nA + nS + r] = Fraction(1)
        T.append(row)
    basis = [nA + nS + r for r in range(m)]
    cost = [Fraction(0)] * (nA + nS) + [Fraction(1)] * m
    while True:
        red = [cost[j] - sum(cost[basis[r]] * T[r][j] for r in range(m)) for j in range(ncol)]
        enter = next((j for j in range(ncol) if red[j] < 0), None)
        if enter is None:
            break
        leave = None
        for r in range(m):
            if T[r][enter] > 0:
                ratio = T[r][-1] / T[r][enter]
                if leave is None or ratio < best or ratio == best and basis[r] < basis[leave]:
                    best, leave = ratio, r
        piv = T[leave][enter]
        T[leave] = [x / piv for x in T[leave]]
        for r in range(m):
            if r != leave and T[r][enter] != 0:
                f = T[r][enter]
                T[r] = [a - f * b for a, b in zip(T[r], T[leave])]
        basis[leave] = enter
    if sum(cost[basis[r]] * T[r][-1] for r in range(m)) == 0:
        return None
    y = []
    for r in range(m):
        j = nA + nS + r
        yr = 1 - (cost[j] - sum(cost[basis[q]] * T[q][j] for q in range(m)))
        y.append(-yr if rows[r][1] < 0 else yr)
    return y


def certificate_valid(n, k, d, y):
    """y >= 0 on inequalities, y.rows <= 0 on every A_i >= 0, y.rhs > 0: then no nonnegative solution exists"""
    rows = lp_rows(n, k, d)
    den = 1
    for v in y:
        den = den * v.denominator // gcd(den, v.denominator)
    yi = [int(v * den) for v in y]
    return (all(yi[r] >= 0 for r in range(len(rows)) if rows[r][2] == "ge")
            and all(sum(yi[r] * rows[r][0][i] for r in range(len(rows))) <= 0 for i in range(n + 1))
            and sum(yi[r] * rows[r][1] for r in range(len(rows))) > 0)


def weight_distribution(gens):
    """weight distribution of the stabilizer group (for the validation of the system on actual codes)"""
    t = len(gens[0]) - 1
    vec = lambda s: (sum(1 << q for q in range(t) if s[q] in "XY") | (sum(1 << q for q in range(t) if s[q] in "YZ") << t))
    S = [0]
    for g in gens:
        S = S + [x ^ vec(g[1:]) for x in S]
    A = [0] * (t + 1)
    for x in S:
        A[bin((x & ((1 << t) - 1)) | (x >> t)).count("1")] += 1
    return A


LP_CLAIMS = {1: {6: 16, 7: 17, 8: 22, 9: 23, 10: 28, 11: 29, 12: 34}, 3: {4: 12, 5: 14, 6: 17, 7: 20, 8: 23}}


def lp_bounds():
    out = {"validation": {}}
    for name, gens in CODES.items():
        t, k = len(gens[0]) - 1, len(gens[0]) - 1 - len(gens)
        dist = 3 if "[[9" in name or "[[15" in name else 2
        A = weight_distribution(gens)
        ok = all((sum(c * a for c, a in zip(coef, A)) == rhs) if kind == "eq" else (sum(c * a for c, a in zip(coef, A)) >= rhs)
                 for coef, rhs, kind in lp_rows(t, k, dist))
        out["validation"][name] = ok
    for k, claims in LP_CLAIMS.items():
        for d, n in claims.items():
            y = farkas(n - 1, k, d)
            out["no [[%d,%d,%d]] code" % (n - 1, k, d)] = y is not None and certificate_valid(n - 1, k, d, y)
            print("k=%d d=%d: certificate for n=%d" % (k, d, n - 1), out["no [[%d,%d,%d]] code" % (n - 1, k, d)], flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    res = {"lemma 3": {}, "orders": {}}
    for name, gens in CODES.items():
        dC, out = lemma3(gens)
        res["orders"][name] = {"order": dC, "harmful Z patterns of that weight": order(gens)[1]}
        res["lemma 3"][name] = out
    print("Lemma 3", res["lemma 3"], flush=True)
    res["proposition 12"] = enumerator_test()
    print("Proposition 12", res["proposition 12"], flush=True)
    res["construction"] = {
        "[[8,3,2]] CCZ code, 1 round": check_construction(CUBE, 1, False),
        "[[9,1,3]] nine-input T code, 1 round": check_construction(NINE, 1, False),
        "[[15,1,3]] Reed-Muller code, 2 rounds": check_construction(RM15, 2, True),
        "[[15,1,3]] Reed-Muller code, 1 round": check_construction(RM15, 1, True)}
    print("construction", json.dumps(res["construction"], indent=1), flush=True)
    res["published circuit"] = published()
    print("published circuit", res["published circuit"], flush=True)
    if "--lp" in sys.argv:
        res["linear-programming bound"] = lp_bounds()
    res["seconds"] = round(time.time() - t0)
    path = os.path.join(HERE, "..", "results", "cultivation", "codes.json")
    if "--lp" not in sys.argv and os.path.exists(path):
        try:
            old = json.load(open(path))
        except ValueError:
            old = {}
        if "linear-programming bound" in old:
            res["linear-programming bound"] = old["linear-programming bound"]
    json.dump(res, open(path, "w"), indent=1, default=int)
    print("wrote", path)
