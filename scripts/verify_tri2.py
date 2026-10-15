"""Exact certification of a generalized triorthogonal protocol in the full stabilizer model.

Registers v=(y,c): k logical, s checks, all |+>.  Column j: T on parity G_j.v (one input; a Z fault multiplies
by (-1)^{G_j.v}).  Phase function g(v) = sum_j [G_j.v] mod 8.  The check-involving part P(y,c) = g(y,c)-g(y,0)
is undone by the diagonal unitary w^{-P}; we verify EXACTLY (Moebius coefficients mod 8) that w^{-P} is a
Clifford (singles even, pairs = 0 mod 4, cubic = 0 mod 8).  Then project checks onto |+>^s:
    out_e(y) = sum_c w^{g(y,c)-P(y,c)} (-1)^{(Ge).(y,c)}.
This is one branch of a stabilizer protocol with n T inputs.  We test collapse (out_e || out_0) for all
|e| < d, count harmful patterns at weight d, and report the Pauli spectrum of out_0.
Usage: python verify_tri2.py file.json k [target]
"""
import sys, json, itertools
import numpy as np
from verify_tri import zw_mul, W, to_c, collinear, spectrum

def phase_fn(cols, N):
    g = np.zeros(1 << N, dtype=np.int64)
    for v in range(1 << N):
        g[v] = sum(bin(cv & v).count("1") & 1 for cv in cols) % 8
    return g

def moebius(f, N, maxdeg=None):
    """coefficients a_S (mod 8) with f(v) = sum_{S subset v} a_S"""
    a = f.copy()
    for i in range(N):
        for v in range(1 << N):
            if (v >> i) & 1: a[v] -= a[v ^ (1 << i)]
    return a % 8

def certify(M, k, d):
    N, n = M.shape; s = N - k
    cols = [int(sum(int(M[a, j]) << a for a in range(N))) for j in range(n)]
    g = phase_fn(cols, N)
    ymask = (1 << k) - 1
    P = np.array([(g[v] - g[v & ymask]) % 8 for v in range(1 << N)])
    a = moebius(P, N)
    clifford = True
    for S in range(1, 1 << N):
        w = bin(S).count("1"); c = int(a[S])
        if w == 1 and c % 2: clifford = False
        if w == 2 and c % 4: clifford = False
        if w >= 3 and c % 8: clifford = False
    h = (g - P) % 8
    faults = [()]
    for w in range(1, d + 1): faults += list(itertools.combinations(range(n), w))
    cache = {}
    def out(ge):
        if ge in cache: return cache[ge]
        o = []
        for y in range(1 << k):
            acc = [0, 0, 0, 0]
            for c in range(1 << s):
                v = y | (c << k)
                p = int(h[v]) + 4 * (bin(ge & v).count("1") & 1)
                acc = [x + t for x, t in zip(acc, W[p % 8])]
            o.append(acc)
        cache[ge] = o; return o
    o0 = out(0)
    if all(all(x == 0 for x in z) for z in o0): return {"clifford_correction": clifford, "ok": False}
    memo = {}; bad = {w: 0 for w in range(1, d + 1)}
    for e in faults[1:]:
        ge = 0
        for j in e: ge ^= cols[j]
        if ge not in memo: memo[ge] = collinear(out(ge), o0)
        if not memo[ge]: bad[len(e)] += 1
    return {"clifford_correction": clifford, "ok": clifford and all(bad[w] == 0 for w in range(1, d)),
            "harmful_by_weight": bad, "spectrum": sorted(set(spectrum([to_c(z) for z in o0]).tolist()))}

if __name__ == "__main__":
    rec = json.load(open(sys.argv[1])) if not sys.argv[1].startswith("{") else json.loads(sys.argv[1])
    M = np.array([[int(ch) for ch in row] for row in rec["matrix"]]); k = int(sys.argv[2])
    print(json.dumps({"n": M.shape[1], "N": M.shape[0], "d": rec["d"], **certify(M, k, rec["d"])}))
