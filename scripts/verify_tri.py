"""Exact verification of a phase-polynomial (generalized triorthogonal) protocol in the full stabilizer model.

Circuit: N registers (k logical y, s checks c) in |+>; for each column G_j apply T to the parity G_j.v
(one injected T state; an input Z fault on it multiplies by (-1)^{G_j.v}); project the checks onto a product
stabilizer state chi; output on the logical registers:
    out_e(y) = sum_c conj(chi(c)) w^{sum_j [G_j.(y,c)]} (-1)^{(G e).(y,c)}.
This is one branch of a stabilizer protocol consuming n T states.  Suppression order >= d iff out_0 != 0
and out_e is a multiple of out_0 for every |e| < d (rank-one / collapse condition of Theorem 2); the
Galois complements then follow from Theorem 1.  Arithmetic exact in Z[w] (w = e^{i pi/4}) as integer
4-vectors.  Also reports the leading error coefficient (number of harmful patterns at weight d) and
whether out_0 is Clifford-equivalent to the target (Pauli-spectrum test + explicit BFS for <=3 qubits).
"""
import sys, json, itertools
import numpy as np

def zw_mul(a, b):  # elements of Z[w]: sum a_t w^t, t=0..3, w^4=-1
    r = [0, 0, 0, 0]
    for i in range(4):
        for j in range(4):
            t = i + j; s = 1
            if t >= 4: t -= 4; s = -1
            r[t] += s * a[i] * b[j]
    return r
W = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]

def outputs(M, k, chi, faults):
    """M: N x n int array. chi: list of s single-qubit states in {'+','-','+i','-i'}. returns dict e -> list over y of Z[w]"""
    N, n = M.shape; s = N - k
    colv = [int(sum(int(M[a, j]) << a for a in range(N))) for j in range(n)]
    # conj(chi(c)) as power of w (unnormalised): '+' -> 1, '-' -> (-1)^c, '+i' -> (-i)^c, '-i' -> i^c
    chipow = {'+': 0, '-': 4, '+i': 6, '-i': 2}
    res = {}; cache = {}
    for e in faults:
        ge = 0
        for j in e: ge ^= colv[j]
        if ge in cache: res[e] = cache[ge]; continue
        out = []
        for y in range(1 << k):
            acc = [0, 0, 0, 0]
            for c in range(1 << s):
                v = y | (c << k)
                p = sum(bin(cv & v).count("1") & 1 for cv in colv)
                p += 4 * (bin(ge & v).count("1") & 1)
                p += sum(chipow[chi[t]] * ((c >> t) & 1) for t in range(s))
                acc = [x + y_ for x, y_ in zip(acc, W[p % 8])]
            out.append(acc)
        res[e] = out; cache[ge] = out
    return res

def to_c(z):
    w = np.exp(1j * np.pi / 4); return sum(z[t] * w ** t for t in range(4))

def collinear(u, v):
    """exact test u = lambda v over Q(w): all 2x2 minors u_i v_j - u_j v_i = 0"""
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            a = zw_mul(u[i], v[j]); b = zw_mul(u[j], v[i])
            if any(x != y for x, y in zip(a, b)): return False
    return True

def check(M, k, d, chi):
    N, n = M.shape
    faults = [()]
    for w in range(1, d + 1): faults += list(itertools.combinations(range(n), w))
    out = outputs(M, k, chi, faults)
    o0 = out[()]
    if all(all(x == 0 for x in z) for z in o0): return {"ok": False, "reason": "zero acceptance"}
    bad = {w: 0 for w in range(1, d + 1)}
    memo = {}
    for e in faults[1:]:
        key = id(out[e])
        if key not in memo: memo[key] = collinear(out[e], o0)
        if not memo[key]: bad[len(e)] += 1
    ok = all(bad[w] == 0 for w in range(1, d))
    return {"ok": ok, "harmful_by_weight": bad, "out0": [to_c(z) for z in o0]}

def spectrum(v):
    v = np.array(v, complex); v = v / np.linalg.norm(v); r = int(np.log2(len(v)))
    idx = np.arange(len(v)); pc = np.vectorize(lambda x: bin(x).count("1"))
    Hm = (-1.0) ** pc(idx[:, None] & idx[None, :]); XB = idx[:, None] ^ idx[None, :]
    return np.sort(np.round(np.abs((np.conj(v[XB]) * v[None, :]) @ Hm).ravel() / len(v) * len(v), 6))

def best_chi(M, k, d):
    s = M.shape[0] - k
    for chi in itertools.product(['+', '-', '+i', '-i'], repeat=s):
        r = check(M, k, d, list(chi))
        if r["ok"]: return list(chi), r
    return None, r

if __name__ == "__main__":
    rec = json.loads(sys.argv[1]) if sys.argv[1].startswith("{") else json.load(open(sys.argv[1]))
    M = np.array([[int(ch) for ch in row] for row in rec["matrix"]]); k = int(sys.argv[2]); d = rec["d"]
    chi, r = best_chi(M, k, d)
    print(json.dumps({"chi": chi, "ok": r["ok"], "harmful_by_weight": r.get("harmful_by_weight"),
                      "spectrum": sorted(set(spectrum(r["out0"]).tolist())) if r["ok"] else None}))
