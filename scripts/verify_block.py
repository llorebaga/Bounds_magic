"""Exact certification of a two-level design (T columns + CCZ blocks) found by sat_block.py / block_mip.py.

Phase g(v) = sum_T [g.v] + 4 sum_blocks [a.v][b.v][c.v]  (mod 8), v = (y, c), y = logical bits 0..k-1.
1. The check-involving part P(y,c) = g(y,c) - g(y,0) must be a diagonal Clifford (Moebius coefficients:
   singles even, pairs = 0 mod 4, cubic and higher = 0 mod 8)  -> the outer branch is a stabilizer branch.
2. For every combination of faulty units with total cost < D and every block pattern l != 0, the output
   out_u(y) = sum_c w^{g-P}(v) (-1)^{u.v} must be a multiple of out_0 (collapse), computed exactly in Z[w].
3. out_0 must have the Pauli spectrum of the target.
Also counts harmful unit combinations of cost exactly D (worst case over block patterns).
Usage: python verify_block.py design.json
"""
import sys, json, itertools
import numpy as np
from verify_tri import W, collinear, to_c, spectrum
from verify_tri2 import moebius

def certify(rec):
    Tcols = rec.get("Tcols", []); blocks = rec.get("blocks", []); D = rec["D"]
    k = {"ccz": 3, "cs12cs13": 3, "ccz2": 6, "cs": 2}[rec["target"]]
    N = len(Tcols[0]) if Tcols else len(blocks[0][1])
    val = lambda s: sum(int(ch) << r for r, ch in enumerate(s))
    tv = [val(s) for s in Tcols]
    bv = [(t, val(a), val(b), val(c)) for t, a, b, c in blocks]
    par = lambda x, v: bin(x & v).count("1") & 1
    g = np.zeros(1 << N, dtype=np.int64)
    for v in range(1 << N):
        g[v] = (sum(par(x, v) for x in tv) + 4 * sum(par(a, v) * par(b, v) * par(c, v) for _, a, b, c in bv)) % 8
    ymask = (1 << k) - 1
    P = np.array([(g[v] - g[v & ymask]) % 8 for v in range(1 << N)])
    co = moebius(P, N)
    cliff = all(not ((bin(S).count("1") == 1 and co[S] % 2) or (bin(S).count("1") == 2 and co[S] % 4)
                     or (bin(S).count("1") >= 3 and co[S] % 8)) for S in range(1, 1 << N))
    h = (g - P) % 8
    s = N - k
    cache = {}
    def out(u):
        if u in cache: return cache[u]
        o = []
        for y in range(1 << k):
            acc = [0, 0, 0, 0]
            for c in range(1 << s):
                v = y | (c << k); p = int(h[v]) + 4 * par(u, v)
                acc = [a + b for a, b in zip(acc, W[p % 8])]
            o.append(acc)
        cache[u] = o; return o
    o0 = out(0)
    if all(all(x == 0 for x in z) for z in o0): return {"clifford": cliff, "ok": False, "reason": "zero"}
    units = [(1, [x]) for x in tv]
    for t, a, b, c in bv:
        units.append((1 if t == 'J' else 2, [(a if l & 1 else 0) ^ (b if l & 2 else 0) ^ (c if l & 4 else 0) for l in range(1, 8)]))
    bad_below = 0; harmful_at_D = 0; memo = {}
    for size in range(1, D + 1):
        for combo in itertools.combinations(range(len(units)), size):
            cost = sum(units[i][0] for i in combo)
            if cost > D: continue
            for choice in itertools.product(*[units[i][1] for i in combo]):
                u = 0
                for x in choice: u ^= x
                if u not in memo: memo[u] = collinear(out(u), o0)
                if not memo[u]:
                    if cost < D: bad_below += 1
                    else: harmful_at_D += 1
    spec = sorted(set(spectrum([to_c(z) for z in o0]).tolist()))
    return {"clifford_correction": cliff, "ok": cliff and bad_below == 0, "violations_below_D": bad_below,
            "harmful_combos_at_cost_D": harmful_at_D, "spectrum": spec, "Tcost": rec.get("Tcost")}

if __name__ == "__main__":
    rec = json.load(open(sys.argv[1])) if not sys.argv[1].startswith("{") else json.loads(sys.argv[1])
    print(json.dumps(certify(rec)))
