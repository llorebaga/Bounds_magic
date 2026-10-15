"""Exhaustive exact search for single-branch protocols  (n inputs, r=3 outputs).

A protocol exists iff some signed [[n,3]] stabilizer code C has
   Pi_C phi_0 != 0  and  Pi_C phi_e collinear with Pi_C phi_0 for all e in E,
(plus decoded state Clifford-equivalent to the target; checked afterwards on survivors),
where E = harmless patterns (|e|<d or |e|>n-d) minus 0.  Collinearity <=> 2x2 minors
G00*Gee - |G0e|^2 = 0 (Cauchy-Schwarz equality), all computed EXACTLY in Z[sqrt2][i].

Parametrization of the unsigned stabilizer group S (rank m=n-3):
   A_X = X-support space (dim a), taken up to S_n;  B_Z = Z-type part (dim m-a) inside A_X^perp;
   lifts X^{a_i} Z^{c_i}, c_i reduced mod B_Z, pairwise commuting, W-gauge c_{i(t),t}=0.
All 2^m sign choices are handled by a Walsh-Hadamard transform over the group.
Usage: python enum_codes.py n d [a_min a_max]
"""
import sys, json, time, os
import numpy as np
from common import *

n = int(sys.argv[1]); d = int(sys.argv[2]); R = int(os.environ.get('R', 3)); m = n - R
a_lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
a_hi = int(sys.argv[4]) if len(sys.argv) > 4 else min(m, n)
shard = int(sys.argv[5]) if len(sys.argv) > 5 else 0
nshards = int(sys.argv[6]) if len(sys.argv) > 6 else 1
E = [e for e in harmless_set(n, d) if e != 0]
C = (n + 1) // 2
MASKN = (1 << n) - 1

def wht(A):  # along axis 0, length 2^m
    A = A.copy(); h = 1; L = A.shape[0]
    while h < L:
        A = A.reshape((L // (2 * h), 2, h) + A.shape[1:])
        A = np.concatenate([A[:, 0] + A[:, 1], A[:, 0] - A[:, 1]], axis=1).reshape((L,) + A.shape[3:])
        h *= 2
    return A

def values(elems, e, f):
    """exact <phi_e|P|phi_f> * 2^C for each group element; returns int array (2^m, N, 4) = (re, re*rt2, im, im*rt2)"""
    out = []
    for (p, X, Z) in elems:
        k = Z ^ e ^ f
        valid = (k & (~X & MASKN)) == 0
        q = (p + 2 * popcount(e & X) - popc_arr(k)) % 4
        w = popcount(X)
        mag = 1 << (C - (w + 1) // 2)
        slot = np.where(q % 2 == 0, 0, 2) + (w % 2)
        sgn = np.where((q == 0) | (q == 1), 1, -1) * valid * mag
        V = np.zeros((len(Z), 4), dtype=np.int64)
        np.put_along_axis(V, slot[:, None], sgn[:, None], axis=1)
        out.append(V)
    return np.stack(out)

def zmul_real(A, B):  # elements of Z[sqrt2]: (a0,a1)
    return np.stack([A[..., 0] * B[..., 0] + 2 * A[..., 1] * B[..., 1], A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]], -1)

def abs2(Z):  # |re + i im|^2 with re,im in Z[sqrt2]
    re, im = Z[..., 0:2], Z[..., 2:4]
    return zmul_real(re, re) + zmul_real(im, im)

def group_elements(lifts_a, lifts_c, bz):
    """lifts_a: list of ints, lifts_c: list of arrays (N,), bz: list of ints -> list over x of (p,X,Z) with arrays"""
    N = len(lifts_c[0]) if lifts_c else 1
    gens = [(popc_arr(a & c) % 4, a, c) for a, c in zip(lifts_a, lifts_c)] + \
           [(np.zeros(N, np.int64), 0, np.full(N, b, np.int64)) for b in bz]
    elems = [(np.zeros(N, np.int64), 0, np.zeros(N, np.int64))]
    for (q, X2, Z2) in gens:
        new = []
        for (p, X, Z) in elems:
            new.append(((p + q + 2 * popc_arr(Z & X2)) % 4, X ^ X2, Z ^ Z2))
        elems = elems + new
    return elems

def check(elems):
    """returns alive mask (2^m, N) of sign choices passing all minors and G00 != 0"""
    G00 = wht(values(elems, 0, 0))
    alive = (G00[..., 0] != 0) | (G00[..., 1] != 0)
    for e in E:
        if not alive.any(): break
        G0e = wht(values(elems, 0, e)); Gee = wht(values(elems, e, e))
        lhs = zmul_real(G00[..., 0:2], Gee[..., 0:2]); rhs = abs2(G0e)
        alive &= (lhs == rhs).all(-1)
    return alive, G00

def lift_space(AX, BZ):
    a = len(AX)
    cons = []
    piv = [r.bit_length() - 1 for r in BZ]
    for p in piv:
        for i in range(a): cons.append(1 << (i * n + p))
    for i in range(a):
        for j in range(i + 1, a):
            msk = 0
            for t in range(n):
                if (AX[i] >> t) & 1: msk ^= 1 << (j * n + t)
                if (AX[j] >> t) & 1: msk ^= 1 << (i * n + t)
            cons.append(msk)
    supp = 0
    for r in AX: supp |= r
    for t in range(n):
        if (supp >> t) & 1 and t not in piv:
            i0 = min(i for i in range(a) if (AX[i] >> t) & 1)
            cons.append(1 << (i0 * n + t))
    return nullspace(cons, a * n)

def all_combos(basis):
    sols = np.zeros(1, dtype=np.int64)
    for b in basis:
        sols = np.concatenate([sols, sols ^ b])
    return sols

def subspaces_within(vecs, k):
    """all k-dim subspaces of span(vecs) (vecs independent), as rref tuples in F2^n"""
    res = set()
    for sub in subspaces(len(vecs), k):
        rows = []
        for r in sub:
            v = 0
            for i in range(len(vecs)):
                if (r >> i) & 1: v ^= vecs[i]
            rows.append(v)
        res.add(rref(rows))
    return sorted(res)

TARGET = os.environ.get("TARGET")
def target_state(name):
    if name == "ccz": f = lambda x: (x & (x >> 1) & (x >> 2)) & 1; r = 3
    elif name == "ccz2": f = lambda x: ((x & (x >> 1) & (x >> 2)) ^ ((x >> 3) & (x >> 4) & (x >> 5))) & 1; r = 6
    elif name == "cccz": f = lambda x: (x & (x >> 1) & (x >> 2) & (x >> 3)) & 1; r = 4
    v = np.array([(-1.0) ** f(x) for x in range(1 << r)]) / np.sqrt(1 << r)
    return v.astype(complex), r
if TARGET:
    _idx = np.arange(1 << n)
    _Hm = (-1.0) ** popc_arr(_idx[:, None] & _idx[None, :]); _XB = _idx[:, None] ^ _idx[None, :]
    _spec = lambda u: np.sort(np.round(np.abs((np.conj(u[_XB]) * u[None, :]) @ _Hm).ravel(), 6))
    _t, _r = target_state(TARGET); assert _r == R
    for _ in range(n - R): _t = np.kron(_t, [1, 0])
    _REF = _spec(_t)
def target_ok(AX, BZ, c, sg):
    gens = [(1 if not (sg >> i) & 1 else -1, herm(ax, cc)) for i, (ax, cc) in enumerate(zip(AX, c))]
    gens += [(1 if not (sg >> (len(AX) + k)) & 1 else -1, herm(0, b)) for k, b in enumerate(BZ)]
    v = code_projector(gens, n) @ phi(0, n)
    return np.allclose(_spec(v / np.linalg.norm(v)), _REF)

t0 = time.time()
stats = {"n": n, "d": d, "r": R, "E_size": len(E), "per_a": {}}
survivors = []
for a in range(a_lo, a_hi + 1):
    if a > n or m - a > n - a: continue
    cnt_struct = cnt_pruned = cnt_cand = 0
    for ix, AX in enumerate(subspace_orbits(n, a)):
        if ix % nshards != shard: continue
        AX = list(AX)
        perp = perp_basis(AX, n)
        for BZ in subspaces_within(perp, m - a):
            cnt_struct += 1
            bad = False
            for e in E:
                if all(popcount(r & e) % 2 == 0 for r in AX) and not in_span_rref(e, BZ):
                    bad = True; break
            if bad: cnt_pruned += 1; continue
            basis = lift_space(AX, list(BZ))
            sols = all_combos(basis)
            for start in range(0, len(sols), 1 << 15):
                chunk = sols[start:start + (1 << 15)]
                cs = [(chunk >> (i * n)) & MASKN for i in range(a)]
                elems = group_elements(AX, cs, list(BZ))
                alive, G00 = check(elems)
                cnt_cand += len(chunk) * (1 << m)
                if alive.any():
                    for sg, j in zip(*np.nonzero(alive)):
                        rec = {"a": a, "AX": AX, "BZ": list(BZ), "c": [int(c[j]) for c in cs], "signs": int(sg)}
                        if TARGET and not target_ok(AX, list(BZ), rec["c"], int(sg)): continue
                        survivors.append(rec)
    stats["per_a"][a] = {"structures": cnt_struct, "pruned_by_logical": cnt_pruned, "signed_candidates": cnt_cand}
    print(f"a={a}: structures {cnt_struct}, pruned {cnt_pruned}, signed candidates {cnt_cand}, survivors so far {len(survivors)}, {time.time()-t0:.0f}s", flush=True)
stats["survivors"] = len(survivors); stats["seconds"] = time.time() - t0
json.dump({"stats": stats, "survivors": survivors[:2000]}, open(f"../results/enum{'' if R == 3 else f'_r{R}'}{'_' + TARGET if TARGET else ''}_n{n}_d{d}_a{a_lo}-{a_hi}" + (f"_s{shard}of{nshards}" if nshards > 1 else "") + ".json", "w"), indent=1)
print(json.dumps(stats))
