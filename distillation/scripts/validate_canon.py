"""Completeness test of the parametrization used by enum_codes.py, for every X-rank a.
For random signed stabilizer groups S of rank m=n-3:
  1. find a permutation taking A_X(S) to its orbit representative (as returned by subspace_orbits),
  2. read off B_Z and the lifts c_i (reduced mod B_Z), apply the W-gauge,
  3. check that (A_X, B_Z, c) lies in the lift space enumerated by enum_codes.lift_space,
  4. check that the transformed signed group has the same Gram singular values over H as S
     (dense, independent of the exact enumerator arithmetic).
Usage: python validate_canon.py n d samples"""
import sys, itertools, random, numpy as np
from common import *
sys.argv_saved = sys.argv
n, d, samples = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); m = n - 3
MASK = (1 << n) - 1
import importlib.util
spec_ = importlib.util.spec_from_file_location("ec", "enum_codes.py")
# reuse lift_space without running the search: exec the needed defs only
src = open("enum_codes.py").read().split("t0 = time.time()")[0]
ns = {}; sys.argv = ["x", str(n), str(d)]; exec(src, ns); sys.argv = sys.argv_saved
lift_space = ns["lift_space"]
reps = {a: [tuple(r) for r in subspace_orbits(n, a)] for a in range(0, m + 1)}
perms = list(itertools.permutations(range(n)))
H = harmless_set(n, d); B = np.array([phi(e, n) for e in H]).T

def sym(u, v): return (popcount((u >> n) & v & MASK) + popcount((v >> n) & u & MASK)) % 2
def random_group():
    rows = []
    while len(rows) < m:
        v = random.randrange(1, 1 << (2 * n))
        if all(sym(v, r) == 0 for r in rows) and not in_span_rref(v, rref(rows)): rows.append(v)
    return rows
def all_elems(gens):  # signed elements as (sign, (p,x,z))
    els = [(1, (0, 0, 0))]
    for s, P in gens:
        els += [(sg * s, pmul(Q, P)) for sg, Q in els]
    return els
def gram_sv(gens):
    Pi = code_projector(gens, n)
    return np.round(np.linalg.svd(B.conj().T @ Pi @ B, compute_uv=False), 7)
def apply_perm(P, pi): p, x, z = P; return (p, permute_bits(x, pi), permute_bits(z, pi))
def apply_W(P, t):  # conjugate by W_t = (X+Y)/sqrt2: X<->Y, Z->-Z on qubit t
    p, x, z = P
    xt, zt = (x >> t) & 1, (z >> t) & 1
    if not xt and not zt: return P
    if not xt and zt: return ((p + 2) % 4, x, z)            # Z -> -Z
    # X_t Z_t^zt with canonical bookkeeping: X->Y=iXZ, Y=iXZ -> X  ; Z_t sign flips
    if xt and not zt: return ((p + 1) % 4, x, z ^ (1 << t))  # X -> i X Z (= Y)
    # here operator contains X_t Z_t (= -iY_t); Y -> X and Z->-Z: X Z -> (iXZ)(-Z) = -i X  ... compute directly
    return ((p + 3) % 4, x, z ^ (1 << t))                   # X Z -> -i X

bad = 0; counts = {}
for trial in range(samples):
    rows = random_group(); signs = [random.choice((1, -1)) for _ in rows]
    gens = [(s, herm(r >> n, r & MASK)) for s, r in zip(signs, rows)]
    sv0 = gram_sv(gens)
    AX = rref([r >> n for r in rows]); a = len(AX); counts[a] = counts.get(a, 0) + 1
    target = None
    for pi in perms:
        t = rref([permute_bits(r, pi) for r in AX])
        if t in reps[a]: target = (pi, t); break
    pi, AXr = target
    gens = [(s, apply_perm(P, pi)) for s, P in gens]
    els = all_elems(gens)
    BZ = rref([P[2] for s, P in els if P[1] == 0 and P[2]])
    piv = [r.bit_length() - 1 for r in BZ]
    def red(c):
        for r in BZ:
            if (c >> (r.bit_length() - 1)) & 1: c ^= r
        return c
    # W-gauge: toggle columns
    supp = 0
    for r in AXr: supp |= r
    lift = {}
    for s, P in els:
        if P[1] in AXr: lift[P[1]] = red(P[2])
    for t in range(n):
        if (supp >> t) & 1 and t not in piv:
            i0 = min(i for i in range(a) if (AXr[i] >> t) & 1)
            if (lift[AXr[i0]] >> t) & 1:
                gens = [(s, apply_W(P, t)) for s, P in gens]
                for x in AXr:
                    if (x >> t) & 1: lift[x] ^= 1 << t
    # recompute lifts from transformed group to be safe
    els = all_elems(gens)
    lift2 = {P[1]: red(P[2]) for s, P in els if P[1] in AXr}
    assert lift2 == lift, "gauge bookkeeping mismatch"
    vec = 0
    for i, x in enumerate(AXr): vec |= lift[x] << (i * n)
    basis = lift_space(list(AXr), list(BZ))
    if not in_span_rref(vec, rref(basis)): bad += 1; print("NOT COVERED", trial, a); continue
    sv1 = gram_sv(gens)
    if not np.allclose(sv0, sv1, atol=1e-6): bad += 1; print("GRAM CHANGED", trial, a, sv0[:3], sv1[:3])
print(f"n={n}: {samples} random groups, by X-rank {dict(sorted(counts.items()))}, failures {bad}")
