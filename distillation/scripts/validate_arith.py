"""Cross-check the exact Z[sqrt2][i] Gram entries of enum_codes.py against dense floating point,
for random enumerated candidates in every X-rank sector.  Usage: python validate_arith.py n d samples"""
import sys, random, numpy as np
from common import *
n, d, samples = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); m = n - 3
saved = sys.argv; sys.argv = ["x", str(n), str(d)]
ns = {}; exec(open("enum_codes.py").read().split("t0 = time.time()")[0], ns); sys.argv = saved
C = ns["C"]; E = ns["E"]
def to_float(V):  # V[...,4] scaled by 2^C
    r2 = np.sqrt(2)
    return (V[..., 0] + r2 * V[..., 1] + 1j * (V[..., 2] + r2 * V[..., 3])) / 2 ** C
worst = 0; tested = 0
for a in range(1, m + 1):
    reps = subspace_orbits(n, a)
    for _ in range(samples):
        AX = list(random.choice(reps)); perp = perp_basis(AX, n)
        BZs = ns["subspaces_within"](perp, m - a)
        if not BZs: continue
        BZ = list(random.choice(BZs)); basis = ns["lift_space"](AX, BZ)
        vec = 0
        for b in basis:
            if random.random() < .5: vec ^= b
        cs = [np.array([(vec >> (i * n)) & ((1 << n) - 1)], np.int64) for i in range(a)]
        elems = ns["group_elements"](AX, cs, BZ)
        sg = random.randrange(1 << m)
        gens = [(1 if not (sg >> i) & 1 else -1, herm(ax, int(c[0]))) for i, (ax, c) in enumerate(zip(AX, cs))]
        gens += [(1 if not (sg >> (a + k)) & 1 else -1, herm(0, b)) for k, b in enumerate(BZ)]
        Pi = code_projector(gens, n)
        for (e, f) in [(0, 0)] + [(0, e) for e in E[:6]] + [(e, e) for e in E[:6]]:
            exact = to_float(ns["wht"](ns["values"](elems, e, f))[sg, 0]) / 2 ** m
            dense = np.vdot(phi(e, n), Pi @ phi(f, n))
            worst = max(worst, abs(exact - dense)); tested += 1
print(f"n={n}: {tested} Gram entries compared, max |exact - dense| = {worst:.2e}")
