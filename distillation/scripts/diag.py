"""Diagnostics: for each signed candidate (after logical pruning), the first failing condition
in the order G00=0, then single-fault patterns (weight 1), then weight n-1, then the all-fault pattern,
then other weights.  Usage: python diag.py n d"""
import sys, numpy as np, collections
n, d = int(sys.argv[1]), int(sys.argv[2])
saved = sys.argv; sys.argv = ["x", str(n), str(d)]
ns = {}; exec(open("enum_codes.py").read().split("t0 = time.time()")[0], ns); sys.argv = saved
from common import *
E = sorted(ns["E"], key=lambda e: (min(popcount(e), n - popcount(e)), -popcount(e)))
m = n - 3; MASKN = (1 << n) - 1
fail = collections.Counter(); per_a = collections.Counter()
for a in range(0, m + 1):
    for AX in subspace_orbits(n, a):
        AX = list(AX); perp = perp_basis(AX, n)
        for BZ in ns["subspaces_within"](perp, m - a):
            if any(all(popcount(r & e) % 2 == 0 for r in AX) and not in_span_rref(e, BZ) for e in E): continue
            sols = ns["all_combos"](ns["lift_space"](AX, list(BZ)))
            cs = [(sols >> (i * n)) & MASKN for i in range(a)]
            elems = ns["group_elements"](AX, cs, list(BZ))
            G00 = ns["wht"](ns["values"](elems, 0, 0))
            alive = (G00[..., 0] != 0) | (G00[..., 1] != 0)
            fail["G00=0"] += int((~alive).sum()); per_a[a] += alive.size
            for e in E:
                G0e = ns["wht"](ns["values"](elems, 0, e)); Gee = ns["wht"](ns["values"](elems, e, e))
                ok = (ns["zmul_real"](G00[..., 0:2], Gee[..., 0:2]) == ns["abs2"](G0e)).all(-1)
                newly = alive & ~ok
                fail[f"weight {popcount(e)}"] += int(newly.sum()); alive &= ok
                if not alive.any(): break
            fail["SURVIVE"] += int(alive.sum())
print("signed candidates by X-rank:", dict(per_a)); print("first failure:", dict(fail))
