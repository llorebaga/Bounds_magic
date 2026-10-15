"""Count (A_X orbit, B_Z) structures surviving the logical pruning, with lift-space dimensions.
Usage: R=3 python count_structs.py n d"""
import sys, os, collections, time
n, d = int(sys.argv[1]), int(sys.argv[2])
saved = sys.argv; sys.argv = ["x", str(n), str(d)]
ns = {}; exec(open("enum_codes.py").read().split("t0 = time.time()")[0], ns); sys.argv = saved
from common import *
m = ns["m"]; E = ns["E"]; t0 = time.time()
for a in range(0, m + 1):
    hist = collections.Counter(); tot = 0
    for AX in subspace_orbits(n, a):
        AX = list(AX); perp = perp_basis(AX, n)
        for BZ in ns["subspaces_within"](perp, m - a):
            tot += 1
            if any(all(popcount(r & e) % 2 == 0 for r in AX) and not in_span_rref(e, BZ) for e in E): continue
            hist[len(ns["lift_space"](AX, list(BZ)))] += 1
    print(f"a={a}: structures {tot}, surviving by lift dim {dict(sorted(hist.items()))}  ({time.time()-t0:.0f}s)", flush=True)
