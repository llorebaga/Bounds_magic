"""Export (A_X, B_Z, lift basis) structures that survive the logical pruning, for fast_enum.jl.
Line format:  a | AX ints | BZ ints | lift-basis ints   (space separated within fields)
Usage: R=3 python export_structs.py n d a [shard nshards]"""
import sys, os
n, d, a = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
shard = int(sys.argv[4]) if len(sys.argv) > 4 else 0
nsh = int(sys.argv[5]) if len(sys.argv) > 5 else 1
saved = sys.argv; sys.argv = ["x", str(n), str(d)]
ns = {}; exec(open(os.environ.get("ENUM", "enum_codes_v2.py")).read().split("t0 = time.time()")[0], ns); sys.argv = saved
from common import *
m = ns["m"]; E = ns["E"]; R = ns["R"]
out = open(f"../results/structs{'' if ns.get('GALOIS', 1) else '_nogal'}_r{R}_n{n}_d{d}_a{a}" + (f"_s{shard}" if nsh > 1 else "") + ".txt", "w")
cnt = 0
for ix, AX in enumerate(subspace_orbits(n, a)):
    if ix % nsh != shard: continue
    AX = list(AX); perp = perp_basis(AX, n)
    for BZ in ns["subspaces_within"](perp, m - a):
        if any(all(popcount(r & e) % 2 == 0 for r in AX) and not in_span_rref(e, BZ) for e in E): continue
        basis = ns["lift_space"](AX, list(BZ))
        out.write(f"{a} | {' '.join(map(str, AX))} | {' '.join(map(str, BZ))} | {' '.join(map(str, basis))}\n")
        cnt += 1
out.close(); print("exported", cnt)
