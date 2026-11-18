"""Fraction of k-subsets of the data of the 6.6.6 colour codes (d=3,5,7) that lie inside some minimum-weight X logical:
the X patterns a check gadget may not leave undetected before the last T layer (code-relative fault tolerance)."""
import sys, itertools, math
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "src"))
from colorcode import color_code_faces
from foldtree import min_weight_logicals
for bw in (3, 5, 7):
    faces, n = color_code_faces(bw)
    data = list(range(n))
    w, L = min_weight_logicals(data, faces, data)
    print(f"n={n} d={w} min-weight X logicals={len(L)}")
    for k in range(2, w):
        cov = {frozenset(S) for l in L for S in itertools.combinations(sorted(l), k)}
        print(f"   k={k}: {len(cov)}/{math.comb(n,k)} = {len(cov)/math.comb(n,k):.4f}")
