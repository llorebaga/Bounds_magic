"""Completeness check of enum_codes.py: brute-force ALL signed [[n,3]] stabilizer groups (dense
projectors) for small n, collect the Pauli-spectrum invariant of Pi_C phi_0 for every survivor,
and compare with the invariants of the structured search's survivors.
Usage: python validate_enum.py n d"""
import sys, json, subprocess, itertools, numpy as np
from common import *

n = int(sys.argv[1]); d = int(sys.argv[2]); m = n - 3; D = 1 << n
E = [e for e in harmless_set(n, d) if e != 0]
idx = np.arange(D)
Hm = (-1.0) ** popc_arr(idx[:, None] & idx[None, :]); XB = idx[:, None] ^ idx[None, :]
def inv(v):
    v = v / np.linalg.norm(v)
    return tuple(np.sort(np.round(np.abs((np.conj(v[XB]) * v[None, :]) @ Hm).ravel(), 5)))
PH = [phi(e, n) for e in [0] + E]
def survives(Pi):
    v0 = Pi @ PH[0]; g00 = np.vdot(PH[0], v0).real
    if g00 < 1e-9: return None
    for ph in PH[1:]:
        ve = Pi @ ph
        # collinear?
        if np.linalg.norm(ve - v0 * (np.vdot(v0, ve) / np.vdot(v0, v0))) > 1e-7: return None
    return inv(v0)

# all isotropic m-dim subspaces of F2^{2n} (vectors (x<<n)|z), via rref keys
def symp(u, v):
    return (popcount((u >> n) & (v & ((1 << n) - 1))) + popcount((v >> n) & (u & ((1 << n) - 1)))) % 2
brute = set(); seen = set()
def rec(rows, start):
    if len(rows) == m:
        key = rref(rows)
        if key in seen: return
        seen.add(key)
        gens = list(key)
        for signs in itertools.product((1, -1), repeat=m):
            st = [(s, herm(g >> n, g & ((1 << n) - 1))) for s, g in zip(signs, gens)]
            r = survives(code_projector(st, n))
            if r is not None: brute.add(r)
        return
    for v in range(start, 1 << (2 * n)):
        if all(symp(v, r) == 0 for r in rows) and not in_span_rref(v, rref(rows)):
            rec(rows + [v], v + 1)
rec([], 1)
print("brute: groups", len(seen), "distinct survivor invariants", len(brute))

subprocess.run([sys.executable, "enum_codes.py", str(n), str(d)], check=True, capture_output=True)
res = json.load(open(f"../results/enum_n{n}_d{d}_a0-{min(m, n)}.json"))
found = set()
for s in res["survivors"]:
    gens = [(1 if not (s["signs"] >> i) & 1 else -1, herm(ax, c)) for i, (ax, c) in enumerate(zip(s["AX"], s["c"]))]
    gens += [(1 if not (s["signs"] >> (len(s["AX"]) + k)) & 1 else -1, herm(0, b)) for k, b in enumerate(s["BZ"])]
    r = survives(code_projector(gens, n))
    assert r is not None, "enumerator survivor fails dense check"
    found.add(r)
print("enum: survivors", res["stats"]["survivors"], "distinct invariants", len(found))
print("brute invariants missing from enum:", len(brute - found), "| enum-only:", len(found - brute))
