"""For survivors of enum_codes.py: verify the rank-one Gram condition densely and test whether the
decoded state Pi_C phi_0 has the Pauli spectrum of CCZ (x) |0>^{n-3} (necessary for Clifford
equivalence to CCZ).  Usage: python target_check.py results.json"""
import sys, json, numpy as np
from common import *

res = json.load(open(sys.argv[1])); n = res["stats"]["n"]; d = res["stats"]["d"]
D = 1 << n; idx = np.arange(D)
Hm = (-1.0) ** popc_arr(idx[:, None] & idx[None, :]); XB = idx[:, None] ^ idx[None, :]
spec = lambda u: np.sort(np.round(np.abs((np.conj(u[XB]) * u[None, :]) @ Hm).ravel(), 6))
c = np.array([(-1) ** (((i >> 2) & (i >> 1) & i) & 1) for i in range(8)]) / np.sqrt(8)
ref = c.astype(complex)
for _ in range(n - 3): ref = np.kron(ref, [1, 0])
REF = spec(ref)
E = [e for e in harmless_set(n, d) if e != 0]
ok = ccz = 0; classes = {}
for s in res["survivors"]:
    gens = [(1 if not (s["signs"] >> i) & 1 else -1, herm(ax, cc)) for i, (ax, cc) in enumerate(zip(s["AX"], s["c"]))]
    gens += [(1 if not (s["signs"] >> (len(s["AX"]) + k)) & 1 else -1, herm(0, b)) for k, b in enumerate(s["BZ"])]
    Pi = code_projector(gens, n)
    v0 = Pi @ phi(0, n)
    good = np.linalg.norm(v0) > 1e-6 and all(
        np.linalg.norm((w := Pi @ phi(e, n)) - v0 * np.vdot(v0, w) / np.vdot(v0, v0)) < 1e-7 for e in E)
    ok += good
    sp = spec(v0 / np.linalg.norm(v0)); key = tuple(sp)
    classes[key] = classes.get(key, 0) + 1
    ccz += np.allclose(sp, REF)
print(f"survivors {len(res['survivors'])}, dense-verified {ok}, CCZ-spectrum {ccz}, distinct decoded spectra {len(classes)}")
