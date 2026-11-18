"""Exact witnesses for a weight-4 fault distance found by the hashed search of fd_fast.

The k=4 path of fd_fast groups pairs of fault classes by a linear hash of their detector patterns and reports
harmful observable masks without re-checking the exact patterns, so a hash collision could in principle report
a spurious harmful mask. This script re-runs the search, and for every harmful mask looks for a 4-set of
distinct fault classes whose exact detector patterns cancel and whose observable flips give that mask.
Usage: python witness_k4.py PATH [default|complement|nopattern]"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast, _hash
from branch import FastExact

path = sys.argv[1]
opt = sys.argv[2] if len(sys.argv) > 2 else "default"
kw = {"default": {}, "complement": {"complement": True}, "nopattern": {"pattern": False}}[opt]
t0 = time.time()
m = TDEM(stim.Circuit.from_file(path), magic="T", **kw)
fe = FastExact(m)
mech = [x for x in m.mechanisms() if x[1] or x[2]]
k, found = fault_distance_fast(fe, 4, mech, verbose=True)
print(f"search: fault distance {k}, {len(found)} harmful masks ({time.time()-t0:.0f}s)", flush=True)
if k != 4:
    sys.exit(0)
harm = {int(x[1]) for x in found}

n = len(mech)
Dm = [x[1] for x in mech]
Om = [int(x[2]) for x in mech]
P = [x[0] for x in mech]
H = _hash(Dm, max(D.bit_length() for D in Dm) + 1)
iu, ju = np.triu_indices(n, 1)
hp = H[iu] ^ H[ju]
order = np.argsort(hp, kind="stable")
hs = hp[order]
starts = np.r_[0, np.nonzero(np.diff(hs))[0] + 1]
ends = np.r_[starts[1:], len(hs)]
witness = {}
for s, e in zip(starts, ends):
    if e - s < 2:
        continue
    idx = order[s:e]
    pairs = [(int(iu[x]), int(ju[x])) for x in idx]
    po = [Om[i] ^ Om[j] for i, j in pairs]
    pD = [Dm[i] ^ Dm[j] for i, j in pairs]
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            o = po[a] ^ po[b]
            if o not in harm or o in witness:
                continue
            S = set(pairs[a]) | set(pairs[b])
            if len(S) < 4 or pD[a] != pD[b]:          # exact detector check
                continue
            witness[o] = sorted(S)
    if len(witness) == len(harm):
        break
res = []
for o, S in witness.items():
    acc, bad = fe.evaluate(o)
    exact = m.evaluate(o)                              # independent check with the contraction
    res.append(dict(mask=str(o), classes=S, prob=float(np.prod([P[i] for i in S])), acc=acc, bad=bad,
                    acc_contraction=exact[0], bad_contraction=exact[1]))
print(f"exact witnesses for {len(witness)} of {len(harm)} harmful masks ({time.time()-t0:.0f}s)", flush=True)
for r in res[:5]:
    print(" ", r, flush=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                   os.path.splitext(os.path.basename(path))[0] + "_witness_k4.json")
json.dump(dict(circuit=os.path.basename(path), harmful_masks=len(harm), witnessed=len(witness), witnesses=res),
          open(out, "w"), indent=1)
