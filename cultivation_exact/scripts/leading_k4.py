"""Exact weight-4 failure term of a real-T circuit of fault distance 4: every malignant 4-set of distinct fault
classes (exact detector patterns cancel, harmful by the Gauss-sum evaluator) with its weight prod(p_i) * bad.

Usage: python leading_k4.py PATH OPTION
Pairs of classes are grouped by a linear hash of their detector patterns; within a group every two pairs with
a harmful combined observable mask and equal exact detector patterns form a candidate 4-set (each 4-set appears
in three splits and is counted once). Prints the number of malignant 4-sets and c4 = sum / p^4 for the circuit's
own noise strength p = 1e-3, and writes results/<stem>_c4.json."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import stim
from tdem import TDEM
from fd_fast import _hash
from branch import FastExact

path, opt = sys.argv[1], sys.argv[2]
kw = {"default": {}, "complement": {"complement": True}, "nopattern": {"pattern": False}}[opt]
t0 = time.time()
m = TDEM(stim.Circuit.from_file(path), magic="T", **kw)
fe = FastExact(m)
mech = [x for x in m.mechanisms() if x[1] or x[2]]
n = len(mech)
P = np.array([x[0] for x in mech])
Dm = [x[1] for x in mech]
O = [int(x[2]) for x in mech]
nw = max(1, (max(o.bit_length() for o in O) + 63) // 64)
Ow = np.array([[(o >> (64 * w)) & 0xFFFFFFFFFFFFFFFF for w in range(nw)] for o in O], dtype=np.uint64)
H = _hash(Dm, max(D.bit_length() for D in Dm) + 1)
iu, ju = np.triu_indices(n, 1)
hp = H[iu] ^ H[ju]
order = np.argsort(hp, kind="stable")
iu, ju, hp = iu[order], ju[order], hp[order]
starts = np.r_[0, np.nonzero(np.diff(hp))[0] + 1]
ends = np.r_[starts[1:], len(hp)]
harm = {}


def bad(o):
    if o not in harm:
        harm[o] = fe.evaluate(o)[1] if o else 0.0
    return harm[o]


sets = {}
for s, e in zip(starts, ends):
    if e - s < 2:
        continue
    pi, pj = iu[s:e], ju[s:e]
    v = Ow[pi] ^ Ow[pj]                                    # pair masks (m x nw)
    X = v[:, None, :] ^ v[None, :, :]
    a_idx, b_idx = np.triu_indices(e - s, 1)
    Xm = X[a_idx, b_idx]
    keys = [sum(int(wd) << (64 * w) for w, wd in enumerate(row)) for row in Xm.tolist()]
    for t, o in enumerate(keys):
        if not o:
            continue
        bo = bad(o)
        if bo <= 1e-12:
            continue
        a, b = int(a_idx[t]), int(b_idx[t])
        S = {int(pi[a]), int(pj[a]), int(pi[b]), int(pj[b])}
        if len(S) < 4:
            continue
        if (Dm[int(pi[a])] ^ Dm[int(pj[a])]) != (Dm[int(pi[b])] ^ Dm[int(pj[b])]):
            continue
        key = tuple(sorted(S))
        if key not in sets:
            sets[key] = (float(np.prod(P[list(key)])), bo, fe.evaluate(o)[0])
tot = sum(pr * bo for pr, bo, _ in sets.values())
accs = sorted({round(a, 9) for _, _, a in sets.values()})
res = dict(circuit=os.path.basename(path), malignant_4sets=len(sets), weight=tot, c4=tot / 1e-3 ** 4,
           distinct_masks_evaluated=len(harm), harmful_masks=sum(1 for v in harm.values() if v > 1e-12),
           acceptances=accs[:10], seconds=round(time.time() - t0))
print(json.dumps(res), flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                                 os.path.splitext(os.path.basename(path))[0] + "_c4.json"), "w"), indent=1)
