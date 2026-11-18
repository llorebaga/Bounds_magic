"""Listing of malignant weight-4 configurations (needed as ILP constraints when certifying fault distance 5).

Pairs (i<j) are grouped by the hash of D_i ^ D_j; inside a group every pair of pairs with total observable
mask o in the harmful set is a candidate 4-set (exact D re-checked; sets with a repeated index are skipped)."""
import numpy as np
from fd_fast import _hash


def harmful_k4(m, mech, tol=1e-9, verbose=True, max_sets=200000):
    mech = [x for x in mech if x[1] or x[2]]
    n = len(mech)
    Dm = [x[1] for x in mech]
    nbits = max(D.bit_length() for D in Dm) + 1
    H = _hash(Dm, nbits)
    O = np.array([x[2] for x in mech], dtype=np.uint64)
    P = np.array([x[0] for x in mech])
    iu, ju = np.triu_indices(n, 1)
    hp = H[iu] ^ H[ju]
    op = O[iu] ^ O[ju]
    order = np.lexsort((op, hp))
    hp, op, iu, ju = hp[order], op[order], iu[order], ju[order]
    starts = np.r_[0, np.nonzero(np.diff(hp))[0] + 1]
    ends = np.r_[starts[1:], len(hp)]
    # distinct o's of 4-sets, evaluated once
    harm = {}
    sets = []
    total = 0
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        v = op[s:e]
        uv, inv = np.unique(v, return_inverse=True)
        X = uv[:, None] ^ uv[None, :]
        xs = np.unique(X)
        for o in xs.tolist():
            if o == 0 or o in harm:
                continue
            acc, bad = m.evaluate(int(o))
            harm[o] = (acc, bad) if bad > tol else None
        bad_pairs = [(a, b) for a in range(len(uv)) for b in range(a, len(uv)) if harm.get(int(X[a, b]))]
        for a, b in bad_pairs:
            ia = np.nonzero(inv == a)[0] + s
            ib = np.nonzero(inv == b)[0] + s
            for x in ia:
                for y in ib:
                    if y <= x and a == b:
                        continue
                    idx = {int(iu[x]), int(ju[x]), int(iu[y]), int(ju[y])}
                    if len(idx) < 4:
                        continue
                    t = tuple(sorted(idx))
                    D = 0
                    for i in t:
                        D ^= Dm[i]
                    if D:
                        continue
                    o = int(op[x] ^ op[y])
                    acc, bad = harm[o]
                    sets.append((t, o, acc, bad, float(np.prod(P[list(t)]))))
                    total += 1
                    if total >= max_sets:
                        break
    # remove duplicates (a 4-set can split into pairs in up to 3 ways)
    seen = {}
    for s_ in sets:
        seen.setdefault(s_[0], s_)
    out = list(seen.values())
    if verbose:
        print(f"  k=4: {len(harm)} distinct masks evaluated, {sum(1 for v in harm.values() if v)} harmful, "
              f"{len(out)} malignant 4-sets", flush=True)
    return out
