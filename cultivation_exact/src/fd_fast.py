"""Vectorized fault-distance search on a list of fault classes (p, D, o) with D, o python ints.

Detector masks are compressed by a random GF(2)-linear 64-bit hash h (h(D1^D2) = h(D1)^h(D2)); every hit is
re-checked on the exact masks.  Observable masks must fit in 64 bits.
  k <= 3: all malignant configurations are listed (indices, o, acc, bad, prob);
  k = 4:  certification only: the distinct observable masks of all D-cancelling 4-sets are evaluated (sets
          with a repeated index reduce to 2-sets, which are already certified harmless when k=4 is reached).
"""
import numpy as np


def _hash(masks, nbits, seed=12345):
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 2 ** 63, size=nbits, dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)
    out = np.zeros(len(masks), dtype=np.uint64)
    for i, D in enumerate(masks):
        h = np.uint64(0)
        b = 0
        while D:
            if D & 1:
                h ^= keys[b]
            D >>= 1
            b += 1
        out[i] = h
    return out


def fault_distance_fast(m, kmax, mech, verbose=True, tol=1e-9, max_list=None, shard=0, nshards=1, kmin=1):
    mech = [x for x in mech if x[1] or x[2]]
    n = len(mech)
    P = np.array([x[0] for x in mech])
    Dm = [x[1] for x in mech]
    nbits = max(D.bit_length() for D in Dm) + 1
    H = _hash(Dm, nbits)
    O = np.array([x[2] for x in mech], dtype=object) if max(x[2] for x in mech).bit_length() > 63 else np.array([x[2] for x in mech], dtype=np.uint64)
    order = np.argsort(H, kind="stable")
    Hs = H[order]

    def harmful(o):
        acc, bad = m.evaluate(int(o))
        return bad > tol, acc, bad

    for k in range(kmin, kmax + 1):
        found = []
        if k == 1:
            for i in np.nonzero(H == 0)[0]:
                if Dm[i] == 0:
                    h, acc, bad = harmful(O[i])
                    if h:
                        found.append(((int(i),), int(O[i]), acc, bad, P[i]))
        elif k == 2:
            # runs of equal hash
            starts = np.r_[0, np.nonzero(np.diff(Hs))[0] + 1]
            ends = np.r_[starts[1:], n]
            for s, e in zip(starts, ends):
                if e - s < 2:
                    continue
                idx = order[s:e]
                for a in range(len(idx)):
                    for b in range(a + 1, len(idx)):
                        i, j = int(idx[a]), int(idx[b])
                        if Dm[i] != Dm[j]:
                            continue
                        o = O[i] ^ O[j]
                        h, acc, bad = harmful(o)
                        if h:
                            found.append(((min(i, j), max(i, j)), int(o), acc, bad, P[i] * P[j]))
        elif k == 3:
            for i in range(shard, n, nshards):
                hij = H[i] ^ H[i + 1:]
                lo = np.searchsorted(Hs, hij, side="left")
                hi = np.searchsorted(Hs, hij, side="right")
                hit = np.nonzero(hi > lo)[0]
                for t in hit:
                    j = i + 1 + int(t)
                    for l in order[lo[t]:hi[t]]:
                        l = int(l)
                        if l <= j or (Dm[i] ^ Dm[j] ^ Dm[l]) != 0:
                            continue
                        o = O[i] ^ O[j] ^ O[l]
                        h, acc, bad = harmful(o)
                        if h:
                            found.append(((i, j, l), int(o), acc, bad, P[i] * P[j] * P[l]))
                            if max_list and len(found) >= max_list:
                                return k, found
        elif k == 4:
            iu, ju = np.triu_indices(n, 1)
            hp = H[iu] ^ H[ju]
            # observable masks as 64-bit words (more than one word when there are more than 64 observables)
            nw = max(1, (max(int(x).bit_length() for x in O) + 63) // 64)
            Ow = np.array([[(int(x) >> (64 * w)) & 0xFFFFFFFFFFFFFFFF for w in range(nw)] for x in O],
                          dtype=np.uint64)
            # unique (hash, o) pairs, then XOR of distinct o's inside each hash group
            key = np.column_stack([hp, Ow[iu] ^ Ow[ju]])
            key = np.unique(key, axis=0)
            hp_u, op_u = key[:, 0], key[:, 1:]
            starts = np.r_[0, np.nonzero(np.diff(hp_u))[0] + 1]
            ends = np.r_[starts[1:], len(hp_u)]
            sizes = ends - starts
            if verbose:
                print(f"  k=4: {len(iu)} pairs, {len(starts)} hash groups, max distinct o per group {sizes.max()}, "
                      f"sum of squares {int((sizes.astype(np.int64) ** 2).sum())}", flush=True)
            allo = set()
            for s, e in zip(starts, ends):
                if e - s < 2:
                    continue
                v = op_u[s:e]
                x = np.unique((v[:, None, :] ^ v[None, :, :]).reshape(-1, nw), axis=0)
                for row in x.tolist():
                    allo.add(sum(int(wd) << (64 * w) for w, wd in enumerate(row)))
            allo.discard(0)
            if verbose:
                print(f"  k=4: {len(allo)} distinct observable masks to evaluate", flush=True)
            for o in allo:
                h, acc, bad = harmful(o)
                if h:
                    found.append((None, int(o), acc, bad, None))
                    if max_list and len(found) >= max_list:
                        break
        else:
            raise ValueError("kmax <= 4")
        if verbose:
            print(f"  k={k}: {len(found)} malignant", flush=True)
        if found:
            return k, found
    return None, []
