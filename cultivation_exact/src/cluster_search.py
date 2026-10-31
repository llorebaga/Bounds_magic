"""Sharded exact searches for malignant fault configurations of weight 3, 4 and 5 (cluster workers).

All searches use the TDEM fault classes (p_i, D_i, o_i). Detector masks are compressed by a random GF(2)-linear
64-bit hash h (every hit is re-checked on the exact masks); observable masks are kept as W 64-bit words, so any
number of T gates is allowed.  A configuration is malignant when its total D is 0 and the exact TDEM output is
harmful (bad weight > tol).

  k3: outer index i in the shard (i % nshards == shard); all j > i, l > j with D_i^D_j^D_l = 0.
  k4: pairs (i<j) grouped by h(D_i^D_j); a shard owns the groups with h % nshards == shard; inside a group every
      pair of pairs with disjoint indices is a 4-set (each 4-set is counted once, by its canonical split).
  k5: canonical split a<b<c (triple) | d<e (pair) with c < d; the shard owns a % nshards == shard; the pair
      table is looked up by h(D_a^D_b^D_c).
Each worker returns the number of malignant sets, their summed weight sum prod(p_i) * bad, and up to `keep`
examples."""
import numpy as np

from fd_fast import _hash


class Classes:
    def __init__(self, m, mech=None):
        mech = [x for x in (mech or m.mechanisms()) if x[1] or x[2]]
        self.m = m
        self.n = len(mech)
        self.P = np.array([x[0] for x in mech])
        self.Dm = [x[1] for x in mech]
        self.H = _hash(self.Dm, max(D.bit_length() for D in self.Dm) + 1)
        nobs = len(m.gens)
        self.W = (nobs + 63) // 64
        O = np.zeros((self.n, self.W), dtype=np.uint64)
        for i, x in enumerate(mech):
            o = x[2]
            for w in range(self.W):
                O[i, w] = np.uint64((o >> (64 * w)) & ((1 << 64) - 1))
        self.O = O
        self.cache = {}
        self.fp_bad = {}

    def fingerprint(self, rows):
        """XOR-linear 64-bit fingerprint of multi-word masks: word w rotated by 23 w bits (rotation is linear over
        XOR, so fp(x ^ y) = fp(x) ^ fp(y)); collisions only cost an extra exact check in k4 (bad() is exact)"""
        f = np.zeros(len(rows), dtype=np.uint64)
        for w in range(self.W):
            r = np.uint64((23 * w) % 64)
            v = rows[:, w]
            f ^= (v << r) | (v >> (np.uint64(64) - r)) if r else v
        return f

    def to_int(self, row):
        v = 0
        for w in range(self.W):
            v |= int(row[w]) << (64 * w)
        return v

    def bad(self, row, tol=1e-9):
        o = self.to_int(row)
        r = self.cache.get(o)
        if r is None:
            acc, b = self.m.evaluate(o) if o else (1.0, 0.0)
            r = b if b > tol else 0.0
            self.cache[o] = r
        return r

    def pairs(self):
        iu, ju = np.triu_indices(self.n, 1)
        hp = self.H[iu] ^ self.H[ju]
        order = np.argsort(hp, kind="stable")
        return iu[order], ju[order], hp[order]


def _exact_zero(C, idx):
    D = 0
    for i in idx:
        D ^= C.Dm[i]
    return D == 0


def k3(C, shard, nshards, keep=20):
    H, O, P = C.H, C.O, C.P
    order = np.argsort(H, kind="stable")
    Hs = H[order]
    cnt, wsum, ex = 0, 0.0, []
    for i in range(shard, C.n, nshards):
        hij = H[i] ^ H[i + 1:]
        lo = np.searchsorted(Hs, hij, "left")
        hi = np.searchsorted(Hs, hij, "right")
        for t in np.nonzero(hi > lo)[0]:
            j = i + 1 + int(t)
            for l in order[lo[t]:hi[t]]:
                l = int(l)
                if l <= j or not _exact_zero(C, (i, j, l)):
                    continue
                b = C.bad(O[i] ^ O[j] ^ O[l])
                if b:
                    cnt += 1
                    w = P[i] * P[j] * P[l] * b
                    wsum += w
                    if len(ex) < keep:
                        ex.append([i, j, l, b])
    return dict(k=3, malignant=cnt, weight=wsum, examples=ex, evaluated=len(C.cache))


def k4(C, shard, nshards, keep=20):
    """per hash group: distinct pair masks u, all XORs u_a ^ u_b evaluated once; 4-sets are enumerated only for
    harmful XORs, each counted once by its canonical split {s0 s1 | s2 s3}"""
    iu, ju, hp = C.pairs()
    starts = np.r_[0, np.nonzero(np.diff(hp))[0] + 1]
    ends = np.r_[starts[1:], len(hp)]
    O, P = C.O, C.P
    cnt, wsum, ex = 0, 0.0, []
    for s, e in zip(starts, ends):
        if e - s < 2 or int(hp[s] % np.uint64(nshards)) != shard:
            continue
        op = O[iu[s:e]] ^ O[ju[s:e]]                       # (g, W)
        uv, inv = np.unique(op, axis=0, return_inverse=True)
        inv = inv.ravel()
        nu = len(uv)
        fp = C.fingerprint(uv)                             # XOR-linear 64-bit fingerprints of the distinct masks
        harmful = []
        for a0 in range(0, nu, 512):                       # blocks of rows a, all b >= a
            rows = np.arange(a0, min(nu, a0 + 512))
            F = fp[rows][:, None] ^ fp[None, :]
            A, B = np.nonzero((F != 0) & (np.arange(nu)[None, :] >= rows[:, None]))
            if len(A) == 0:
                continue
            Fv = F[A, B]
            # evaluate each new fingerprint once (exact mask from its first representative)
            uq, first = np.unique(Fv, return_index=True)
            for f, t in zip(uq.tolist(), first.tolist()):
                if f not in C.fp_bad:
                    C.fp_bad[f] = C.bad(uv[rows[A[t]]] ^ uv[B[t]])
            bad_f = np.array([C.fp_bad[f] > 0 for f in uq.tolist()])
            if not bad_f.any():
                continue
            hit = np.isin(Fv, uq[bad_f])
            harmful += [(int(rows[A[t]]), int(B[t])) for t in np.nonzero(hit)[0]]
        if not harmful:
            continue
        members = [np.nonzero(inv == a)[0] + s for a in range(nu)]
        for a, b in harmful:
            for x in members[a]:
                for y in members[b]:
                    if a == b and y <= x:
                        continue
                    i1, j1, i2, j2 = int(iu[x]), int(ju[x]), int(iu[y]), int(ju[y])
                    idx = {i1, j1, i2, j2}
                    if len(idx) < 4:
                        continue
                    srt = sorted(idx)
                    if (srt[0], srt[1]) not in ((i1, j1), (i2, j2)):
                        continue
                    if not _exact_zero(C, srt):
                        continue
                    bb = C.bad(O[i1] ^ O[j1] ^ O[i2] ^ O[j2])
                    if bb:
                        cnt += 1
                        wsum += P[i1] * P[j1] * P[i2] * P[j2] * bb
                        if len(ex) < keep:
                            ex.append(srt + [bb])
    return dict(k=4, malignant=cnt, weight=wsum, examples=ex, evaluated=len(C.cache))


def k5(C, shard, nshards, keep=20):
    iu, ju, hp = C.pairs()
    H, O, P = C.H, C.O, C.P
    cnt, wsum, ex = 0, 0.0, []
    for a in range(shard, C.n, nshards):
        for b in range(a + 1, C.n):
            hab = H[a] ^ H[b]
            cs = np.arange(b + 1, C.n)
            if len(cs) == 0:
                continue
            h3 = hab ^ H[cs]
            lo = np.searchsorted(hp, h3, "left")
            hi = np.searchsorted(hp, h3, "right")
            for t in np.nonzero(hi > lo)[0]:
                c = int(cs[t])
                for x in range(lo[t], hi[t]):
                    d, e = int(iu[x]), int(ju[x])
                    if d <= c:
                        continue
                    if not _exact_zero(C, (a, b, c, d, e)):
                        continue
                    bb = C.bad(O[a] ^ O[b] ^ O[c] ^ O[d] ^ O[e])
                    if bb:
                        cnt += 1
                        w = P[a] * P[b] * P[c] * P[d] * P[e] * bb
                        wsum += w
                        if len(ex) < keep:
                            ex.append([a, b, c, d, e, bb])
    return dict(k=5, malignant=cnt, weight=wsum, examples=ex, evaluated=len(C.cache))
