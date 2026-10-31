"""P_bad by meet-in-the-middle completion (low-variance estimator for rare failures).

With r_i = p_i/(1-p_i) and Q = prod(1-p_i):  P_bad = Q sum_k A_k,  A_k = sum_{|S|=k} prod r_S [D_S = 0] b(o_S).
Split S = T u P with |P| = 2 (each k-set has C(k,2) splits):
    A_k = e_{k-2}(r) / C(k,2) * E_T[ g(T) ],   g(T) = sum_{P disjoint from T, D_P = D_T} r_P b(o_T ^ o_P),
with T drawn exactly ~ prod r_T (i.i.d. draws conditioned on distinct).  k = 2 and k = 3 are exact sums.
Acceptance comes from direct sampling (it is not rare)."""
import math
import numpy as np
from fd_fast import _hash
from stratified import elementary


class MITM:
    def __init__(self, m, mech=None, seed=7):
        self.m = m
        mech = [x for x in (mech or m.mechanisms()) if x[1] or x[2]]
        self.p0 = np.array([x[0] for x in mech])
        self.Dm = [x[1] for x in mech]
        self.H = _hash(self.Dm, max(D.bit_length() for D in self.Dm) + 1)
        self.O = np.array([x[2] for x in mech], dtype=np.uint64)
        self.n = len(mech)
        iu, ju = np.triu_indices(self.n, 1)
        hp = self.H[iu] ^ self.H[ju]
        order = np.argsort(hp, kind="stable")
        self.pi, self.pj, self.ph = iu[order], ju[order], hp[order]
        self.rng = np.random.default_rng(seed)
        self.cache_b = {}

    def b(self, o):
        o = int(o)
        v = self.cache_b.get(o)
        if v is None:
            v = self.m.evaluate(o)[1]
            self.cache_b[o] = v
        return v

    def g(self, T, r):
        """sum over pairs P disjoint from T with D_P = D_T of r_P b(o_T ^ o_P)"""
        h = np.uint64(0)
        o = np.uint64(0)
        D = 0
        for i in T:
            h ^= self.H[i]
            o ^= self.O[i]
            D ^= self.Dm[i]
        lo = np.searchsorted(self.ph, h, "left")
        hi = np.searchsorted(self.ph, h, "right")
        s = 0.0
        Ts = set(T)
        for t in range(lo, hi):
            i, j = int(self.pi[t]), int(self.pj[t])
            if i in Ts or j in Ts or (self.Dm[i] ^ self.Dm[j]) != D:
                continue
            bb = self.b(o ^ self.O[i] ^ self.O[j])
            if bb:
                s += r[i] * r[j] * bb
        return s

    def P_bad(self, scale, kmax, samples, verbose=False):
        p = np.minimum(self.p0 * scale, 0.5)
        r = p / (1 - p)
        e = elementary(r, kmax)
        Q = float(np.prod(1 - p))
        tot = var = 0.0
        rows = []
        for k in range(2, kmax + 1):
            if k == 2:
                Ak = self.g((), r) / 1.0          # T empty: pairs with D_P = 0
                se = 0.0
            elif k == 3:
                vals = np.array([r[i] * self.g((i,), r) for i in range(self.n)])
                Ak = vals.sum() / 3.0
                se = 0.0
            else:
                w = r / r.sum()
                vals = []
                while len(vals) < samples:
                    idx = self.rng.choice(self.n, size=k - 2, p=w)
                    if len(set(idx.tolist())) < k - 2:
                        continue
                    vals.append(self.g(tuple(int(x) for x in idx), r))
                vals = np.array(vals)
                c = e[k - 2] / math.comb(k, 2)
                Ak = c * vals.mean()
                se = c * vals.std() / np.sqrt(len(vals))
            tot += Ak
            var += se ** 2
            rows.append((k, Ak * Q, se * Q))
            if verbose:
                print(f"    k={k}: P_bad contribution {Ak * Q:.4g} +- {se * Q:.2g}", flush=True)
        return Q * tot, Q * np.sqrt(var), rows
