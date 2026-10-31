"""Logical error and acceptance of real-T circuits with the Gauss-sum evaluator (branch.FastExact).

Same estimators as src/mitm_rate.py (P_bad by meet-in-the-middle completion, exact for the strata k = 2, 3 and
unbiased for k >= 4) and src/mc.py (P_acc by sampling the detector error model with the exact acceptance of
each sampled class), but with observable masks as Python integers, so circuits with more than 64 observables
(t >= 64 T gates, e.g. d = 7) are supported, and with every class evaluated by Gauss sums."""
import math
import numpy as np
from fd_fast import _hash
from stratified import elementary


class FastMITM:
    def __init__(self, fe, mech=None, seed=7):
        self.fe = fe
        mech = [x for x in (mech or fe.m.mechanisms()) if x[1] or x[2]]
        self.p0 = np.array([x[0] for x in mech])
        self.Dm = [x[1] for x in mech]
        self.O = [int(x[2]) for x in mech]
        self.H = _hash(self.Dm, max(D.bit_length() for D in self.Dm) + 1)
        self.n = len(mech)
        iu, ju = np.triu_indices(self.n, 1)
        hp = self.H[iu] ^ self.H[ju]
        order = np.argsort(hp, kind="stable")
        self.pi, self.pj, self.ph = iu[order], ju[order], hp[order]
        self.rng = np.random.default_rng(seed)

    def b(self, o):
        bb = self.fe.evaluate(o)[1] if o else 0.0
        return bb if bb > 1e-12 else 0.0              # rounding noise of harmless classes

    def g(self, T, r):
        """sum over pairs P disjoint from T with D_P = D_T of r_P b(o_T ^ o_P)"""
        h = np.uint64(0)
        o = 0
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
                Ak, se = self.g((), r), 0.0
            elif k == 3:
                Ak, se = sum(r[i] * self.g((i,), r) for i in range(self.n)) / 3.0, 0.0
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


def acceptance(fe, dem, shots, batch=1_000_000, seed=1):
    """P_acc = E[1{D=0} acc(o)] and its standard error, by sampling the detector error model"""
    sampler = dem.compile_sampler(seed=seed)
    tot = tot2 = 0.0
    n = 0
    while n < shots:
        b = min(batch, shots - n)
        det, obs, _ = sampler.sample(b, bit_packed=True)
        keep = ~det.any(axis=1)
        ob = obs[keep]
        if len(ob):
            keys = [int.from_bytes(row.tobytes(), "little") for row in ob]
            uniq, cnt = np.unique(np.array(keys, dtype=object), return_counts=True)
            for key, c in zip(uniq, cnt):
                a = fe.evaluate(int(key))[0] if key else 1.0
                tot += c * a
                tot2 += c * a * a
        n += b
    Pa = tot / n
    return Pa, math.sqrt(max(tot2 / n - Pa ** 2, 0) / n)
