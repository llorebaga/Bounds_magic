"""Stratified estimate of acceptance and logical error from a TDEM's fault classes (p_i, D_i, o_i).

With independent classes and r_i = p_i / (1 - p_i),
    P_acc / Q = sum_k e_k(r) E_k[1{D=0} a(o)],   P_bad / Q = sum_k e_k(r) E_k[1{D=0} b(o)],
where Q = prod (1 - p_i), e_k is the elementary symmetric polynomial and E_k averages over k-sets drawn with
probability proportional to prod r_i.  Such a set is drawn exactly by k i.i.d. draws with probabilities r_i / R
conditioned on all distinct.  eps = P_bad / P_acc does not need Q.  Strata k <= k_exact can be supplied exactly
(e.g. the full list of malignant configurations of weight < d_target from the fault-distance search)."""
import numpy as np
from fd_fast import _hash


def elementary(r, kmax):
    e = np.zeros(kmax + 1)
    e[0] = 1.0
    for x in r:
        e[1:] = e[1:] + x * e[:-1]
    return e


class Stratified:
    def __init__(self, m, mech=None, seed=1):
        self.m = m
        mech = [x for x in (mech or m.mechanisms()) if x[1] or x[2]]
        self.p = np.array([x[0] for x in mech])
        self.Dm = [x[1] for x in mech]
        nbits = max(D.bit_length() for D in self.Dm) + 1
        self.H = _hash(self.Dm, nbits)
        self.O = np.array([x[2] for x in mech], dtype=np.uint64)
        self.rng = np.random.default_rng(seed)

    def stratum(self, k, samples, scale=1.0, batch=2_000_000):
        """returns (mean of 1{D=0} a, mean of 1{D=0} b, their standard errors, number of distinct samples)"""
        p = np.minimum(self.p * scale, 0.5)
        r = p / (1 - p)
        w = r / r.sum()
        sa = sb = sa2 = sb2 = 0.0
        n = 0
        while n < samples:
            b = min(batch, samples - n)
            idx = self.rng.choice(len(r), size=(b, k), p=w)
            if k > 1:
                s = np.sort(idx, axis=1)
                ok = ~(s[:, 1:] == s[:, :-1]).any(axis=1)
                idx = idx[ok]
            h = np.zeros(len(idx), dtype=np.uint64)
            o = np.zeros(len(idx), dtype=np.uint64)
            for j in range(k):
                h ^= self.H[idx[:, j]]
                o ^= self.O[idx[:, j]]
            hit = np.nonzero(h == 0)[0]
            for t in hit:
                D = 0
                for i in idx[t]:
                    D ^= self.Dm[i]
                if D:
                    continue
                acc, bad = self.m.evaluate(int(o[t]))
                sa += acc
                sb += bad
                sa2 += acc * acc
                sb2 += bad * bad
            n += len(idx)
        ma, mb = sa / n, sb / n
        return ma, mb, np.sqrt(max(sa2 / n - ma * ma, 0) / n), np.sqrt(max(sb2 / n - mb * mb, 0) / n), n

    def estimate(self, scale, kmax, samples, exact_bad=None, exact_acc=None, verbose=False):
        """exact_bad / exact_acc: {k: value of E_k[...]} known exactly (overrides sampling for those k)"""
        p = np.minimum(self.p * scale, 0.5)
        r = p / (1 - p)
        e = elementary(r, kmax)
        A = B = 0.0
        vA = vB = 0.0
        rows = []
        for k in range(kmax + 1):
            if k == 0:
                ma, mb, sa, sb = 1.0, 0.0, 0.0, 0.0
            else:
                ma, mb, sa, sb, n = self.stratum(k, samples if not isinstance(samples, dict) else samples.get(k, 0), scale)
            if exact_bad and k in exact_bad:
                mb, sb = exact_bad[k], 0.0
            if exact_acc and k in exact_acc:
                ma, sa = exact_acc[k], 0.0
            A += e[k] * ma
            B += e[k] * mb
            vA += (e[k] * sa) ** 2
            vB += (e[k] * sb) ** 2
            rows.append((k, e[k], ma, mb))
            if verbose:
                print(f"    k={k}: e_k={e[k]:.3g} E[a]={ma:.4g} E[b]={mb:.4g}", flush=True)
        Q = np.prod(1 - p)
        eps = B / A
        se = eps * np.sqrt(vB / B ** 2 + vA / A ** 2) if B > 0 else 0.0
        return dict(P_acc=Q * A, P_bad=Q * B, eps=eps, se_eps=se, strata=rows)
