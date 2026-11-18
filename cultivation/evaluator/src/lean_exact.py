"""Exact real-T evaluation for skeletons with a large affine dimension k (here up to about 30).

The TDEM writes the Choi state of the skeleton as a sum over an affine space x0 + span(X_1..X_k) with phases
c(x + X_i) = i^{lam_i} (-1)^{Z_i . x} c(x), and FastExact needs from it only
  * the Walsh-Hadamard tables v0, v1 of F(x) = c(x) omega^{|x_ref|}, split by the output bit, at the few indices
    A where they do not vanish (A = {0, beta(Z_L)} for deterministic checks),
  * the fault-free output (index 0), the map L (j -> beta(e_j)) and the particular solution x0.
TDEM._affine stores every point as a byte row (k = 27 needs about 14 GB). LeanTDEM instead keeps each point as
two 64-bit words and a phase exponent, forms F directly in complex128 and transforms it in place: about 4.5 GB
at k = 27. LeanExact is FastExact on top of these tables, with the same Gauss sums for every fault class.
Validated against TDEM + FastExact class by class (scripts/lean_check.py)."""
import numpy as np
from tdem import TDEM, popcount64
from branch import FastExact
from gf2 import solve, right_inverse

OMEGA8 = np.exp(1j * np.pi / 4 * np.arange(8)).astype(np.complex128)


def wht_inplace(F):
    """unnormalised Walsh-Hadamard transform in place (length a power of two)"""
    n = len(F)
    h = 1
    while h < n:
        V = F.reshape(-1, 2, h)
        a = V[:, 0, :].copy()
        V[:, 0, :] += V[:, 1, :]
        a -= V[:, 1, :]
        V[:, 1, :] = a
        del a
        h *= 2
    return F


class LeanTDEM(TDEM):
    def _affine(self, light=False):
        assert self.magic == "T"
        t, N = self.t, len(self.S)
        assert N <= 128, "points are packed into two 64-bit words"
        X = np.zeros((N, N), dtype=np.uint8)
        Z = np.zeros((N, N), dtype=np.uint8)
        lam = np.zeros(N, dtype=np.int64)
        for i, Q in enumerate(self.gens):
            x, z = Q.to_numpy()
            X[i], Z[i] = x, z
            lam[i] = (2 if Q.sign == -1 else 0) + int(np.sum(x & z))
        r = 0
        for c in range(N):
            rows = [i for i in range(r, N) if X[i, c]]
            if not rows:
                continue
            p = rows[0]
            if p != r:
                X[[r, p]] = X[[p, r]]
                Z[[r, p]] = Z[[p, r]]
                lam[[r, p]] = lam[[p, r]]
            for i in range(N):
                if i != r and X[i, c]:
                    lam[i] = (lam[i] + lam[r] + 2 * int(np.sum(Z[i] & X[r]))) % 4
                    X[i] ^= X[r]
                    Z[i] ^= Z[r]
            r += 1
        k = r
        self.k = self.kaff = k
        Zc, bc = Z[k:], (lam[k:] // 2) % 2
        assert np.all(lam[k:] % 2 == 0)
        x0 = solve(Zc, bc) if len(Zc) else np.zeros(N, dtype=np.uint8)
        assert x0 is not None

        def pack2(bits):
            lo = hi = 0
            for j in np.nonzero(bits)[0]:
                if j < 64:
                    lo |= 1 << int(j)
                else:
                    hi |= 1 << int(j - 64)
            return np.uint64(lo), np.uint64(hi)

        size = 1 << k
        lo = np.empty(size, dtype=np.uint64)
        hi = np.empty(size, dtype=np.uint64)
        ph = np.empty(size, dtype=np.uint8)              # exponent of i in c(x)
        lo[0], hi[0] = pack2(x0)
        ph[0] = 0
        for i in range(k):
            s = 1 << i
            xl, xh = pack2(X[i])
            zl, zh = pack2(Z[i])
            par = (popcount64(lo[:s] & zl) + popcount64(hi[:s] & zh)) & 1
            lo[s:2 * s] = lo[:s] ^ xl
            hi[s:2 * s] = hi[:s] ^ xh
            ph[s:2 * s] = (ph[:s].astype(np.int64) + lam[i] + 2 * par) % 4
        ref = np.zeros(N, dtype=np.uint8)
        ref[:t] = 1
        rl, rh = pack2(ref)
        w = (popcount64(lo & rl) + popcount64(hi & rh)).astype(np.int64)
        ob = (hi >> np.uint64(t - 64)) & np.uint64(1) if t >= 64 else (lo >> np.uint64(t)) & np.uint64(1)
        xo = ob.astype(bool)
        del lo, hi
        e8 = (2 * ph.astype(np.int64) + w) % 8           # c(x) omega^{|x_ref|} = omega^{2 ph + w}
        del ph, w
        F = OMEGA8[e8]
        del e8
        G = F.copy()
        G[xo] = 0
        F[~xo] = 0
        del xo
        v0 = wht_inplace(G)
        v1 = wht_inplace(F)
        acc = np.abs(v0) ** 2 + np.abs(v1) ** 2
        n0 = acc[0]
        if n0 < 1e-12:
            raise RuntimeError("fault-free output vanishes")
        A = np.nonzero(acc / n0 > 1e-12)[0]
        self.A = A
        self.TA = np.stack([v0[A], v1[A]], axis=1)
        del v0, v1, G, F, acc
        self.n0 = float(n0)
        self.v0 = self.TA[0].copy()
        self.u0 = self.v0 / np.sqrt(self.n0)
        self.Lbits = X[:k, :t].T.copy()                  # t x k: beta(e_j)
        self.x0bits = x0[:t].copy()
        self.phase_pow = OMEGA8[np.arange(t + 1) % 8]
        G2 = np.zeros((N, 2 * t), dtype=np.uint8)
        for i, Q in enumerate(self.gens):
            x, z = Q.to_numpy()
            G2[i, :t] = z[:t]
            G2[i, t:] = x[:t]
        self.Rinv = right_inverse(G2)
        self._cache = {}

class LeanExact(FastExact):
    def __init__(self, m, tol=1e-9, max_enum=22):
        self.m = m
        self.tol = tol
        self.max_enum = max_enum
        self.k = m.kaff
        self.L = m.Lbits.astype(np.uint8)
        self.x0 = m.x0bits.astype(np.uint8)
        self.A = m.A
        self.TA = m.TA
        self.Abits = np.array([[(int(x) >> i) & 1 for i in range(self.k)] for x in self.A], dtype=np.uint8)
        self._cache = {}
        self.max_terms = 0
