"""Branch screen for the real-T harm test (Lemma 3 made fast).

For a fault class with observable flips o, the TDEM gives E = R(o) = X^a Z^b on the T inputs.  Since
X|T> = (|T> + iZ|T>)/sqrt2, E|T>^t is a superposition of the branches Z^{b+w}|T>^t, w subset of supp(a), and
the class is harmless whenever every branch is (Lemma 3).  The output of a pure Z pattern Z^e depends on e
only through beta(e) in F_2^k (k = affine dimension of the Choi state), so ONE Walsh-Hadamard transform at
a = 0 tabulates the bad weight of every Z pattern.  A class is then screened by checking whether the coset
beta(b) + span{beta(e_j): j in supp a} meets the harmful set: table lookups instead of an exact contraction.

BranchScreen.evaluate(o)  -> (1, 1) if some branch is harmful, else (1, 0): fault distance FD_branch <= FD_real.
Hybrid(m).evaluate(o)     -> exact (acc, bad), but the exact contraction runs only for classes the screen
                             cannot clear; classes that the screen flags but are exactly harmless are counted
                             (coherent cancellation)."""
import numpy as np
from tdem import wht, popcount64


class BranchScreen:
    def __init__(self, m, tol=1e-9):
        self.m = m
        self.tol = tol
        k, t = m.kaff, m.t
        cols = np.zeros(t, dtype=np.int64)
        for i in range(k):
            for j in range(t):
                if (int(m.U_ref[i, j // 64]) >> (j % 64)) & 1:
                    cols[j] |= 1 << i
        self.cols = cols
        w = np.zeros(len(m.cs), dtype=np.int64)
        for q in range(m.W):
            w += popcount64(m.xr[:, q])
        F = m.cs * m.phase_pow[w]
        v0, v1 = wht(np.where(m.xo, 0, F)), wht(np.where(m.xo, F, 0))
        acc = (np.abs(v0) ** 2 + np.abs(v1) ** 2) / m.n0
        good = np.abs(np.conj(m.u0[0]) * v0 + np.conj(m.u0[1]) * v1) ** 2 / m.n0
        self.badw = acc - good
        self.bad = self.badw > tol
        assert abs(acc[0] - 1) < 1e-9 and not self.bad[0], "beta = 0 must be the fault-free output"
        self.n_harmful_patterns = int(self.bad.sum())
        self._cache = {}

    def branches(self, o):
        """array of beta indices of all branches of class o"""
        a, b = self.m.R_of(o)
        beta0 = int(np.bitwise_xor.reduce(self.cols[b == 1])) if b.any() else 0
        basis = []
        for v in self.cols[a == 1].tolist():
            for u in basis:
                v = min(v, v ^ u)
            if v:
                basis.append(v)
        arr = np.array([beta0], dtype=np.int64)
        for v in basis:
            arr = np.concatenate([arr, arr ^ v])
        return arr

    def harmful(self, o):
        if o not in self._cache:
            self._cache[o] = bool(self.bad[self.branches(o)].any()) if o else False
        return self._cache[o]

    def evaluate(self, o):
        return (1.0, 1.0 if self.harmful(o) else 0.0)


def _add_parity(c, L, B, a, t, U):
    """add a * (t XOR XOR_{u in U} z_u) to the Z_4 form: the XOR of bits is sum b - 2 sum_{pairs} b b' mod 4"""
    c = (c + a * t) % 4
    for u in U:
        L[u] = (L[u] + a - 2 * a * t) % 4
    if a & 1:
        for x in range(len(U)):
            for y in range(x + 1, len(U)):
                B[U[x]][U[y]] ^= 1
                B[U[y]][U[x]] ^= 1
    return c


def z4_gauss_sum(c, L, B):
    """sum over z in F_2^n of i^{c + L.z + 2 sum_{l<m} B_lm z_l z_m} (L in Z_4^n, B symmetric 0/1, zero diagonal),
    by eliminating one variable at a time (O(n^3))"""
    L = [int(x) % 4 for x in L]
    n = len(L)
    B = [[int(B[i][j]) & 1 for j in range(n)] for i in range(n)]
    c = int(c) % 4
    pref = 1 + 0j
    alive = list(range(n))
    while alive:
        v = alive.pop()
        nb = [j for j in alive if B[v][j]]
        if L[v] % 2 == 0:
            # sum over z_v: 1 + (-1)^{L_v/2 + parity(nb)} = 2 [parity(nb) = L_v/2]
            t = (L[v] // 2) & 1
            if not nb:
                if t:
                    return 0j
                pref *= 2
                continue
            pref *= 2
            m, U = nb[-1], nb[:-1]
            alive.remove(m)
            c = _add_parity(c, L, B, L[m], t, U)              # L_m z_m with z_m = t + sum_U z_u
            for j in alive:
                if B[m][j]:                                     # 2 z_m z_j = 2 z_j (t + sum_U z_u)
                    L[j] = (L[j] + 2 * t) % 4
                    for u in U:
                        if u == j:
                            L[j] = (L[j] + 2) % 4
                        else:
                            B[u][j] ^= 1
                            B[j][u] ^= 1
        else:
            # 1 + i^{L_v} (-1)^y = (1+i) i^{3y} for L_v = 1, (1-i) i^{y} for L_v = 3
            if L[v] == 1:
                pref *= 1 + 1j
                c = _add_parity(c, L, B, 3, 0, nb)
            else:
                pref *= 1 - 1j
                c = _add_parity(c, L, B, 1, 0, nb)
    return pref * 1j ** c


class FastExact:
    """Exact (acc, bad) of every class from the Z-pattern table alone.

    With phi_j = omega^{x_j XOR 1} = (omega^{x_j} + i omega^{x_j} (-1)^{x_j})/sqrt2 on every X-hit input,
        out(a, b) = 2^{-|a|/2} (-1)^{b.x0} sum_{w subset a} i^{|w|} (-1)^{w.x0} T[beta(b) + beta(w)],
    where T[beta] is the output of the Z pattern with index beta (one WHT).  Only the few indices A with
    T[beta] != 0 contribute (for cultivation circuits A = {0, beta(Z_L)}), so per class we need
        g(gamma) = sum_{w: L w = gamma} prod_{j in w} i (-1)^{x0_j},   L = [beta(e_j)]_{j in a},
    a character sum over an affine subspace, evaluated over the kernel of L (2^{|a|-r} terms) or over its row
    space (2^r terms), whichever is smaller."""
    def __init__(self, m, tol=1e-9, max_enum=22):
        self.m = m
        self.tol = tol
        self.max_enum = max_enum
        scr = BranchScreen(m, tol)
        k, t = m.kaff, m.t
        self.k = k
        self.L = np.array([[(c >> i) & 1 for i in range(k)] for c in scr.cols.tolist()], dtype=np.uint8)  # t x k
        self.x0 = np.array([(int(m.x0_ref[j // 64]) >> (j % 64)) & 1 for j in range(t)], dtype=np.uint8)
        w = np.zeros(len(m.cs), dtype=np.int64)
        for q in range(m.W):
            w += popcount64(m.xr[:, q])
        F = m.cs * m.phase_pow[w]
        v0, v1 = wht(np.where(m.xo, 0, F)), wht(np.where(m.xo, F, 0))
        acc = (np.abs(v0) ** 2 + np.abs(v1) ** 2) / m.n0
        self.A = np.nonzero(acc > 1e-12)[0]
        self.TA = np.stack([v0[self.A], v1[self.A]], axis=1)          # |A| x 2
        self.Abits = np.array([[(int(x) >> i) & 1 for i in range(k)] for x in self.A], dtype=np.uint8)
        self._cache = {}
        self.max_terms = 0

    def _g(self, J, gammas):
        """g(gamma) by a Z_4 Gauss sum over the kernel parametrisation w = w_p + sum_l z_l k_l:
        i^{|w|} (-1)^{x0.w} = i^{c + L.z + 2 sum_{l<m} B_lm z_l z_m} with c = |w_p| + 2 x0.w_p,
        L_l = |k_l| - 2|w_p & k_l| + 2 x0.k_l, B_lm = |k_l & k_m| mod 2 (inclusion-exclusion mod 4)."""
        from gf2 import rref
        Lj = self.L[J].T
        nJ = len(J)
        # one RREF of [L_J | gammas] gives the kernel and a particular solution for every gamma
        R, piv = rref(np.hstack([Lj, np.asarray(gammas, dtype=np.uint8).T]))
        pl = [p for p in piv if p < nJ]
        r = len(pl)
        free = [f for f in range(nJ) if f not in set(pl)]
        K = np.zeros((len(free), nJ), dtype=np.int64)
        for i, f in enumerate(free):
            K[i, f] = 1
            for row, p in enumerate(pl):
                K[i, p] = R[row, f]
        x0 = self.x0[J].astype(np.int64)
        B = (K @ K.T) % 2
        np.fill_diagonal(B, 0)
        out = np.zeros(len(gammas), dtype=complex)
        for n in range(len(gammas)):
            col = R[:, nJ + n]
            if col[r:].any():
                continue
            wp = np.zeros(nJ, dtype=np.int64)
            for row, p in enumerate(pl):
                wp[p] = col[row]
            c = int(wp.sum() + 2 * (x0 @ wp))
            Lz = (K.sum(axis=1) - 2 * (K @ wp) + 2 * (K @ x0)) % 4
            out[n] = z4_gauss_sum(c, Lz, B)
        return out

    def _g_enum(self, J, gammas):
        """g(gamma) for every row gamma of `gammas` (bits over the k affine coordinates), by enumeration"""
        from gf2 import rref, solve, nullspace
        Lj = self.L[J].T                                   # k x |J|
        s = np.where(self.x0[J], -1j, 1j)                  # i (-1)^{x0_j}
        R, piv = rref(Lj.copy())
        r = len(piv)
        out = np.zeros(len(gammas), dtype=complex)
        if r <= len(J) - r:
            rows = R[:r]                                   # basis of the row space of L_J
            # pivot coordinates of the RREF are the k-coordinates p with row(p) independent; y over them
            Y = ((np.arange(1 << r)[:, None] >> np.arange(r)[None, :]) & 1).astype(np.uint8)
            V = (Y.astype(np.int64) @ rows.astype(np.int64)) % 2          # 2^r x |J|
            prod = np.prod(1 + s[None, :] * (1 - 2 * V), axis=1)
            self.max_terms = max(self.max_terms, 1 << r)
            for n, gam in enumerate(gammas):
                # gamma must be in the column space; express it in the RREF basis: rows = E Lj, so the
                # character of row combination y on gamma is y.(E gamma)
                Eg = self._E(Lj, gam)
                if Eg is None:
                    continue
                sg = 1 - 2 * ((Y.astype(np.int64) @ Eg[:r].astype(np.int64)) % 2)
                out[n] = (sg * prod).sum() / (1 << r)
        else:
            K = nullspace(Lj)
            K = K.reshape(-1, len(J)) if K.size else np.zeros((0, len(J)), dtype=np.uint8)
            dk = len(K)
            Z = ((np.arange(1 << dk)[:, None] >> np.arange(dk)[None, :]) & 1).astype(np.uint8)
            Kz = (Z.astype(np.int64) @ K.astype(np.int64)) % 2 if dk else np.zeros((1, len(J)), dtype=np.int64)
            self.max_terms = max(self.max_terms, 1 << dk)
            for n, gam in enumerate(gammas):
                wp = solve(Lj, gam)
                if wp is None:
                    continue
                W = (Kz + wp[None, :]) % 2
                out[n] = np.prod(np.where(W == 1, s[None, :], 1), axis=1).sum()
        return out

    def _E(self, Lj, gam):
        """row operations E with E Lj = RREF(Lj), applied to gamma (None if gamma not in the column space)"""
        k, n = Lj.shape
        M = np.hstack([Lj, np.eye(k, dtype=np.uint8)])
        from gf2 import rref
        R, piv = rref(M.copy())
        piv_left = [p for p in piv if p < n]
        E = R[:, n:]
        Eg = (E.astype(np.int64) @ gam.astype(np.int64)) % 2
        if Eg[len(piv_left):].any():
            return None
        return Eg

    def evaluate(self, o):
        if o in self._cache:
            return self._cache[o]
        m = self.m
        if o == 0:
            return (1.0, 0.0)
        a, b = m.R_of(o)
        J = np.nonzero(a)[0]
        bb = (self.L[b == 1].sum(axis=0) % 2).astype(np.uint8) if b.any() else np.zeros(self.k, np.uint8)
        sgn = -1 if int(self.x0[b == 1].sum()) % 2 else 1
        gammas = (self.Abits ^ bb[None, :]).astype(np.uint8)
        if len(J) == 0:
            g = np.array([1.0 + 0j if not gm.any() else 0j for gm in gammas])
        else:
            g = self._g(J, gammas) / 2 ** (len(J) / 2)
        v = sgn * (g[:, None] * self.TA).sum(axis=0)
        acc = float(np.vdot(v, v).real / m.n0)
        good = float(abs(np.vdot(m.u0, v)) ** 2 / m.n0)
        res = (acc, max(acc - good, 0.0))
        self._cache[o] = res
        return res


class Hybrid:
    def __init__(self, m, tol=1e-9):
        self.m = m
        self.tol = tol
        self.screen = BranchScreen(m, tol)
        self.stats = dict(cleared=0, exact=0, cancelled=0)
        self._cache = {}

    def evaluate(self, o):
        if o in self._cache:
            return self._cache[o]
        if not self.screen.harmful(o):
            self.stats["cleared"] += 1
            res = (1.0, 0.0)
        else:
            self.stats["exact"] += 1
            res = self.m.evaluate(o)
            if res[1] <= self.tol:
                self.stats["cancelled"] += 1
        self._cache[o] = res
        return res
