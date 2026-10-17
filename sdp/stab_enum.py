"""Exact enumeration of stabilizer states and their distributions over T-basis fault patterns.

Every N-qubit stabilizer state has a unique form (Dehaene-De Moor)
    |s> = 2^{-t/2} sum_{u in F2^t} i^{l.u} (-1)^{q(u)} |G u + c>,
with G an N x t matrix in reduced column echelon form (a basis of a t-dim subspace),
c a coset representative (zero on the pivot rows), l in F2^t and q a quadratic form
with zero diagonal. We enumerate all of them and return, for each state, the vector
    d_e(s) = |<phi_e|s>|^2,  e in F2^N,  phi_e = Z^e |T>^{N},
which is the distribution over fault patterns obtained by dephasing s in the T basis.
"""
import itertools
import numpy as np

OMEGA = np.exp(1j * np.pi / 4)


def subspaces(N, t):
    """All t-dim subspaces of F2^N as (basis rows u_1..u_t in RREF, pivot list)."""
    out = []
    for pivots in itertools.combinations(range(N), t):
        free = []  # positions that may be set: to the right of the pivot, not a pivot
        for i, p in enumerate(pivots):
            free.append([j for j in range(p + 1, N) if j not in pivots])
        nfree = sum(len(f) for f in free)
        for bits in itertools.product((0, 1), repeat=nfree):
            B = np.zeros((t, N), dtype=np.uint8)
            it = iter(bits)
            for i, p in enumerate(pivots):
                B[i, p] = 1
                for j in free[i]:
                    B[i, j] = next(it)
            out.append((B, pivots))
    return out


def bitrows(k):
    """All vectors of F2^k as rows of a (2^k, k) int array (k = 0 gives one empty row)."""
    return np.array(list(itertools.product((0, 1), repeat=k)), dtype=np.int64).reshape(2 ** k, k)


def popcount_rows(X):
    return X.sum(axis=1)


def enumerate_distributions(N, dedupe_decimals=10):
    """Return array D of shape (#distinct, 2^N) of fault-pattern distributions."""
    E = np.array(list(itertools.product((0, 1), repeat=N)), dtype=np.int64)  # e vectors, row = e
    found = {}
    total = 0
    for t in range(N + 1):
        U = bitrows(t)
        # phase table for all (l, q): exponent of i is l.u + 2*q(u) (mod 4)
        pairs = [(a, b) for a in range(t) for b in range(a + 1, t)]
        L = bitrows(t)
        Q = bitrows(len(pairs))
        # i^{l.u} (-1)^{h.u}: the four phases 1, i, -1, -i on each basis direction
        lu = ((L @ U.T)[:, None, :] + 2 * (L @ U.T)[None, :, :]).reshape(-1, len(U))  # (#l*#h, 2^t)
        if pairs:
            uu = np.stack([U[:, a] * U[:, b] for a, b in pairs], axis=1)  # (2^t, #pairs)
            qu = Q @ uu.T  # (#q, 2^t)
        else:
            qu = np.zeros((1, len(U)), dtype=np.int64)
        expo = (lu[:, None, :] + 2 * qu[None, :, :]) % 4  # (#l, #q, 2^t)
        phases = (1j ** expo).reshape(-1, len(U))  # (#l*#q, 2^t)
        for B, pivots in subspaces(N, t):
            nonpiv = [j for j in range(N) if j not in pivots]
            for cbits in itertools.product((0, 1), repeat=len(nonpiv)):
                c = np.zeros(N, dtype=np.int64)
                c[nonpiv] = cbits
                X = (U @ B.astype(np.int64) + c) % 2  # support points, (2^t, N)
                # <phi_e|x> = 2^{-N/2} conj(omega)^{|x|} (-1)^{e.x}
                M = (np.conj(OMEGA) ** popcount_rows(X))[:, None] * (-1.0) ** ((X @ E.T) % 2)  # (2^t, 2^N)
                amp = phases @ M * (2.0 ** (-(N + t) / 2))
                D = np.abs(amp) ** 2
                total += len(D)
                for row in np.round(D, dedupe_decimals):
                    found.setdefault(row.tobytes(), row)
    D = np.array(list(found.values()))
    return D, total


def count_states(N):
    def gauss(n, k):
        num = den = 1
        for i in range(k):
            num *= 2 ** (n - i) - 1
            den *= 2 ** (i + 1) - 1
        return num // den
    return sum(gauss(N, t) * 2 ** (N - t) * 2 ** (t * (t + 3) // 2) for t in range(N + 1))


if __name__ == "__main__":
    import sys, time
    N = int(sys.argv[1])
    t0 = time.time()
    D, total = enumerate_distributions(N)
    print(f"N={N}: enumerated {total} states (expected {count_states(N)}), "
          f"{len(D)} distinct distributions, {time.time()-t0:.1f}s")
    assert total == count_states(N)
    assert np.allclose(D.sum(axis=1), 1.0)
    np.save(f"dist_N{N}.npy", D)
