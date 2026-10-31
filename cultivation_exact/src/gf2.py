"""Small GF(2) linear algebra on numpy uint8 matrices (entries 0/1)."""
import numpy as np


def rref(A):
    """Reduced row echelon form. Returns (R, pivots) with R[:len(pivots)] the nonzero rows."""
    R = (np.asarray(A, dtype=np.uint8) & 1).copy()
    m, n = R.shape
    piv = []
    r = 0
    for c in range(n):
        if r == m:
            break
        rows = np.nonzero(R[r:, c])[0]
        if len(rows) == 0:
            continue
        p = r + rows[0]
        if p != r:
            R[[r, p]] = R[[p, r]]
        hit = np.nonzero(R[:, c])[0]
        hit = hit[hit != r]
        if len(hit):
            R[hit] ^= R[r]
        piv.append(c)
        r += 1
    return R, piv


def rank(A):
    return len(rref(A)[1])


def nullspace(A):
    """Basis (rows) of {x : A x = 0}."""
    A = np.asarray(A, dtype=np.uint8)
    n = A.shape[1]
    R, piv = rref(A)
    free = [c for c in range(n) if c not in set(piv)]
    B = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        B[i, f] = 1
        for r, p in enumerate(piv):
            B[i, p] = R[r, f]
    return B


def solve(A, b):
    """One solution x of A x = b, or None."""
    A = np.asarray(A, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8).reshape(-1, 1)
    R, piv = rref(np.hstack([A, b]))
    n = A.shape[1]
    if n in piv:
        return None
    x = np.zeros(n, dtype=np.uint8)
    for r, p in enumerate(piv):
        x[p] = R[r, n]
    return x


def right_inverse(A):
    """Matrix X (n x m) with A X = I_m, for A (m x n) of full row rank."""
    m = A.shape[0]
    cols = []
    for i in range(m):
        e = np.zeros(m, dtype=np.uint8)
        e[i] = 1
        x = solve(A, e)
        if x is None:
            raise ValueError("matrix does not have full row rank")
        cols.append(x)
    return np.array(cols, dtype=np.uint8).T
