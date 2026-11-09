"""Shared conventions for the Choi-program searches.

Qubit j <-> bit j of an integer (LSB = qubit 0).  A Pauli is (p, x, z) meaning i^p X^x Z^z
(X part applied after Z part in operator order X^x Z^z).  Hermitian canonical phase p = |x&z|.
Input fault states phi_e = Z^e |T>^n, |T> = (|0> + w|1>)/sqrt2, w = e^{i pi/4}.
"""
import numpy as np
from itertools import combinations

def popcount(v):
    return bin(v).count("1")

POP = np.array([popcount(i) for i in range(1 << 16)], dtype=np.int64)

def popc_arr(a):
    a = np.asarray(a, dtype=np.int64)
    return POP[a & 0xFFFF] + POP[(a >> 16) & 0xFFFF]

def pmul(P, Q):
    """(i^p X^x Z^z)(i^q X^x' Z^z') = i^{p+q+2|z&x'|} X^{x^x'} Z^{z^z'}"""
    p, x, z = P; q, x2, z2 = Q
    return ((p + q + 2 * popcount(z & x2)) % 4, x ^ x2, z ^ z2)

def herm(x, z):
    return (popcount(x & z) % 4, x, z)

# ---------- F2 linear algebra on int bitvectors ----------
def rref(rows):
    """fully reduced row echelon form (pivot = highest bit), as a descending tuple"""
    basis = []
    for r in rows:
        for b in basis:
            if (r >> (b.bit_length() - 1)) & 1: r ^= b
        if r:
            hb = r.bit_length() - 1
            basis = [b ^ r if (b >> hb) & 1 else b for b in basis]
            basis.append(r)
    return tuple(sorted(basis, reverse=True))

def span(rows):
    s = {0}
    for r in rows:
        s |= {v ^ r for v in s}
    return s

def in_span_rref(v, rr):
    for r in rr:
        hb = r.bit_length() - 1
        if (v >> hb) & 1: v ^= r
    return v == 0

def perp_basis(rows, n):
    """basis of {v in F2^n : v.r = 0 for r in rows}"""
    sols = []
    # solve via brute force small n
    rr = list(rref(rows))
    piv = [r.bit_length() - 1 for r in rr]
    free = [j for j in range(n) if j not in piv]
    for f in free:
        v = 1 << f
        # choose pivot bits so that each row's dot product is 0
        for r, p in zip(rr, piv):
            if popcount(r & v) % 2: v ^= 1 << p
        sols.append(v)
    # verify
    for v in sols:
        for r in rows: assert popcount(r & v) % 2 == 0
    return sols

def nullspace(constraints, nvars):
    """constraints: list of int masks over nvars bits; returns basis of solutions x with popcount(c&x) even."""
    rr = []
    for c in constraints:
        for r in rr:
            hb = r.bit_length() - 1
            if (c >> hb) & 1: c ^= r
        if c:
            hb = c.bit_length() - 1
            rr = [r ^ c if (r >> hb) & 1 else r for r in rr]
            rr.append(c)
    piv = {r.bit_length() - 1: r for r in rr}
    basis = []
    for f in range(nvars):
        if f in piv: continue
        v = 1 << f
        for p, r in piv.items():
            if popcount(r & v) % 2: v ^= 1 << p
        basis.append(v)
    return basis

def subspaces(n, k):
    """all k-dim subspaces of F2^n as canonical rref tuples"""
    out = set()
    # direct: enumerate rref matrices via combinations of pivot positions
    for pivs in combinations(range(n), k):
        pivs = sorted(pivs, reverse=True)
        # row i has pivot pivs[i]; free bits: positions below pivot that are not other pivots
        frees = [[b for b in range(p) if b not in pivs] for p in pivs]
        tot = sum(len(f) for f in frees)
        for m in range(1 << tot):
            rows = []; t = 0
            for p, fr in zip(pivs, frees):
                r = 1 << p
                for b in fr:
                    if (m >> t) & 1: r |= 1 << b
                    t += 1
                rows.append(r)
            out.add(tuple(sorted(rows, reverse=True)))
    return out

def permute_bits(v, perm):
    r = 0
    for j, pj in enumerate(perm):
        if (v >> j) & 1: r |= 1 << pj
    return r

def subspace_orbits(n, k):
    """orbit representatives of k-dim subspaces of F2^n under S_n (adjacent transpositions + n-cycle)."""
    allsp = subspaces(n, k)
    gens = [tuple(list(range(i)) + [i + 1, i] + list(range(i + 2, n))) for i in range(n - 1)]
    seen = set(); reps = []
    for s in sorted(allsp):
        if s in seen: continue
        reps.append(s); stack = [s]; seen.add(s)
        while stack:
            cur = stack.pop()
            for g in gens:
                t = rref([permute_bits(r, g) for r in cur])
                if t not in seen:
                    seen.add(t); stack.append(t)
    assert len(seen) == len(allsp)
    return reps

# ---------- dense numerics for verification ----------
W8 = np.exp(1j * np.pi / 4)
def phi(e, n):
    v = np.array([1.0 + 0j])
    for j in reversed(range(n)):   # kron order: qubit n-1 is most significant
        v = np.kron(v, np.array([1, -W8 if (e >> j) & 1 else W8]) / np.sqrt(2))
    return v

def pauli_dense(P, n):
    p, x, z = P
    D = 1 << n; idx = np.arange(D)
    ph = (-1.0) ** popc_arr(idx & z)
    M = np.zeros((D, D), complex)
    M[idx ^ x, idx] = ph   # Z^z then X^x : |b> -> (-1)^{b.z} |b^x>
    return (1j ** p) * M

def code_projector(gens, n):
    D = 1 << n
    Pm = np.eye(D, dtype=complex)
    for (s, P) in gens:
        Pm = Pm @ (np.eye(D) + s * pauli_dense(P, n)) / 2
    return Pm

def harmless_set(n, d):
    return [e for e in range(1 << n) if popcount(e) < d or popcount(e) > n - d]
