"""Validate the relaxation: twirled Choi vectors of real protocols must satisfy every constraint.

For a code C with r = 1 (2-dim code space) encoding |T>, lam[v,b] is the average over patterns e of
weight v of |<u_b|Pi_C phi_e>|^2, where u_0 is the normalized Pi_C phi_0 and u_1 spans the rest
of the code space; this is <T|Lambda(rho_e)|T> and <ZT|Lambda(rho_e)|ZT> after Clifford decoding.
"""
import itertools
import numpy as np

from t_cut import facets, contraction_matrix
from t_lp import conditioning_set

OMEGA = np.exp(1j * np.pi / 4)
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1, -1])}


def op(s):
    sign = 1
    if s[0] in "+-":
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    M = np.array([[1]])
    for c in s:
        M = np.kron(M, PAULI[c])
    return sign * M


def lam_of_code(stabs):
    n = len(stabs[0].lstrip("+-"))
    P = np.eye(2 ** n, dtype=complex)
    for g in stabs:
        P = P @ (np.eye(2 ** n) + op(g)) / 2
    T = np.array([1, OMEGA]) / np.sqrt(2)
    ZT = np.array([1, -OMEGA]) / np.sqrt(2)
    lam = np.zeros((n + 1, 2))
    cnt = np.zeros(n + 1)
    # code space basis
    w, V = np.linalg.eigh(P)
    Cb = V[:, w > 0.5]
    assert Cb.shape[1] == 2
    phi0 = T
    for _ in range(n - 1):
        phi0 = np.kron(phi0, T)
    u0 = P @ phi0
    u0 /= np.linalg.norm(u0)
    u1 = Cb @ (Cb.conj().T @ np.random.default_rng(1).normal(size=2 ** n))
    u1 = u1 - u0 * (u0.conj() @ u1)
    u1 /= np.linalg.norm(u1)
    for e in itertools.product((0, 1), repeat=n):
        phi = np.array([1.0 + 0j])
        for b in e:
            phi = np.kron(phi, ZT if b else T)
        pe = P @ phi
        lam[sum(e), 0] += abs(u0.conj() @ pe) ** 2
        lam[sum(e), 1] += abs(u1.conj() @ pe) ** 2
        cnt[sum(e)] += 1
    lam /= cnt[:, None]
    return lam / lam[0, 0]


def check(lam, levels, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(lam) - 1
    worst = 0.0
    for k in levels:
        if k > n:
            continue
        F = facets(k)
        ms = conditioning_set(n - k, rng, 1000) if n > k else np.array([[1.0]])
        for mv in ms:
            A = contraction_matrix(n, k, mv)
            worst = min(worst, (F @ (A @ lam.reshape(-1))).min())
    return worst


if __name__ == "__main__":
    t9 = ["-XZIIZIIIX", "+XIZIZIIXI", "+ZXZZIIXZI", "+ZZXIZXIIZ",
          "-YXYIXIIZZ", "-YYXXIIIII", "+IZIZZIZII", "+IIZZZZIII"]
    lam = lam_of_code(t9)
    np.set_printoptions(precision=5, suppress=True)
    print("nine-input factory lam[v,b] (v = 0..9):")
    print(lam)
    print("harmless weights v<2 have lam[v,1] = 0:", np.allclose(lam[:2, 1], 0))
    print("most negative facet value (should be >= 0):", check(lam, (1, 2, 3)))
    triv = np.array([[1.0, 0.0], [0.0, 1.0]])
    print("one-input identity protocol, most negative facet value:", check(triv, (1,)))
