"""Independent checks of Theorem 11 and Proposition 12 of cultivation.tex.

1. Theorem 11: for the skeleton codes of single-qubit cultivation circuits
   (inject T, then k checks T^dag . proj X . T, so t = 1 + 2k inputs), compare
   Pi_C E |T>^t with mu 2^{-|a|/2} (g_0 + g_1 Z^l) Pi_C |T>^t for every Pauli E,
   and the acceptance and bad weight with their formulas.
2. Proposition 12: evaluate sum_z i^{q(z)} by eliminating one variable at a
   time (fresh implementation) and compare with brute force on random forms.
Prints ALL OK.
"""
import itertools
import numpy as np

rng = np.random.default_rng(7)
w = np.exp(1j * np.pi / 4)
Tst = np.array([1, w]) / np.sqrt(2)


# ---------------------------------------------------------------- 1. Theorem 11
def skeleton_map(k):
    """Linear map M from t = 1+2k ancilla inputs to one output qubit."""
    t = 1 + 2 * k
    M = np.zeros((2, 2**t), complex)
    for x in range(2**t):
        bits = [(x >> (t - 1 - j)) & 1 for j in range(t)]
        # data qubit state as a 2-vector; each gadget CX(q->a) then <0|_a
        # applied to |psi>|bit> keeps |q> components with q XOR bit = 0.
        psi = np.array([1, 1], complex) / np.sqrt(2)          # |+>
        j = 0

        def gadget(psi, b):
            out = np.zeros(2, complex)
            out[b] = psi[b] * np.sqrt(2)                       # sqrt2 <0|_a CX |psi>|b>
            return out
        psi = gadget(psi, bits[j]); j += 1                     # injection T
        for _ in range(k):
            psi = gadget(psi, bits[j]); j += 1                 # T of T^dag = S^dag T
            psi = np.diag([1, -1j]) @ psi                      # S^dag
            psi = (np.eye(2) + np.array([[0, 1], [1, 0]])) / 2 @ psi   # project X = +1
            psi = gadget(psi, bits[j]); j += 1                 # T
        M[:, x] = psi
    return M


def kronlist(vs):
    out = np.array([1.0 + 0j])
    for v in vs:
        out = np.kron(out, v)
    return out


def xpat(t, a):
    X = np.array([[0, 1], [1, 0]], complex)
    out = np.array([[1.0 + 0j]])
    for j in range(t):
        out = np.kron(out, X if a[j] else np.eye(2, dtype=complex))
    return out


def zpat(t, e):
    return np.diag([(-1) ** sum(((x >> (t - 1 - j)) & 1) * e[j] for j in range(t)) for x in range(2**t)]).astype(complex)


ok = True
for k in (1, 2, 3):
    t = 1 + 2 * k
    M = skeleton_map(k)
    Tt = kronlist([Tst] * t)
    out0 = M @ Tt
    assert np.allclose(out0 / out0[0], Tst / Tst[0])                      # correct circuit
    G = M.conj().T @ M
    Pi = G / np.trace(G).real * 2                                          # rank-2 projector (r = 1)
    assert np.allclose(Pi @ Pi, Pi)
    v0 = Pi @ Tt
    # classify Z patterns
    pats = list(itertools.product([0, 1], repeat=t))
    vec = {e: Pi @ zpat(t, e) @ Tt for e in pats}
    Sigma = [e for e in pats if np.allclose(vec[e], v0) or np.allclose(vec[e], -v0)]
    xi = {e: (1 if np.allclose(vec[e], v0) else -1) for e in Sigma}
    nonzero = [e for e in pats if np.linalg.norm(vec[e]) > 1e-9]
    rest = [e for e in nonzero if e not in Sigma]
    l = rest[0]
    Zl = zpat(t, l)
    assert np.allclose(Zl @ Pi, Pi @ Zl)                                   # Z^l commutes with S_C
    assert set(rest) == {tuple((np.array(s) + np.array(l)) % 2) for s in Sigma}   # Definition 10
    for a in pats:
        for b in pats:
            g = [0j, 0j]
            for wv in itertools.product([0, 1], repeat=t):
                if any(wv[j] and not a[j] for j in range(t)):
                    continue
                e = tuple((np.array(wv) + np.array(b)) % 2)
                for c in (0, 1):
                    s = tuple((np.array(e) + c * np.array(l)) % 2)
                    if s in xi:
                        g[c] += 1j ** sum(wv) * xi[s]
            mu = (-1) ** sum(x & y for x, y in zip(a, b))
            E = xpat(t, a) @ zpat(t, b)                                    # E = X^a Z^b
            lhs = Pi @ E @ Tt
            rhs = mu * 2 ** (-sum(a) / 2) * (g[0] * v0 + g[1] * Zl @ v0)
            ok &= np.allclose(lhs, rhs)
            acc = np.vdot(lhs, lhs).real / np.vdot(v0, v0).real
            ok &= np.isclose(acc, 2 ** -sum(a) * (abs(g[0])**2 + abs(g[1])**2))
            outv = M @ E @ Tt
            if np.linalg.norm(outv) > 1e-9:
                bad = 1 - abs(np.vdot(Tst, outv))**2 / np.vdot(outv, outv).real
                bad_w = bad * np.vdot(outv, outv).real / np.vdot(out0, out0).real
                ok &= np.isclose(bad_w, 2 ** -sum(a) * abs(g[1])**2)
    print(f"Theorem 11, k={k} (t={t}): {len(Sigma)} stabilizer patterns, all {4**t} Paulis checked")


# ---------------------------------------------------------------- 2. Proposition 12
def gauss_brute(q0, L, B):
    m = len(L)
    s = 0j
    for z in itertools.product([0, 1], repeat=m):
        q = q0 + sum(L[i] * z[i] for i in range(m)) + 2 * sum(B[i][j] * z[i] * z[j] for i in range(m) for j in range(i + 1, m))
        s += 1j ** (q % 4)
    return s


def gauss_elim(q0, L, B):
    """sum_z i^{q0 + sum L_i z_i + 2 sum_{i<j} B_ij z_i z_j}, eliminating variables."""
    L = [x % 4 for x in L]
    B = [[B[i][j] % 2 if i != j else 0 for j in range(len(L))] for i in range(len(L))]
    for i in range(len(L)):
        for j in range(i):
            B[i][j] = B[j][i]
    alive = list(range(len(L)))
    factor = 1 + 0j
    q0 %= 4
    while alive:
        v = alive[0]
        N = [u for u in alive if u != v and B[v][u]]
        alive.remove(v)
        if L[v] % 2:
            # 1 + i^{L_v} (-1)^y = (1 + i^{L_v}) * i^{c y}, with c = 3 if L_v = 1, c = 1 if L_v = 3
            factor *= 1 + 1j ** L[v]
            c = 3 if L[v] == 1 else 1
            # add c * (sum_N z - 2 sum_{pairs in N} z z) to q
            for u in N:
                L[u] = (L[u] + c) % 4
            for x, y in itertools.combinations(N, 2):
                B[x][y] ^= 1; B[y][x] ^= 1          # -2c mod 4 = 2 (c odd)
        else:
            target = (L[v] // 2) % 2                 # need y = target
            if not N:
                if target:
                    return 0j
                factor *= 2
                continue
            factor *= 2
            u = N[0]
            others = [x for x in N if x != u]
            # z_u = target XOR (XOR of others): substitute
            # as integer mod 4: z_u = t' where t' = target + sum others - 2 sum pairs(target & others ...)
            # do it by expanding: define parity p = target + S (mod 2), with S = XOR others
            alive.remove(u)
            Lu, Bu = L[u], [B[u][x] for x in range(len(L))]
            # contribution of z_u: Lu z_u + 2 sum_{x alive} Bu[x] z_u z_x
            # z_u (as 0/1 integer) = parity of (target, others)
            # integer value mod 4 of parity of bits b_1..b_n: sum b - 2 sum_{pairs} b b
            bits = others[:]
            const = target
            # Lu * parity: linear part Lu*(const + sum bits) - 2 Lu*(const*sum bits + sum pairs)
            q0 = (q0 + Lu * const) % 4
            for x in bits:
                L[x] = (L[x] + Lu - 2 * Lu * const) % 4
            for x, y in itertools.combinations(bits, 2):
                if Lu % 2:
                    B[x][y] ^= 1; B[y][x] ^= 1
            # 2 Bu[x] z_u z_x: 2 * parity * z_x = 2 z_x (const + sum bits) mod 4
            for x in alive:
                if Bu[x]:
                    if const:
                        L[x] = (L[x] + 2) % 4
                    for y in bits:
                        if y == x:
                            L[x] = (L[x] + 2) % 4    # z_x^2 = z_x
                        else:
                            B[x][y] ^= 1; B[y][x] ^= 1
    return factor * 1j ** q0


for trial in range(2000):
    m = int(rng.integers(1, 9))
    q0 = int(rng.integers(0, 4))
    L = [int(x) for x in rng.integers(0, 4, m)]
    B = [[int(x) for x in row] for row in rng.integers(0, 2, (m, m))]
    for i in range(m):
        for j in range(i):
            B[i][j] = B[j][i]
    a, b = gauss_brute(q0, L, B), gauss_elim(q0, L, B)
    ok &= np.isclose(a, b)
    if abs(a) > 1e-9:
        ok &= np.isclose(np.log2(abs(a)) * 2, round(np.log2(abs(a)) * 2))  # power of sqrt2
print("Proposition 12: 2000 random forms with up to 8 variables")
print("ALL OK" if ok else "FAILED")
