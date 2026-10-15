"""Numerical search for witnesses with magic (Section "Limits of witnesses").

Find n-qubit pure states psi stabilized by a group of size 2^(n-numax) whose
T-basis fault weights avoid the band [d, n-d], and with c_0 or c_{1^n} nonzero, then test whether a single Pauli projection
(the only nullity drop allowed when nu(psi)=nu(CCZ)+1) gives a state with the Pauli
spectrum of CCZ padded with |0>s. Floating point; supporting evidence, not a proof.
Usage: python scripts/53_search_magic_witnesses.py n d numax   (paper: 6 2 4 and 7 2 4)
"""
import itertools, sys
import numpy as np

n = int(sys.argv[1]); d = int(sys.argv[2]); numax = int(sys.argv[3])
D = 2 ** n
w8 = np.exp(1j * np.pi / 4)
T = np.array([1, w8]) / np.sqrt(2)
Z = np.array([1, -1])

def phi(e):
    v = np.array([1.0 + 0j])
    for j in range(n):
        v = np.kron(v, T * (Z if (e >> (n - 1 - j)) & 1 else 1))
    return v

wt = lambda e: bin(e).count("1")
allowed = [e for e in range(D) if wt(e) < d or wt(e) > n - d]
B = np.array([phi(e) for e in allowed]).T  # D x |A|
phi0, phi1 = phi(0), phi(D - 1)

idx = np.arange(D)
popc = np.array([wt(i) for i in range(D)])

def apply_pauli(x, z, V):
    # P = i^{|x&z|} X^x Z^z, Hermitian; acts on rows of V (basis index)
    ph = (-1.0) ** popc[idx & z]  # Z^z |b> = (-1)^{b.z}|b>
    W = V * ph[:, None]
    W = W[idx ^ x]  # X^x |b> = |b^x>: (X W)[b] = W[b^x]
    return (1j ** wt(x & z)) * W

def commute(p, q):
    return (wt(p[0] & q[1]) + wt(p[1] & q[0])) % 2 == 0

def restrict(Q, x, z, s):
    M = Q.conj().T @ (s * apply_pauli(x, z, Q))
    vals, vecs = np.linalg.eigh((M + M.conj().T) / 2)
    keep = vals > 1 - 1e-9
    return Q @ vecs[:, keep]

def alive(Q):
    if Q.shape[1] == 0:
        return False
    return np.linalg.norm(Q.conj().T @ phi0) ** 2 + np.linalg.norm(Q.conj().T @ phi1) ** 2 > 1e-9

def gf2_rank(vecs):
    rows = list(vecs); r = 0
    for bit in reversed(range(2 * n)):
        piv = next((i for i in range(r, len(rows)) if rows[i] >> bit & 1), None)
        if piv is None: continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and rows[i] >> bit & 1: rows[i] ^= rows[r]
        r += 1
    return r

paulis = [(x, z) for x in range(D) for z in range(D) if (x, z) != (0, 0)]
need = n - numax
Q0, _ = np.linalg.qr(B)
results = []
seen = set()

def canon(gens, Q):
    # canonical key: the stabilizer subspace projector rounded
    P = Q @ Q.conj().T
    return np.round(P, 6).tobytes()

def group_elems(gens):
    els = [(0, 0)]
    for g in gens:
        els = els + [(e[0] ^ g[0], e[1] ^ g[1]) for e in els]
    return els[1:]

def search(gens, Q, start):
    if len(gens) == need:
        key = canon(gens, Q)
        if key not in seen:
            seen.add(key); results.append((list(gens), Q))
        return
    for k in range(start, len(paulis)):
        x, z = paulis[k]
        if not all(commute((x, z), g[:2]) for g in gens):
            continue
        if gf2_rank([(g[0] << n) | g[1] for g in gens] + [(x << n) | z]) <= len(gens):
            continue
        # canonical generating sequence: new generator is the least-index element of its coset
        k_idx = x * D + z
        ok = True
        for h in group_elems(gens):
            if (x ^ h[0]) * D + (z ^ h[1]) < k_idx: ok = False; break
        if not ok: continue
        for s in (1, -1):
            Q2 = restrict(Q, x, z, s)
            if alive(Q2):
                search(gens + [(x, z, s)], Q2, k + 1)

search([], Q0, 0)

H = (-1.0) ** popc[idx[:, None] & idx[None, :]]
XB = idx[:, None] ^ idx[None, :]
def spec(v):
    return np.abs((np.conj(v[XB]) * v[None, :]) @ H)  # [x, z]
c = np.array([(-1) ** (((i >> 2) & (i >> 1) & i) & 1) for i in range(8)]) / np.sqrt(8)
ref = c.astype(complex)
for _ in range(n - 3): ref = np.kron(ref, [1, 0])
REF = np.sort(np.round(spec(ref).ravel(), 6))
def is_ccz(u):
    s = spec(u).ravel()
    if np.any(np.minimum(np.minimum(abs(s), abs(s - .5)), abs(s - 1)) > 1e-6): return False
    return np.allclose(np.sort(np.round(s, 6)), REF)
rng = np.random.default_rng(7)
hits = tested = 0; dims = {}; nus = {}
for gens, Q in results:
    dims[Q.shape[1]] = dims.get(Q.shape[1], 0) + 1
    cvec = rng.normal(size=Q.shape[1]) + 1j * rng.normal(size=Q.shape[1])
    v = Q @ cvec; v /= np.linalg.norm(v)
    if abs(np.vdot(phi0, v)) ** 2 + abs(np.vdot(phi1, v)) ** 2 < 1e-9: continue
    S = spec(v); nu = int(round(n - np.log2((S > 1 - 1e-7).sum())))
    nus[nu] = nus.get(nu, 0) + 1
    tested += 1
    if nu < 3: print("nullity<3 witness", gens); continue
    if nu == 3: conv = is_ccz(v)
    else:
        conv = False
        sx, sz = np.nonzero(S > 1 - 1e-7)
        for x in range(D):
            for z in range(D):
                if S[x, z] > 1 - 1e-7: continue
                if not all(commute((x, z), (a, b)) for a, b in zip(sx, sz)): continue
                Pv = apply_pauli(x, z, v[:, None])[:, 0]
                for sgn in (1, -1):
                    u = v + sgn * Pv; nm = np.linalg.norm(u)
                    if nm < 1e-9: continue
                    if is_ccz(u / nm): conv = True; break
                if conv: break
            if conv: break
    hits += conv
print(f"n={n} d={d} numax={numax}: subspaces by dim {dims}; witness nullities {nus}; tested {tested}; CCZ-convertible {hits}; NOT convertible {tested-hits}")
