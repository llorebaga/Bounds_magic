"""Numerical checks of the generalized statements of paper/cultivation.tex (state vectors; stim only draws random Clifford tableaux).

A. Theorem 1 for any non-Clifford gates (Proposition 10). Random circuits with Clifford gates, Pauli projections,
   ideal removals and non-Clifford gates of four kinds: T, Z rotations by random angles, CS and CCZ (diagonal,
   injected by CX(q -> a), <0|_a from D|+>^k), and random single-qubit unitaries (teleported through their Choi
   state). For every single fault and random pairs we check: (i) the gadget identity, K_F|0> = M_F |chi>;
   (ii) M = c V Pi_C (M^dag M is a multiple of a projector); (iii) M_F = 0 or M_F = lambda R M E with Paulis R on
   the output and E on the inputs, where R can be taken to be the identity unless it leaves the range of M;
   (iv) harmless (computed directly) iff M_F = 0, or R = I and Pi_C E|chi> is parallel to Pi_C|chi>, or R != I
   and Pi_C E|chi> = 0; (v) a Z fault after a diagonal gate gives E = Z on the matching input qubit.
B. Gauss sums for any diagonal inputs and any checks (Theorem 15). Random stabilizer codes C on blocks of inputs
   D|+>^k (T, T^dag, S, CS, CCZ: third level; Z rotations by random angles and sqrt(T): fourth level and beyond).
   For every Pauli X^a Z^b: Pi_C X^a Z^b |chi> = (-1)^{a.b} sum_beta g_beta v_beta over the accepted cosets beta
   of Sigma = {e : Z^e in +-S_C}, with g_beta = sum_{z : z + b in beta} c_a(z) xi(z + b + e_beta) and c_a the Z
   Fourier coefficients of D^dag X^a D X^a. For the third level, each g_beta is also evaluated as a Z4 Gauss sum
   over (x, u) by variable elimination, and |g_beta| is zero or 2^(-K) times a power of sqrt(2).
C. Coherent completion (Lemma 4) on CSS codes with any angles: on the [[7,1,3]] and [[19,1,5]] colour codes and
   the [[9,1,3]] rotated surface code, Pi X^(l\\u) V X^u V^dag |tau_L> equals (sum_{w in W_u} eps_w c_w) X^l
   |tau_L>, with c_w = prod_{u\\w} cos(theta) prod_w (-i sin(theta)), W_u the Z stabilizers inside u.
D. Transversal checks with strongly fault-tolerant gadgets (Theorem 6): on the [[7,1,3]] and [[19,1,5]] codes,
   Paulis E_0, ..., E_4 interleaved with U^dag, (I + s X^n)/2, U and a stabilizer projection, with fewer than
   delta qubits in the union of their supports, never give a wrong output; a coherent completion with delta
   qubits does.
Prints ALL OK.  Usage: python cultivation_general_check.py [seed]
"""
import sys, itertools
import numpy as np
import stim

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
rng = np.random.default_rng(seed)
TOL = 1e-9
W8 = np.exp(1j * np.pi / 4)
P1 = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], complex),
      "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]).astype(complex)}
C1 = {"H": np.array([[1, 1], [1, -1]]) / np.sqrt(2), "S": np.diag([1, 1j]), "Sdg": np.diag([1, -1j])}
ALL_OK = True


def report(name, ok):
    global ALL_OK
    ALL_OK &= bool(ok)
    print(f"{name}: {'OK' if ok else 'FAILED'}", flush=True)


def apply1(psi, U, q):
    return np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)


def apply_cx(psi, c, t):
    psi = psi.copy()
    idx = [slice(None)] * psi.ndim
    idx[c] = 1
    sub = psi[tuple(idx)]
    psi[tuple(idx)] = np.flip(sub, axis=t - 1 if t > c else t).copy()
    return psi


def apply_diag(psi, dvec, qs):
    k = len(qs)
    D = dvec.reshape([2] * k)
    shape = [1] * psi.ndim
    for j, q in enumerate(qs):
        shape[q] = 2
    order = np.argsort(qs)                         # broadcast D along the axes qs (in increasing order)
    Dt = np.transpose(D, order).reshape([2 if i in qs else 1 for i in range(psi.ndim)])
    return psi * Dt


def apply_pauli(psi, ps, qs):
    for p, q in zip(ps, qs):
        if p != "I":
            psi = apply1(psi, P1[p], q)
    return psi


def remove0(psi, q):
    psi = psi.copy()
    idx = [slice(None)] * psi.ndim
    idx[q] = 1
    psi[tuple(idx)] = 0
    return psi


def paulis(n):
    return ["".join(p) for p in itertools.product("IXYZ", repeat=n)]


def pauli_mat(s):
    out = np.array([[1.0 + 0j]])
    for ch in s:
        out = np.kron(out, P1[ch])
    return out


# ============================================================== A. Theorem 1 for any non-Clifford gates
DIAG = {"T": lambda: np.array([1, W8]), "Rz": lambda: np.array([1, np.exp(1j * rng.uniform(0, 2 * np.pi))]),
        "CS": lambda: np.array([1, 1, 1, 1j]), "CCZ": lambda: np.array([1, 1, 1, 1, 1, 1, 1, -1], complex)}


def random_circuit(nq):
    """ops: ('g1', name, q), ('cx', c, t), ('diag', dvec, qs), ('u1', U, q), ('proj', ps, qs, sign), ('rm', q)"""
    ops, n_in = [], 0
    for q in range(nq):
        ops.append(("g1", "H", q))
    kinds = []
    while n_in < 3:
        kd = rng.choice(["T", "Rz", "CS", "CCZ", "u1"] if nq >= 3 else ["T", "Rz", "CS", "u1"])
        k = {"T": 1, "Rz": 1, "CS": 2, "CCZ": 3, "u1": 2}[kd]
        if n_in + k > 5:
            break
        kinds.append(kd)
        n_in += k
    for kd in kinds:
        for _ in range(int(rng.integers(1, 3))):
            if rng.random() < 0.5:
                ops.append(("g1", str(rng.choice(["H", "S", "Sdg"])), int(rng.integers(nq))))
            else:
                c, t = rng.choice(nq, 2, replace=False)
                ops.append(("cx", int(c), int(t)))
        if kd == "u1":
            A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            U, _ = np.linalg.qr(A)
            ops.append(("u1", U, int(rng.integers(nq))))
        else:
            k = {"T": 1, "Rz": 1, "CS": 2, "CCZ": 3}[kd]
            qs = [int(x) for x in rng.choice(nq, k, replace=False)]
            ops.append(("diag", DIAG[kd](), qs))
        if rng.random() < 0.6:
            qs = [int(x) for x in rng.choice(nq, int(rng.integers(1, nq + 1)), replace=False)]
            ps = "".join(rng.choice(list("XYZ"), len(qs)))
            ops.append(("proj", ps, qs, 1))
    if rng.random() < 0.5:
        ops.append(("rm", int(rng.integers(nq))))
    return ops


def n_inputs(ops):
    return sum(len(op[2]) if op[0] == "diag" else 2 for op in ops if op[0] in ("diag", "u1"))


def support(op):
    if op[0] in ("g1", "rm"):
        return [op[2]] if op[0] == "g1" else [op[1]]
    if op[0] == "cx":
        return [op[1], op[2]]
    if op[0] == "diag":
        return list(op[2])
    if op[0] == "u1":
        return [op[2]]
    return list(op[2])


def run(ops, nq, F, mode, anc=None):
    """mode 'direct': the circuit itself. mode 'skel': every non-Clifford gate replaced by its gadget, with the
    resource register (qubits nq ...) in the basis state anc, or in the resource state if anc is None"""
    nin = n_inputs(ops)
    ntot = nq + (nin if mode == "skel" else 0)
    psi = np.zeros([2] * ntot, complex)
    if mode == "skel" and anc is not None:
        psi[tuple([0] * nq + list(anc))] = 1
    else:
        psi[tuple([0] * ntot)] = 1
    if mode == "skel" and anc is None:                     # prepare the resource states
        a = nq
        for op in ops:
            if op[0] == "diag":
                for j in range(len(op[2])):
                    psi = apply1(psi, C1["H"], a + j)
                psi = apply_diag(psi, op[1], list(range(a, a + len(op[2]))))
                a += len(op[2])
            elif op[0] == "u1":
                psi = apply1(psi, C1["H"], a)
                psi = apply_cx(psi, a, a + 1)
                psi = apply1(psi, op[1], a + 1)
                a += 2
    a = nq
    for i, op in enumerate(ops):
        f = F.get(i)
        if op[0] == "proj":
            _, ps, qs, sign = op
            if f is not None and f != "flip":
                psi = apply_pauli(psi, f, qs)
            s = -sign if f == "flip" else sign
            psi = (psi + s * apply_pauli(psi, ps, qs)) / 2
            continue
        if op[0] == "g1":
            psi = apply1(psi, C1[op[1]], op[2])
        elif op[0] == "cx":
            psi = apply_cx(psi, op[1], op[2])
        elif op[0] == "rm":
            psi = remove0(psi, op[1])
        elif op[0] == "diag":
            qs = op[2]
            if mode == "direct":
                psi = apply_diag(psi, op[1], qs)
            else:
                for j, q in enumerate(qs):
                    psi = apply_cx(psi, q, a + j)
                    psi = remove0(psi, a + j) * np.sqrt(2)
                a += len(qs)
        elif op[0] == "u1":
            q = op[2]
            if mode == "direct":
                psi = apply1(psi, op[1], q)
            else:                                           # 2 <Phi|_{q,a} |psi>_q (I x U)|Phi>_{a,b}, then SWAP(q, b)
                psi = apply_cx(psi, q, a)
                psi = apply1(psi, C1["H"], q)
                psi = remove0(remove0(psi, q), a) * 2
                b = a + 1
                psi = apply_cx(apply_cx(apply_cx(psi, q, b), b, q), q, b)
                a += 2
        if f is not None:
            psi = apply_pauli(psi, f, support(op))
    return psi


def out_vec(psi, ops, nq):
    removed = {op[1] for op in ops if op[0] == "rm"}
    idx = tuple(0 if (q in removed or q >= nq) else slice(None) for q in range(psi.ndim))
    return psi[idx].reshape(-1)


def skel_map(ops, nq, F):
    nin = n_inputs(ops)
    cols = [out_vec(run(ops, nq, F, "skel", anc=b), ops, nq) for b in itertools.product([0, 1], repeat=nin)]
    return np.array(cols).T


def fault_list(ops):
    singles = []
    for i, op in enumerate(ops):
        if op[0] == "rm":
            continue
        k = len(support(op))
        for p in paulis(k)[1:]:
            singles.append({i: p})
        if op[0] == "proj":
            singles.append({i: "flip"})
    return singles


def check_theorem1(n_circ=30):
    ok, n_cfg, n_R, n_rej, n_harm, n_z = True, 0, 0, 0, 0, 0
    kinds_seen = set()
    for _ in range(n_circ):
        nq = int(rng.choice([2, 3]))
        ops = random_circuit(nq)
        psi0 = out_vec(run(ops, nq, {}, "direct"), ops, nq)
        if np.linalg.norm(psi0) < 1e-6:
            continue
        tau = psi0 / np.linalg.norm(psi0)
        kinds_seen |= {("u1" if op[0] == "u1" else len(op[2]) if op[0] == "diag" else None) for op in ops}
        nin = n_inputs(ops)
        a_psi = np.zeros([2] * nin, complex)
        a_psi[tuple([0] * nin)] = 1
        a = 0
        for op in ops:
            if op[0] == "diag":
                for j in range(len(op[2])):
                    a_psi = apply1(a_psi, C1["H"], a + j)
                a_psi = apply_diag(a_psi, op[1], list(range(a, a + len(op[2]))))
                a += len(op[2])
            elif op[0] == "u1":
                a_psi = apply1(a_psi, C1["H"], a)
                a_psi = apply_cx(a_psi, a, a + 1)
                a_psi = apply1(a_psi, op[1], a + 1)
                a += 2
        chi_vec = a_psi.reshape(-1)
        M = skel_map(ops, nq, {})
        G = M.conj().T @ M
        ev = np.linalg.eigvalsh(G)
        nz = ev[ev > 1e-9]
        ok &= np.allclose(nz, nz[0])                       # M^dag M = |c|^2 Pi_C
        PiC = G / nz[0]
        rngM = M @ np.linalg.pinv(M)                       # projector onto the range of M
        dimo = M.shape[0]
        nout = int(np.log2(dimo))
        Ps_in = paulis(nin)
        ME = np.array([(M @ pauli_mat(e)).reshape(-1) for e in Ps_in])
        nME = np.linalg.norm(ME, axis=1)
        Rs = [r for r in paulis(nout)]
        Rmat = {r: pauli_mat(r) for r in Rs}
        R_in = {r: np.allclose(Rmat[r] @ rngM @ Rmat[r].conj().T, rngM) for r in Rs}
        v0 = PiC @ chi_vec
        singles = fault_list(ops)
        cfgs = singles + [dict(list(singles[i].items()) + list(singles[j].items()))
                          for i, j in rng.choice(len(singles), (40, 2)) if
                          list(singles[i])[0] != list(singles[j])[0]]
        for F in cfgs:
            n_cfg += 1
            direct = out_vec(run(ops, nq, F, "direct"), ops, nq)
            MF = skel_map(ops, nq, F)
            ok &= np.allclose(direct, MF @ chi_vec, atol=1e-9)              # (i) gadget identity
            harmless = np.linalg.norm(direct) < 1e-9 or \
                abs(abs(np.vdot(tau, direct)) - np.linalg.norm(direct)) < 1e-7
            if np.linalg.norm(MF) < 1e-9:
                n_rej += 1
                ok &= harmless
                continue
            found = None
            for r in sorted(Rs, key=lambda r: (r != "I" * nout, not R_in[r])):
                A = (Rmat[r].conj().T @ MF).reshape(-1)
                ov = np.abs(ME.conj() @ A)
                hit = np.where(np.abs(ov - nME * np.linalg.norm(A)) < 1e-7 * (1 + nME * np.linalg.norm(A)))[0]
                if len(hit):
                    found = (r, Ps_in[hit[0]])
                    break
            ok &= found is not None                                          # (iii)
            if found is None:
                continue
            r, e = found
            if r != "I" * nout:
                n_R += 1
                ok &= not R_in[r]                                            # R = I whenever R keeps the range
            w = PiC @ pauli_mat(e) @ chi_vec
            if r == "I" * nout:
                pred = np.linalg.norm(w) < 1e-9 or abs(abs(np.vdot(v0, w)) - np.linalg.norm(v0) * np.linalg.norm(w)) < 1e-7
            else:
                pred = np.linalg.norm(w) < 1e-9
            ok &= pred == harmless                                           # (iv)
            n_harm += not harmless
        a = 0                                                                # (v) Z faults at diagonal gates
        for i, op in enumerate(ops):
            if op[0] == "diag":
                for j, q in enumerate(op[2]):
                    F = {i: "".join("Z" if jj == j else "I" for jj in range(len(op[2])))}
                    MF = skel_map(ops, nq, F)
                    z = "".join("Z" if k == a + j else "I" for k in range(nin))
                    MZ = M @ pauli_mat(z)
                    ok &= abs(abs(np.vdot(MZ.reshape(-1), MF.reshape(-1))) - np.linalg.norm(MZ) * np.linalg.norm(MF)) < 1e-7
                    n_z += 1
                a += len(op[2])
            elif op[0] == "u1":
                a += 2
    print(f"   {n_cfg} configurations ({n_rej} rejected in the skeleton, {n_harm} harmful, {n_R} with R outside"
          f" the range), {n_z} Z faults at diagonal gates; gate kinds {sorted(str(k) for k in kinds_seen if k)}")
    return ok


report("A. Theorem 1 for T, Z rotations, CS, CCZ and arbitrary single-qubit gates", check_theorem1())


# ============================================================== B. Gauss sums for any diagonal inputs
def gauss_elim(q0, L, B):
    """sum_z i^{q0 + sum L_i z_i + 2 sum_{i<j} B_ij z_i z_j}, eliminating one variable at a time
    (the procedure of Proposition 14; same implementation as cultivation_gauss_check.py)"""
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
            factor *= 1 + 1j ** L[v]
            c = 3 if L[v] == 1 else 1
            for u in N:
                L[u] = (L[u] + c) % 4
            for x, y in itertools.combinations(N, 2):
                B[x][y] ^= 1; B[y][x] ^= 1
        else:
            target = (L[v] // 2) % 2
            if not N:
                if target:
                    return 0j
                factor *= 2
                continue
            factor *= 2
            u = N[0]
            others = [x for x in N if x != u]
            alive.remove(u)
            Lu, Bu = L[u], [B[u][x] for x in range(len(L))]
            bits, const = others[:], target
            q0 = (q0 + Lu * const) % 4
            for x in bits:
                L[x] = (L[x] + Lu - 2 * Lu * const) % 4
            for x, y in itertools.combinations(bits, 2):
                if Lu % 2:
                    B[x][y] ^= 1; B[y][x] ^= 1
            for x in alive:
                if Bu[x]:
                    if const:
                        L[x] = (L[x] + 2) % 4
                    for y in bits:
                        if y == x:
                            L[x] = (L[x] + 2) % 4
                        else:
                            B[x][y] ^= 1; B[y][x] ^= 1
    return factor * 1j ** q0


def gf2_rank(rows):
    rows = [int(r) for r in rows]
    rank = 0
    for bit in range(64):
        piv = next((i for i in range(rank, len(rows)) if (rows[i] >> bit) & 1), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and (rows[i] >> bit) & 1:
                rows[i] ^= rows[rank]
        rank += 1
    return rank


def solve_gf2(A, rhs):
    """one solution x of A x = rhs over GF(2) (A: list of int rows over n bits), or None"""
    A = [(int(r), int(b)) for r, b in zip(A, rhs)]
    piv_cols, rows = [], []
    for r, b in A:
        for pc, (pr, pb) in zip(piv_cols, rows):
            if (r >> pc) & 1:
                r ^= pr; b ^= pb
        if r == 0:
            if b:
                return None
            continue
        pc = r.bit_length() - 1
        for k, (pr, pb) in enumerate(rows):
            if (pr >> pc) & 1:
                rows[k] = (pr ^ r, pb ^ b)
        piv_cols.append(pc); rows.append((r, b))
    x = 0
    for pc, (pr, pb) in zip(piv_cols, rows):
        if pb:
            x |= 1 << pc
    return x


BLOCKS3 = {"T": np.array([1, W8]), "Tdg": np.array([1, np.conj(W8)]), "S": np.array([1, 1j]),
           "CS": np.array([1, 1, 1, 1j]), "CCZ": np.array([1, 1, 1, 1, 1, 1, 1, -1], complex)}


def check_code(Pi, chi, blocks, N, r, third, stats, n_samples=12):
    """the checks of part B on one code: Pi the projector onto C, chi the input state"""
    ok = True
    zp = lambda e: np.diag([(-1) ** bin(x & e).count("1") for x in range(2 ** N)]).astype(complex)
    bit = lambda e, q: (e >> (N - 1 - q)) & 1                 # qubit 0 is the most significant bit
    vec = {e: Pi @ zp(e) @ chi for e in range(2 ** N)}
    Sigma, xi = [], {}
    for e in range(2 ** N):
        Ze = zp(e)
        if np.allclose(Ze @ Pi, Pi):
            Sigma.append(e); xi[e] = 1
        elif np.allclose(Ze @ Pi, -Pi):
            Sigma.append(e); xi[e] = -1
    Sset = set(Sigma)
    # cosets of Sigma and their representatives
    rep, cos = {}, []
    for e in range(2 ** N):
        if e in rep:
            continue
        for s in Sigma:
            rep[e ^ s] = e
        cos.append(e)
    live = [c for c in cos if np.linalg.norm(vec[c]) > 1e-9]
    stats['cos'] += len(live)
    stats['det'] += len(live) == 2 ** r
    for e in range(2 ** N):                          # Pi Z^e chi = xi(e + e_beta) v_beta
        c = rep[e]
        ok &= np.allclose(vec[e], xi[e ^ c] * vec[c])
    # xi is a character of Sigma: (-1)^{y.e}
    basis = []
    for s in Sigma:
        if gf2_rank(basis + [s]) > len(basis):
            basis.append(s)
    y = solve_gf2(basis, [0 if xi[s] == 1 else 1 for s in basis])
    ok &= all(xi[s] == (-1) ** bin(y & s).count("1") for s in Sigma)
    for _ in range(n_samples):
        a, b = int(rng.integers(2 ** N)), int(rng.integers(2 ** N))
        stats['pa'] += 1
        E = pauli_mat("".join("X" if bit(a, q) else "I" for q in range(N))) @ \
            pauli_mat("".join("Z" if bit(b, q) else "I" for q in range(N)))          # X^a Z^b
        lhs = Pi @ E @ chi
        # Fourier coefficients of C_a = D^dag X^a D X^a, block by block
        touched = [blk for blk in blocks if any(bit(a, q) for q in blk[2])]
        cdiag = np.array([1.0 + 0j])
        tq = []
        for name, dv, qs in touched:
            k = len(qs)
            ab = sum(bit(a, q) << (k - 1 - j) for j, q in enumerate(qs))
            Xd = np.array([dv[x ^ ab] for x in range(2 ** k)])
            cdiag = np.kron(cdiag, np.conj(dv) * Xd)
            tq += qs
        K = len(tq)
        chat = np.array([sum(cdiag[x] * (-1) ** bin(x & z).count("1") for x in range(2 ** K)) / 2 ** K
                         for z in range(2 ** K)])
        emb = lambda z: sum(((z >> (K - 1 - j)) & 1) << (N - 1 - q) for j, q in enumerate(tq))
        g = {c: 0j for c in live}
        for z in range(2 ** K):
            e = emb(z) ^ b
            c = rep[e]
            if c in g:
                g[c] += chat[z] * xi[e ^ c]
        rhs = (-1) ** bin(a & b).count("1") * sum(g[c] * vec[c] for c in live)
        ok &= np.allclose(lhs, rhs)
        if not third:
            continue
        stats['third'] += 1
        # C_a is a diagonal Clifford: cdiag = phase * i^{q(x)}, q quadratic with even cross terms
        ph = cdiag[0]
        qv = np.round(np.angle(cdiag / ph) / (np.pi / 2)).astype(int) % 4
        ok &= np.allclose(cdiag / ph, 1j ** qv)
        Lq = [int(qv[1 << (K - 1 - j)]) for j in range(K)]
        Bq = [[0] * K for _ in range(K)]
        for i_, j_ in itertools.combinations(range(K), 2):
            v = (int(qv[(1 << (K - 1 - i_)) | (1 << (K - 1 - j_))]) - Lq[i_] - Lq[j_]) % 4
            ok &= v % 2 == 0
            Bq[i_][j_] = Bq[j_][i_] = v // 2
        ok &= all(qv[x] == (sum(Lq[j] for j in range(K) if (x >> (K - 1 - j)) & 1) +
                            2 * sum(Bq[i_][j_] for i_, j_ in itertools.combinations(range(K), 2)
                                    if (x >> (K - 1 - i_)) & 1 and (x >> (K - 1 - j_)) & 1)) % 4
                  for x in range(2 ** K))
        for c in live:
            # {z in F2^K : emb(z) + b in c + Sigma} as an affine space z0 + span(k_l): the condition is that
            # emb(z) + b + c is orthogonal to the annihilator of Sigma
            ann = [h for h in range(2 ** N) if all(bin(h & s).count("1") % 2 == 0 for s in basis)]
            annb = []
            for h in ann:
                if gf2_rank(annb + [h]) > len(annb):
                    annb.append(h)
            rows = [sum(((h >> (N - 1 - q)) & 1) << (K - 1 - j) for j, q in enumerate(tq)) for h in annb]
            rhsb = [bin(h & (b ^ c)).count("1") % 2 for h in annb]
            z0 = solve_gf2(rows, rhsb)
            if z0 is None:
                ok &= abs(g[c]) < 1e-9
                continue
            kern = [z for z in range(1, 2 ** K) if all(bin(rw & z).count("1") % 2 == 0 for rw in rows)]
            kb = []
            for z in kern:
                if gf2_rank(kb + [z]) > len(kb):
                    kb.append(z)
            m = len(kb)
            # variables: x_0..x_{K-1}, u_0..u_{m-1}; exponent q(x) + 2 x.(z0 + sum u_l k_l)
            #            + 2 y.(emb(z0 + sum u k) + b + c)
            nv = K + m
            L = Lq + [0] * m
            Bm = [[0] * nv for _ in range(nv)]
            for i_ in range(K):
                for j_ in range(K):
                    Bm[i_][j_] = Bq[i_][j_]
            xb = lambda z, j: (z >> (K - 1 - j)) & 1
            for j in range(K):
                L[j] = (L[j] + 2 * xb(z0, j)) % 4
            for l_, kl in enumerate(kb):
                for j in range(K):
                    if xb(kl, j):
                        Bm[j][K + l_] ^= 1; Bm[K + l_][j] ^= 1
                L[K + l_] = (L[K + l_] + 2 * (bin(y & emb(kl)).count("1") % 2)) % 4
            q0 = 2 * (bin(y & (emb(z0) ^ b ^ c)).count("1") % 2)
            Gs = gauss_elim(q0, L, Bm)
            gval = ph * Gs / 2 ** K
            ok &= np.isclose(gval, g[c], atol=1e-9)
            if abs(g[c]) > 1e-9:
                lg = np.log2(abs(g[c]) * 2 ** K) * 2
                ok &= abs(lg - round(lg)) < 1e-9
    return ok


def check_gauss_general(n_codes=60):
    ok, stats = True, dict(pa=0, cos=0, third=0, det=0)
    for trial in range(n_codes):
        third = trial % 3 != 2
        blocks, N = [], 0
        while N < 4:
            if third:
                name = str(rng.choice(list(BLOCKS3)))
                dv = BLOCKS3[name]
            else:
                name = str(rng.choice(["Rz", "sqrtT", "T", "CS"]))
                dv = {"Rz": np.array([1, np.exp(1j * rng.uniform(0, 2 * np.pi))]),
                      "sqrtT": np.array([1, np.exp(1j * np.pi / 8)]), "T": BLOCKS3["T"], "CS": BLOCKS3["CS"]}[name]
            k = int(np.log2(len(dv)))
            if N + k > 6:
                continue
            blocks.append((name, dv, list(range(N, N + k))))
            N += k
        r = int(rng.choice([1, 2]))
        tab = stim.Tableau.random(N)
        gens = [tab.z_output(j) for j in range(r, N)]
        chi = np.array([1.0 + 0j])
        for name, dv, qs in blocks:
            chi = np.kron(chi, dv * 2 ** (-len(qs) / 2))
        gm = [g.to_unitary_matrix(endian="big") for g in gens]
        for _ in range(20):
            signs = rng.choice([1, -1], len(gens))
            Pi = np.eye(2 ** N, dtype=complex)
            for s, g in zip(signs, gm):
                Pi = Pi @ (np.eye(2 ** N) + s * g) / 2
            if np.linalg.norm(Pi @ chi) > 1e-6:
                break
        else:
            continue
        ok &= check_code(Pi, chi, blocks, N, r, third, stats)
    print(f"   {stats['pa']} Pauli operators on {n_codes} random codes, {stats['cos']} accepted cosets in all, {stats['det']} codes"
          f" with 2^r accepted cosets; {stats['third']} operators on third-level inputs checked as Z4 Gauss sums")
    return ok


report("B. Gauss sums for third-level diagonal inputs, any angle and any checks", check_gauss_general())


def check_ccz_circuit():
    """skeleton of |+++> -> CCZ (injection) -> CCZ^dag -> proj X_1 = +1 -> CCZ: nine CCZ inputs, r = 3;
    its check is deterministic, so exactly 2^3 cosets are accepted and their outputs are Z^c |CCZ>"""
    ccz = BLOCKS3["CCZ"]
    ops = [("g1", "H", 0), ("g1", "H", 1), ("g1", "H", 2), ("diag", ccz, [0, 1, 2]), ("diag", ccz, [0, 1, 2]),
           ("proj", "X", [0], 1), ("diag", ccz, [0, 1, 2])]
    M = skel_map(ops, 3, {})
    Gm = M.conj().T @ M
    ev = np.linalg.eigvalsh(Gm)
    nz = ev[ev > 1e-9]
    ok = np.allclose(nz, nz[0]) and len(nz) == 8
    Pi = Gm / nz[0]
    blocks = [("CCZ", ccz, [0, 1, 2]), ("CCZ", ccz, [3, 4, 5]), ("CCZ", ccz, [6, 7, 8])]
    chi = np.kron(np.kron(ccz, ccz), ccz) / 2 ** 4.5
    tau = ccz / np.sqrt(8)
    out0 = M @ chi
    ok &= abs(abs(np.vdot(tau, out0)) - np.linalg.norm(out0)) < 1e-9
    stats = dict(pa=0, cos=0, third=0, det=0)
    ok &= check_code(Pi, chi, blocks, 9, 3, True, stats, n_samples=150)
    ok &= stats["det"] == 1
    # the accepted cosets decode to Z^c tau, c in F2^3
    zp = lambda e: np.diag([(-1) ** bin(x & e).count("1") for x in range(2 ** 9)]).astype(complex)
    outs = []
    for e in range(2 ** 9):
        o = M @ zp(e) @ chi
        if np.linalg.norm(o) > 1e-9:
            o = o / np.linalg.norm(o)
            c = [cc for cc in range(8) if abs(abs(np.vdot(np.diag([(-1) ** bin(x & cc).count("1") for x in range(8)]) @ tau, o)) - 1) < 1e-9]
            ok &= len(c) == 1
            outs.append(c[0] if c else None)
    ok &= set(outs) == set(range(8))
    print(f"   CCZ check (9 inputs, r = 3): {stats['cos']} accepted cosets, decoded to all 8 states Z^c|CCZ>;"
          f" {stats['pa']} Pauli operators checked as Z4 Gauss sums")
    return ok


report("B'. Deterministic check on CCZ inputs (2^r Gauss sums)", check_ccz_circuit())


# ============================================================== shared: codes as state vectors
def color_faces(bw):
    sys.path.insert(0, __file__.rsplit("checks", 1)[0] + "evaluator/src")
    from colorcode import color_code_faces
    return color_code_faces(bw)


def zsign(n, qs):
    """(-1)^{sum_{q in qs} x_q} as a broadcastable tensor"""
    out = np.ones([2] * n)
    for q in qs:
        sh = [1] * n
        sh[q] = 2
        out = out * np.array([1, -1]).reshape(sh)
    return out


def xflip(psi, qs):
    return np.flip(psi, axis=tuple(qs)) if qs else psi


def diag_layer(n, thetas):
    ph = np.zeros([2] * n)
    for q, th in enumerate(thetas):
        sh = [1] * n
        sh[q] = 2
        ph = ph + np.array([0, th]).reshape(sh)
    return np.exp(1j * ph)


def code_projector(n, xst, zst):
    def P(psi):
        for f in zst:
            psi = (psi + zsign(n, f) * psi) / 2
        for f in xst:
            psi = (psi + xflip(psi, f)) / 2
        return psi
    return P


def logical_basis(n, xst, zst, xl):
    P = code_projector(n, xst, zst)
    z = np.zeros([2] * n, complex)
    z[(0,) * n] = 1
    zero = P(z)
    zero /= np.linalg.norm(zero)
    one = xflip(zero, xl)
    return zero, one, P


def min_weight_x_logicals(n, xst, zst, xl, delta):
    """all X logical supports of weight delta (X^l commutes with every Z stabilizer, not in the X stabilizers)"""
    out = []
    zs = [set(f) for f in zst]
    span = {0}
    for f in xst:
        v = sum(1 << q for q in f)
        span |= {s ^ v for s in span}
    for l in itertools.combinations(range(n), delta):
        if all(len(set(l) & f) % 2 == 0 for f in zs) and sum(1 << q for q in l) not in span:
            out.append(l)
    return out


# ============================================================== C. Coherent completion on CSS codes
def check_completion():
    ok, n_cases, n_cancel = True, 0, 0
    codes = []
    for bw in (3, 5):
        faces, n = color_faces(bw)
        codes.append((f"[[{n},1,{bw}]] colour", n, faces, faces, list(range(n)), bw))
    sx = [[0, 1, 3, 4], [4, 5, 7, 8], [1, 2], [6, 7]]
    sz = [[1, 2, 4, 5], [3, 4, 6, 7], [0, 3], [5, 8]]
    codes.append(("[[9,1,3]] surface", 9, sx, sz, [0, 3, 6], 3))
    for name, n, xst, zst, xl, delta in codes:
        zero, one, P = logical_basis(n, xst, zst, xl)
        al = rng.normal(size=2) + 1j * rng.normal(size=2)
        al /= np.linalg.norm(al)
        tau = al[0] * zero + al[1] * one
        Ls = min_weight_x_logicals(n, xst, zst, xl, delta) if n <= 9 else None
        if Ls is None:
            sys.path.insert(0, __file__.rsplit("checks", 1)[0] + "evaluator/src")
            from foldtree import min_weight_logicals
            _, Lf = min_weight_logicals(list(range(n)), xst, list(range(n)))
            Ls = [tuple(sorted(l)) for l in Lf]
        zspan = {}
        for signs in itertools.product([0, 1], repeat=len(zst)):
            v = 0
            for s, f in zip(signs, zst):
                if s:
                    v ^= sum(1 << q for q in f)
            zspan[v] = 1                                  # all Z stabilizers have sign +1 on |0_L>, |1_L>
        for li, l in enumerate(Ls[: 12]):
            Xl = xflip(tau, list(l))
            for _ in range(4):
                k = int(rng.integers(1, delta))
                u = [int(x) for x in rng.choice(l, k, replace=False)]
                rest = [q for q in l if q not in u]
                kind = rng.integers(4)
                th = rng.uniform(0, 2 * np.pi, n) if kind == 0 else \
                    rng.choice([np.pi / 4, -np.pi / 4], n) if kind == 1 else \
                    rng.choice([np.pi / 2, -np.pi / 2], n) if kind == 2 else rng.choice([np.pi / 8, -np.pi / 4], n)
                V = diag_layer(n, th)
                before = np.conj(V) * tau                 # V^dag |tau_L>
                phi = P(xflip(V * xflip(before, u), rest))
                Wu = [w for w in range(1 << n) if (w & ~sum(1 << q for q in u)) == 0 and w in zspan]
                amp = 0j
                for w in Wu:
                    c = 1 + 0j
                    for q in u:
                        c *= (-1j * np.sin(th[q])) if (w >> q) & 1 else np.cos(th[q])
                    amp += c
                ok &= np.allclose(phi, amp * Xl, atol=1e-9)
                if len(Wu) == 1:
                    ok &= abs(np.linalg.norm(phi) ** 2 - np.prod([np.cos(th[q]) ** 2 for q in u])) < 1e-9
                else:
                    n_cancel += 1
                n_cases += 1
        # the [[9,1,3]] case with the weight-2 Z stabilizer Z0 Z3 inside l = {0, 3, 6}: T and T^dag cancel
        if name.startswith("[[9"):
            for s0, s3, expect in ((np.pi / 4, -np.pi / 4, 1.0), (np.pi / 4, np.pi / 4, 0.0)):
                th = np.zeros(n)
                th[0], th[3] = s0, s3
                V = diag_layer(n, th)
                phi = P(xflip(V * xflip(np.conj(V) * tau, [0, 3]), [6]))
                ok &= abs(np.linalg.norm(phi) ** 2 - expect) < 1e-9
                n_cases += 1
    print(f"   {n_cases} configurations on the [[7,1,3]], [[19,1,5]] colour codes and the [[9,1,3]] surface code,"
          f" {n_cancel} with a Z stabilizer inside u")
    return ok


report("C. Coherent completion for CSS codes and any angles", check_completion())


# ============================================================== D. Strongly fault-tolerant transversal checks
def check_strong_ft(trials=150):
    sys.path.insert(0, __file__.rsplit("checks", 1)[0] + "evaluator/src")
    from tdem import sublattice_signs
    ok, n_tot, n_rej, n_lemma = True, 0, 0, 0
    for bw in (3, 5):
        faces, n = color_faces(bw)
        delta = bw
        zero, one, P = logical_basis(n, faces, faces, list(range(n)))
        signs = sublattice_signs(n, faces)
        th = np.array([np.pi / 4 if s > 0 else -np.pi / 4 for s in signs])
        U = diag_layer(n, th)
        G = lambda psi: U * xflip(np.conj(U) * psi, list(range(n)))          # U X^n U^dag
        Gm = np.array([[np.vdot(a, G(b)) for b in (zero, one)] for a in (zero, one)])
        ok &= np.allclose(Gm.conj().T @ Gm, np.eye(2))                       # G preserves the code space
        evals, evecs = np.linalg.eig(Gm)
        j = int(np.argmin(np.abs(evals - 1)))
        ok &= abs(evals[j] - 1) < 1e-9
        tau = evecs[0, j] * zero + evecs[1, j] * one
        tau /= np.linalg.norm(tau)
        letters = "IXYZ"

        def rand_pauli(qs):
            return {q: letters[int(rng.integers(4))] for q in qs}

        def ap(psi, Pd):
            for q, ch in Pd.items():
                if ch != "I":
                    psi = apply1(psi, P1[ch], q)
            return psi

        def circuit(E, s, stab):
            psi = ap(tau, E[0])
            psi = np.conj(U) * psi
            psi = ap(psi, E[1])
            psi = (psi + s * xflip(psi, list(range(n)))) / 2
            psi = ap(psi, E[2])
            psi = U * psi
            psi = ap(psi, E[3])
            f, typ, sg = stab
            psi = (psi + sg * (xflip(psi, f) if typ == "X" else zsign(n, f) * psi)) / 2
            psi = ap(psi, E[4])
            return P(psi)

        for _ in range(trials):
            k = int(rng.integers(1, delta))
            Q = [int(x) for x in rng.choice(n, k, replace=False)]
            E = [rand_pauli(list(rng.choice(Q, int(rng.integers(0, k + 1)), replace=False))) for _ in range(5)]
            if rng.random() < 0.7:                     # mostly X errors, which real T gates make coherent
                E = [{q: ("X" if ch != "I" else "I") for q, ch in e.items()} for e in E]
            stab = (faces[int(rng.integers(len(faces)))], str(rng.choice(["X", "Z"])), 1 if rng.random() < 0.8 else -1)
            out = circuit(E, 1 if rng.random() < 0.8 else -1, stab)
            nrm = np.linalg.norm(out)
            n_tot += 1
            n_rej += nrm < 1e-9
            ok &= nrm < 1e-9 or abs(abs(np.vdot(tau, out)) - nrm) < 1e-9
        # control: a hook on two qubits of a minimum-weight logical before U, completed after it (delta qubits)
        sys.path.insert(0, __file__.rsplit("checks", 1)[0] + "evaluator/src")
        from foldtree import min_weight_logicals
        _, Lf = min_weight_logicals(list(range(n)), faces, list(range(n)))
        l = sorted(Lf[0])
        E = [{}, {}, {q: "X" for q in l[:2]}, {q: "X" for q in l[2:]}, {}]
        out = circuit(E, 1, (faces[0], "Z", 1))
        nrm = np.linalg.norm(out)
        harmful = nrm > 1e-9 and abs(abs(np.vdot(tau, out)) - nrm) > 1e-6
        ok &= harmful and abs(nrm ** 2 - 0.25) < 1e-9
        n_lemma += harmful
    print(f"   {n_tot} configurations with fewer than delta qubits in the union of supports ({n_rej} rejected),"
          f" none harmful; {n_lemma} controls with delta qubits harmful as Lemma 4 predicts")
    return ok


report("D. Transversal checks with strongly fault-tolerant gadgets", check_strong_ft())
print("ALL OK" if ALL_OK else "FAILED")
