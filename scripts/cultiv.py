"""Circuit-level fault analysis of Clifford+T circuits (paper/cultivation.tex, Theorem 2).

A circuit is a list of operations on nq qubits, all starting in |0>:
  ("R", q, b)            preparation of qubit q (in |0>) in |0> (b="Z") or |+> (b="X")
  ("U1", q, g)           single-qubit Clifford g in G1
  ("T", q, k)            T^k, k = +1 or -1
  ("U2", g, c, t)        two-qubit Clifford: g in {"CX", "CY", "CZ"} (controlled-Pauli from c onto t)
  ("M", P, qs, s, reset) projection (I + s P_qs)/2 for a Pauli string P on qubits qs; if reset, the (single)
                         measured qubit is reset to |0> afterwards
  ("I", qs)              identity on the qubits qs (a location for idle noise)
  ("RST", q, b)          ideal discard of qubit q, which is in an eigenstate of b_q (b = "Z" or "X") for every
                         input of the skeleton, and reset to |0>
Each operation other than RST is a noisy location unless its index is in the set `ideal`. Faults: a Pauli after R, U1, T, U2
on its support, and a Pauli before M on its support or a flip of its sign ("flip"). The output is the state of
the qubits in `out`; all other qubits must end in |0>. A fault configuration is harmless when the output is a
multiple of the fault-free output (zero included).

The skeleton replaces T gate j by CX(q -> a_j), <0|_{a_j} on an ideal input |T>_{a_j} (and S^dagger for T^-1).
`Skeleton` follows its Choi state symbolically and returns, for every fault configuration, the detector vector
D(F) and the Pauli E(F) on the T inputs modulo the stabilizers of the code C, both linear in F.
"""
import itertools
import numpy as np

W8 = np.exp(1j * np.pi / 4)
P1 = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], dtype=complex),
      "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]).astype(complex)}
G1 = {"H": np.array([[1, 1], [1, -1]]) / np.sqrt(2), "S": np.diag([1, 1j]), "Sdg": np.diag([1, -1j]),
      "X": P1["X"], "Y": P1["Y"], "Z": P1["Z"], "I": P1["I"]}
G2 = {"CX": "X", "CY": "Y", "CZ": "Z"}
TOL = 1e-9


# ------------------------------------------------------------------ state vectors
def _apply1(psi, U, q):
    return np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)


def _apply_paulis(psi, ps, qs):
    for p, q in zip(ps, qs):
        if p != "I":
            psi = _apply1(psi, P1[p], q)
    return psi


def _apply_cp(psi, P, c, t):
    """controlled-P from c onto t"""
    psi = psi.copy()
    idx = [slice(None)] * psi.ndim
    idx[c] = 1
    sub = psi[tuple(idx)]
    tt = t - 1 if t > c else t
    psi[tuple(idx)] = _apply1(sub, P1[P], tt)
    return psi


def _reset(psi, q, b):
    """reset of a qubit that is in an eigenstate of Z (b="Z") or X (b="X") to |0>"""
    idx0 = [slice(None)] * psi.ndim
    idx1 = list(idx0)
    idx0[q], idx1[q] = 0, 1
    a0, a1 = psi[tuple(idx0)], psi[tuple(idx1)]
    out = np.zeros_like(psi)
    out[tuple(idx0)] = a0 + a1 if b == "Z" else np.sqrt(2) * a0
    return out


def simulate(ops, nq, F=None, texp=None, zafter=None, tinput=None):
    """final state with fault configuration F (dict op index -> Pauli string or "flip"); texp[j] replaces the
    exponent of T gate j, zafter is a set of T gates followed by Z (used to apply Paulis on the T inputs);
    tinput[j] = (a0, a1) replaces T gate j by its injection gadget fed with a0|0> + a1|1> (the skeleton)"""
    F = F or {}
    psi = np.zeros([2] * nq, dtype=complex)
    psi[(0,) * nq] = 1
    j = 0
    for i, op in enumerate(ops):
        f = F.get(i)
        kind = op[0]
        if kind == "M":
            _, P, qs, s, reset = op
            if f is not None and f != "flip":
                psi = _apply_paulis(psi, f, qs)
            if f == "flip":
                s = -s
            psi = (psi + s * _apply_paulis(psi, P, qs)) / 2
            if reset:
                psi = _reset(psi, qs[0], "Z" if P == "Z" else "X")
            continue
        if kind == "RST":
            psi = _reset(psi, op[1], op[2])
            continue
        if kind == "I":
            pass
        elif kind == "R":
            if op[2] == "X":
                psi = _apply1(psi, G1["H"], op[1])
        elif kind == "U1":
            psi = _apply1(psi, G1[op[2]], op[1])
        elif kind == "U2":
            psi = _apply_cp(psi, G2[op[1]], op[2], op[3])
        elif kind == "T":
            if tinput is not None:
                a0, a1 = tinput[j]
                psi = _apply1(psi, np.sqrt(2) * np.diag([a0, a1]), op[1])
                if op[2] == -1:
                    psi = _apply1(psi, G1["Sdg"], op[1])
            else:
                k = op[2] if texp is None else texp.get(j, op[2])
                psi = _apply1(psi, np.diag([1, W8 ** k]), op[1])
            if zafter and j in zafter:
                psi = _apply1(psi, P1["Z"], op[1])
            j += 1
        if f is not None:
            psi = _apply_paulis(psi, f, support(op))
    return psi


def support(op):
    if op[0] in ("R", "U1", "T", "RST"):
        return [op[1]]
    if op[0] == "I":
        return list(op[1])
    if op[0] == "U2":
        return [op[2], op[3]]
    return list(op[2])


def locations(ops, ideal=()):
    """(op index, list of elementary faults) for every noisy operation"""
    locs = []
    for i, op in enumerate(ops):
        if i in ideal or op[0] == "RST":
            continue
        sup = support(op)
        ps = ["".join(p) for p in itertools.product("IXYZ", repeat=len(sup)) if set(p) != {"I"}]
        if op[0] == "M":
            ps = (ps if len(sup) <= 2 else []) + ["flip"]
        locs.append((i, ps))
    return locs


def resolve(ops, nq):
    """fix the sign of every projection to its fault-free outcome and the basis of every RST; returns the new
    list and the list of (op index, probability of the kept outcome) for projections that are not
    deterministic in the fault-free run"""
    ops = list(ops)
    random = []
    for i, op in enumerate(ops):
        if op[0] == "M":
            psi = simulate(ops[:i], nq)
            n0 = np.vdot(psi, psi).real
            _, P, qs, s, reset = op
            Ppsi = _apply_paulis(psi, P, qs)
            pplus = (n0 + np.vdot(psi, Ppsi).real) / (2 * n0)
            s = 1 if pplus > 0.5 - 1e-9 else -1
            if min(pplus, 1 - pplus) > 1e-9:
                random.append((i, max(pplus, 1 - pplus)))
            ops[i] = ("M", P, qs, s, reset)
        elif op[0] == "RST":
            psi = simulate(ops[:i], nq)
            n0 = np.vdot(psi, psi).real
            for b in ("Z", "X"):
                ev = np.vdot(psi, _apply_paulis(psi, b, [op[1]])).real / n0
                if abs(abs(ev) - 1) < 1e-9:
                    ops[i] = ("RST", op[1], b)
                    break
            else:
                raise ValueError("discarded qubit %d is not in a Z or X eigenstate" % op[1])
    return ops, random


def out_vec(psi, nq, out):
    idx = tuple(slice(None) if q in out else 0 for q in range(nq))
    return psi[idx].reshape(-1)


def parallel(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < TOL * max(1.0, nv):
        return True
    return abs(abs(np.vdot(v, u)) - nu * nv) < 1e-8 * nu * nv


# ------------------------------------------------------------------ Pauli frames and the symbolic skeleton
def _conj1(g, x, z):
    """conjugate a single-qubit Pauli (x, z) by Clifford g, up to sign"""
    if g in ("H",):
        return z, x
    if g in ("S", "Sdg"):
        return x, z ^ x
    return x, z


def _conj2(g, xc, zc, xt, zt):
    """conjugate (c, t) Pauli bits by controlled-P (c -> t), up to sign"""
    px, pz = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}[g]
    # controlled-P = |0><0| + |1><1| P. X_c -> X_c P_t ; Z_c -> Z_c ; P_t -> P_t ; Q_t -> Z_c Q_t if {Q,P}=0
    anti = (xt & pz) ^ (zt & px)          # target part anticommutes with P
    xt2, zt2 = xt ^ (xc & px), zt ^ (xc & pz)
    zc2 = zc ^ anti
    return xc, zc2, xt2, zt2


class Pauli:
    """unsigned Pauli on N qubits as two Python ints (bit q = qubit q)"""
    __slots__ = ("x", "z")

    def __init__(self, x=0, z=0):
        self.x, self.z = x, z

    def copy(self):
        return Pauli(self.x, self.z)

    def get(self, q):
        return (self.x >> q) & 1, (self.z >> q) & 1

    def set(self, q, xb, zb):
        m = 1 << q
        self.x = (self.x & ~m) | (xb << q)
        self.z = (self.z & ~m) | (zb << q)

    def anticommutes(self, o):
        return (bin(self.x & o.z).count("1") + bin(self.z & o.x).count("1")) & 1

    def mul(self, o):
        self.x ^= o.x
        self.z ^= o.z

    def conj1(self, g, q):
        xb, zb = self.get(q)
        self.set(q, *_conj1(g, xb, zb))

    def conj2(self, g, c, t):
        xc, zc = self.get(c)
        xt, zt = self.get(t)
        xc, zc, xt, zt = _conj2(g, xc, zc, xt, zt)
        self.set(c, xc, zc)
        self.set(t, xt, zt)


def pauli_from_string(P, qs):
    p = Pauli()
    for c, q in zip(P, qs):
        p.set(q, int(c in "XY"), int(c in "YZ"))
    return p


class Skeleton:
    """Choi-state analysis of the skeleton. Qubits: 0..nq-1 physical, nq+j input a_j, nq+t+j reference r_j."""

    def __init__(self, ops, nq, out):
        self.ops, self.nq, self.out = ops, nq, list(out)
        self.tgates = [i for i, op in enumerate(ops) if op[0] == "T"]
        self.t = t = len(self.tgates)
        # projection indices: every M, and every T gadget removal
        self.proj_of_op, self.proj_of_t = {}, {}
        k = 0
        for i, op in enumerate(ops):
            if op[0] == "M":
                self.proj_of_op[i] = k
                k += 1
            elif op[0] == "T":
                self.proj_of_t[i] = k
                k += 1
        self.nproj = k
        self._run_choi()

    # generators: list of [Pauli, a] with a an int over projection indices (sign dependence)
    def _run_choi(self):
        nq, t = self.nq, self.t
        gens = []
        for q in range(nq):
            gens.append([Pauli(0, 1 << q), 0])
        for j in range(t):
            a, r = nq + j, nq + t + j
            gens.append([Pauli((1 << a) | (1 << r), 0), 0])
            gens.append([Pauli(0, (1 << a) | (1 << r)), 0])
        self.det_rows = []          # (projection index, linear form over projections)
        self.deterministic = {}
        j = 0
        for i, op in enumerate(self.ops):
            kind = op[0]
            if kind == "R":
                if op[2] == "X":
                    for g in gens:
                        g[0].conj1("H", op[1])
            elif kind == "U1":
                for g in gens:
                    g[0].conj1(op[2], op[1])
            elif kind == "U2":
                for g in gens:
                    g[0].conj2(G2[op[1]], op[2], op[3])
            elif kind == "T":
                q, a = op[1], nq + j
                for g in gens:
                    g[0].conj2("X", q, a)
                p = self.proj_of_t[i]
                gens = self._project(gens, pauli_from_string("Z", [a]), p)
                gens = self._remove(gens, a, "Z")
                if op[2] == -1:
                    for g in gens:
                        g[0].conj1("Sdg", q)
                j += 1
            elif kind == "M":
                _, P, qs, s, reset = op
                p = self.proj_of_op[i]
                gens = self._project(gens, pauli_from_string(P, qs), p)
                if reset:
                    gens = self._remove(gens, qs[0], "Z" if P == "Z" else "X")
                    gens.append([Pauli(0, 1 << qs[0]), 0])
            elif kind == "RST":
                q, b = op[1], op[2]
                Q = pauli_from_string(b, [q])
                comb = self._express(gens, Q)      # asserts that the discarded qubit is deterministic
                for k in comb[1:]:
                    gens[comb[0]][0].mul(gens[k][0])
                    gens[comb[0]][1] ^= gens[k][1]
                gens = self._remove(gens, q, b)
                gens.append([Pauli(0, 1 << q), 0])
        self.gens = gens
        self._final()

    def _project(self, gens, Q, p):
        anti = [k for k, g in enumerate(gens) if g[0].anticommutes(Q)]
        if anti:
            k0 = anti[0]
            for k in anti[1:]:
                gens[k][0].mul(gens[k0][0])
                gens[k][1] ^= gens[k0][1]
            gens[k0] = [Q.copy(), 1 << p]
            self.deterministic[p] = False
        else:
            comb = self._express(gens, Q)
            form = 1 << p
            for k in comb:
                form ^= gens[k][1]
            self.det_rows.append((p, form))
            self.deterministic[p] = True
            # make Q a generator so that the qubit can be removed
            k0 = comb[0]
            for k in comb[1:]:
                gens[k0][0].mul(gens[k][0])
                gens[k0][1] ^= gens[k][1]
        return gens

    def _express(self, gens, Q):
        """indices of generators whose product is Q (up to sign)"""
        N = 2 * (self.nq + 2 * self.t)
        rows = [(g[0].x | (g[0].z << (N // 2)), 1 << k) for k, g in enumerate(gens)]
        target = Q.x | (Q.z << (N // 2))
        basis = []
        for v, tag in rows:
            for b, bt in basis:
                if v & (1 << (b.bit_length() - 1)):
                    v ^= b
                    tag ^= bt
            if v:
                basis.append((v, tag))
                basis.sort(key=lambda bt: -bt[0])
        tag = 0
        v = target
        for b, bt in basis:
            if v & (1 << (b.bit_length() - 1)):
                v ^= b
                tag ^= bt
        assert v == 0, "deterministic projection not in the stabilizer group"
        return [k for k in range(len(gens)) if (tag >> k) & 1]

    def _remove(self, gens, q, b):
        """qubit q is in an eigenstate of b_q (a generator); drop it"""
        Q = pauli_from_string(b, [q])
        k0 = None
        for k, g in enumerate(gens):
            if g[0].x == Q.x and g[0].z == Q.z:
                k0 = k
                break
        assert k0 is not None
        for k, g in enumerate(gens):
            if k != k0 and (g[0].get(q) != (0, 0)):
                g[0].mul(gens[k0][0])
                g[1] ^= gens[k0][1]
                assert g[0].get(q) == (0, 0)
        return gens[:k0] + gens[k0 + 1:]

    def _final(self):
        nq, t = self.nq, self.t
        self.N = N = nq + 2 * t
        gens = self.gens
        m = len(gens)
        assert m == nq + t, (m, nq + t)
        # symplectic matrix rows: (x | z) as ints of 2N bits
        vec = [g[0].x | (g[0].z << N) for g in gens]
        # destabilizers: d_k with <d_k, g_l> = delta_kl. Solve with A Omega d = e_k.
        A = np.array([[(v >> b) & 1 for b in range(2 * N)] for v in vec], dtype=np.uint8)
        Aw = np.concatenate([A[:, N:], A[:, :N]], axis=1)       # <d, g> = Aw @ d
        self.destab = _right_inverse(Aw)                        # 2N x m, columns d_k
        self.vec = vec
        physmask = (1 << nq) - 1
        refmask = ((1 << t) - 1) << (nq + t)
        # S_C: group elements with zero physical part -> reference parts
        phys = [((v & physmask) | (((v >> N) & physmask) << nq)) for v in vec]
        self._phys_rows = phys
        sc = _kernel_combinations(phys, m)
        self.SC = []
        for comb in sc:
            v = 0
            for k in range(m):
                if (comb >> k) & 1:
                    v ^= vec[k]
            self.SC.append(self._ref_part(v))
        self.SC_rref = _rref([s for s in self.SC])
        self.det_forms = {p: f for p, f in self.det_rows}

    def _ref_part(self, v):
        """reference part of a 2N-bit symplectic vector, as 2t bits (x then z), r_j -> bit j"""
        nq, t, N = self.nq, self.t, self.N
        x = (v >> (nq + t)) & ((1 << t) - 1)
        z = (v >> (N + nq + t)) & ((1 << t) - 1)
        return x | (z << t)

    # -------------------------------------------------------------- faults
    def propagate(self, start, P):
        """push Pauli P (on physical qubits, Pauli object) from just before op `start` to the end; returns
        (flips as int over projections, residual Pauli on physical qubits)"""
        nq = self.nq
        flips = 0
        P = P.copy()
        j = sum(1 for i in self.tgates if i < start)
        for i in range(start, len(self.ops)):
            op = self.ops[i]
            kind = op[0]
            if kind == "R":
                if op[2] == "X":
                    P.conj1("H", op[1])
            elif kind == "U1":
                P.conj1(op[2], op[1])
            elif kind == "U2":
                P.conj2(G2[op[1]], op[2], op[3])
            elif kind == "T":
                q = op[1]
                xb, zb = P.get(q)
                if xb:
                    flips ^= 1 << self.proj_of_t[i]   # X on a_j after the CX flips the removal
                if op[2] == -1:
                    P.conj1("Sdg", q)
                j += 1
            elif kind == "M":
                _, Pm, qs, s, reset = op
                if P.anticommutes(pauli_from_string(Pm, qs)):
                    flips ^= 1 << self.proj_of_op[i]
                if reset:
                    P.set(qs[0], 0, 0)
            elif kind == "RST":
                P.set(op[1], 0, 0)
        return flips, P

    def fault_effect(self, i, f):
        """(flips, residual) of elementary fault f at op i"""
        op = self.ops[i]
        if f == "flip":
            return 1 << self.proj_of_op[i], Pauli()
        P = pauli_from_string(f, support(op))
        return self.propagate(i if op[0] == "M" else i + 1, P)

    def detectors(self, flips):
        """D(F): bit p set for every deterministic projection p whose sign condition fails"""
        d = 0
        for p, form in self.det_rows:
            if bin(form & flips).count("1") & 1:
                d |= 1 << p
        return d

    def input_pauli(self, flips, residual):
        """(U, E): E(F) as 2t bits (x | z << t) on the T inputs, reduced modulo S_C, and U a Pauli on the
        physical qubits (x | z << nq) left over at the output; U = 0 whenever D(F) = 0 and the circuit maps
        onto its whole output space, as in Theorem 2"""
        nq, t, N = self.nq, self.t, self.N
        m = len(self.gens)
        c = np.array([bin(g[1] & flips).count("1") & 1 for g in self.gens], dtype=np.uint8)
        dvec = (self.destab @ c) % 2                              # 2N bits
        R = 0
        for b in np.nonzero(dvec)[0]:
            R |= 1 << int(b)
        G = R ^ (residual.x | (residual.z << N))
        # find a stabilizer with the same physical part as G
        gphys = (G & ((1 << nq) - 1)) | (((G >> N) & ((1 << nq) - 1)) << nq)
        comb, rest = _solve_combination(self._phys_rows, gphys)
        s = 0
        for k in range(m):
            if (comb >> k) & 1:
                s ^= self.vec[k]
        E = self._ref_part(G ^ s)
        return rest, _reduce(E, self.SC_rref)

    def key(self, flips, residual):
        return (self.detectors(flips),) + self.input_pauli(flips, residual)


# ------------------------------------------------------------------ F2 helpers on Python ints
def _rref(rows):
    basis = []
    for r in rows:
        for b in basis:
            if (r >> (b.bit_length() - 1)) & 1:
                r ^= b
        if r:
            hb = r.bit_length() - 1
            basis = [b ^ r if (b >> hb) & 1 else b for b in basis]
            basis.append(r)
    return sorted(basis, reverse=True)


def _reduce(v, rr):
    for b in rr:
        if (v >> (b.bit_length() - 1)) & 1:
            v ^= b
    return v


def _solve_combination(rows, target):
    """(tag, rest): tag is the bitmask of rows whose XOR is the component of target in their span, rest the
    remaining canonical component; both are linear functions of target"""
    basis = []
    for k, v in enumerate(rows):
        tag = 1 << k
        for b, bt in basis:
            if v & (1 << (b.bit_length() - 1)):
                v ^= b
                tag ^= bt
        if v:
            basis.append((v, tag))
            basis.sort(key=lambda x: -x[0])
    # fully reduce the basis so that the remainder is canonical
    for a in range(len(basis)):
        hb = basis[a][0].bit_length() - 1
        for c in range(len(basis)):
            if c != a and (basis[c][0] >> hb) & 1:
                basis[c] = (basis[c][0] ^ basis[a][0], basis[c][1] ^ basis[a][1])
    tag = 0
    for b, bt in basis:
        if target & (1 << (b.bit_length() - 1)):
            target ^= b
            tag ^= bt
    return tag, target


def _kernel_combinations(rows, m):
    """basis (as bitmasks over rows) of combinations of rows with XOR zero"""
    basis, kern = [], []
    for k, v in enumerate(rows):
        tag = 1 << k
        for b, bt in basis:
            if v & (1 << (b.bit_length() - 1)):
                v ^= b
                tag ^= bt
        if v:
            basis.append((v, tag))
            basis.sort(key=lambda x: -x[0])
        else:
            kern.append(tag)
    return kern


def _right_inverse(A):
    """X with A X = I over F2, for A (m x n) of full row rank"""
    m, n = A.shape
    Ar = A.copy() % 2
    E = np.eye(m, dtype=np.uint8)
    piv = []
    r = 0
    for c in range(n):
        rows = [i for i in range(r, m) if Ar[i, c]]
        if not rows:
            continue
        i0 = rows[0]
        Ar[[r, i0]] = Ar[[i0, r]]
        E[[r, i0]] = E[[i0, r]]
        for i in range(m):
            if i != r and Ar[i, c]:
                Ar[i] ^= Ar[r]
                E[i] ^= E[r]
        piv.append(c)
        r += 1
        if r == m:
            break
    assert r == m, "not full row rank"
    # E A = Ar with Ar[:, piv] = I, so A[:, piv] = E^-1 and X = (E on the pivot rows) is a right inverse
    X = np.zeros((n, m), dtype=np.uint8)
    for k, c in enumerate(piv):
        X[c, :] = E[k, :]
    assert np.array_equal((A.astype(np.int64) @ X) % 2, np.eye(m, dtype=np.int64))
    return X


# ------------------------------------------------------------------ analysis
class Analysis:
    """fault keys (D, E) for every elementary fault and harmlessness of key classes"""

    def __init__(self, ops, nq, out, ideal=()):
        self.ops, self.nq, self.out, self.ideal = ops, nq, list(out), set(ideal)
        self.sk = Skeleton(ops, nq, out)
        self.ref = out_vec(simulate(ops, nq), nq, self.out)
        assert np.linalg.norm(self.ref) > 1e-9, "fault-free run rejected"
        self.locs = locations(ops, self.ideal)
        self.faults = []        # (loc index, op index, fault, key)
        for li, (i, fs) in enumerate(self.locs):
            for f in fs:
                fl, res = self.sk.fault_effect(i, f)
                self.faults.append((li, i, f, self.sk.key(fl, res)))
        self.cache = {}

    def output_of(self, U, E):
        """output for D = 0: the circuit with the T inputs replaced by E|T>^t, followed by the Pauli U"""
        t, nq = self.sk.t, self.nq
        texp, zafter = {}, set()
        for j in range(t):
            if (E >> j) & 1:
                texp[j] = self.ops[self.sk.tgates[j]][2] - 2
            if (E >> (t + j)) & 1:
                zafter.add(j)
        psi = simulate(self.ops, nq, texp=texp, zafter=zafter)
        ps = "".join("IXZY"[((U >> q) & 1) + 2 * ((U >> (nq + q)) & 1)] for q in range(nq))
        psi = _apply_paulis(psi, ps, list(range(nq)))
        return out_vec(psi, nq, self.out)

    def harmless_UE(self, U, E):
        if (U, E) not in self.cache:
            self.cache[(U, E)] = parallel(self.output_of(U, E), self.ref)
        return self.cache[(U, E)]

    def harmless_key(self, key):
        D, U, E = key
        return D != 0 or self.harmless_UE(U, E)

    def harmless_direct(self, F):
        return parallel(out_vec(simulate(self.ops, self.nq, F), self.nq, self.out), self.ref)

    def classify(self, key):
        """'detected', 'trivial' (E in S_C), 'absorbed' (collapse with nonzero projection), 'rejected'
        (collapse to zero), or 'harmful'"""
        D, U, E = key
        if D:
            return "detected"
        if E == 0 and U == 0:
            return "trivial"
        v = self.output_of(U, E)
        if np.linalg.norm(v) < 1e-9:
            return "rejected"
        return "absorbed" if parallel(v, self.ref) else "harmful"

    def weight_classes(self, w):
        """distinct keys of fault configurations of weight w (faults at w distinct locations), with counts"""
        by_loc = {}
        for li, i, f, k in self.faults:
            by_loc.setdefault(li, []).append(k)
        locs = sorted(by_loc)
        counts = {}
        for combo in itertools.combinations(locs, w):
            for ks in itertools.product(*(by_loc[l] for l in combo)):
                D = U = E = 0
                for d, u, e in ks:
                    D ^= d
                    U ^= u
                    E ^= e
                counts[(D, U, E)] = counts.get((D, U, E), 0) + 1
        return counts

    def harmful_count(self, w):
        cnt = self.weight_classes(w)
        return sum(c for k, c in cnt.items() if not self.harmless_key(k)), sum(cnt.values())
