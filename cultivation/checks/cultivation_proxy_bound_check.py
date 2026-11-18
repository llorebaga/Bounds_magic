"""State-vector check that the S proxy never understates the effect of a late error (paper/cultivation.tex).

Setting of the late-error theorem: an ideally encoded target on a triangular colour code, a last check with a layer U
of T and T^dag gates whose check is logical, X^A just before U, X^A' and Z^B just after it, then the ideal decoder.
The S proxy uses the same pattern with S and S^dag (angles +-pi/2) and its own target, the +1 eigenstate of its check.

Claims tested:
 (1) every configuration (A, A', B) that is harmful with S gates is harmful with T gates;
 (2) consequently, for every late pattern (A, B0) the least number of single-qubit Pauli faults after U that make it
     harmful is no larger with T gates than with S gates (brute force over all such faults at delta = 3);
 (3) the converse fails, e.g. a hook on two qubits of a minimum-weight logical support (coherent completion).
Prints ALL OK.
Usage: python cultivation_proxy_bound_check.py [seed]"""
import sys, os, itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluator", "src"))
from colorcode import color_code_faces

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rng = np.random.default_rng(seed)
pc = lambda m: bin(m).count("1")
ALL_OK = True


def span(gens):
    s = [0]
    for g in gens:
        s += [x ^ g for x in s]
    return s


class Code:
    def __init__(self, delta):
        faces, n = color_code_faces(delta)
        self.n, self.delta = n, delta
        self.F = [sum(1 << q for q in f) for f in faces]
        self.SX = span(self.F)
        self.all = (1 << n) - 1
        self.LX = [self.all ^ s for s in self.SX]
        self.idx = np.arange(1 << n, dtype=np.int64)
        self.par = np.array([pc(i) & 1 for i in range(1 << n)], dtype=np.int8)

    def zsign(self, mask):
        return 1 - 2 * self.par[self.idx & mask]

    def project(self, psi):
        for f in self.F:
            psi = (psi + psi[self.idx ^ f]) / 2
            psi = (psi + self.zsign(f) * psi) / 2
        return psi

    def basis(self):
        z = np.zeros(1 << self.n, complex); z[0] = 1
        b0 = self.project(z); b0 /= np.linalg.norm(b0)
        return b0, b0[self.idx ^ self.all]


class Layer:
    """a check with angles unit * sgn_q on the data; target = +1 eigenvector of U X^n U^dag on the code"""
    def __init__(self, code, sgn, unit):
        self.code = code
        self.ph = np.ones(1 << code.n, complex)
        for q, s in enumerate(sgn):
            self.ph *= np.where((code.idx >> q) & 1, np.exp(1j * unit * s), 1)
        b0, b1 = code.basis()
        self.b = (b0, b1)
        H = np.zeros((2, 2), complex)
        for j, bj in enumerate((b0, b1)):
            v = self.ph * (np.conj(self.ph) * bj)[code.idx ^ code.all]
            H[0, j], H[1, j] = np.vdot(b0, v), np.vdot(b1, v)
        self.logical = np.allclose(H.conj().T @ H, np.eye(2), atol=1e-9)
        if self.logical:
            ev, evec = np.linalg.eigh((H + H.conj().T) / 2)
            self.tau = evec[:, np.argmax(ev)]
            assert abs(ev.max() - 1) < 1e-9
            tauL = self.tau[0] * b0 + self.tau[1] * b1
            self.pre = np.conj(self.ph) * tauL          # U^dag tau_L, the fault-free state inside the check

    def run(self, A, Ap, B):
        c = self.code
        psi = self.ph * self.pre[c.idx ^ A]
        psi = psi[c.idx ^ Ap] * c.zsign(B)
        a = np.array([np.vdot(self.b[0], psi), np.vdot(self.b[1], psi)])
        acc = float(np.vdot(a, a).real)
        if acc < 1e-12:
            return False
        return abs(np.vdot(self.tau, a)) ** 2 / acc < 1 - 1e-9


def after_faults(n, wmax):
    """all products of single-qubit Pauli faults on at most wmax data qubits: (weight, X mask, Z mask)"""
    out = []
    for w in range(wmax + 1):
        for qs in itertools.combinations(range(n), w):
            for ps in itertools.product((1, 2, 3), repeat=w):      # 1 = X, 2 = Z, 3 = Y
                x = sum(1 << q for q, p in zip(qs, ps) if p & 1)
                z = sum(1 << q for q, p in zip(qs, ps) if p & 2)
                out.append((w, x, z))
    return out


def run(delta, nrand, brute):
    global ALL_OK
    code = Code(delta)
    n = code.n
    pats = [[-1 if q == 8 else 1 for q in range(n)], [1 if q == 8 else -1 for q in range(n)]] if delta == 5 else []
    for _ in range(200 if delta == 3 else 0):
        s = list(rng.choice([-1, 1], size=n))
        T = Layer(code, s, np.pi / 4)
        if T.logical:
            pats.append(s)
        if len(pats) >= (4 if delta == 3 else 2):
            break
    viol, nS, nT, total = 0, 0, 0, 0
    for s in pats:
        T, S = Layer(code, s, np.pi / 4), Layer(code, s, np.pi / 2)
        assert S.logical
        for _ in range(nrand):
            A = sum(1 << int(q) for q in rng.choice(n, size=int(rng.integers(1, delta + 2)), replace=False))
            tgt = int(rng.choice(code.SX + code.LX))
            Ap = A ^ tgt if rng.random() < 0.8 else int(rng.integers(0, 1 << n))
            r = rng.random()
            if r < 0.5:                       # Z part that the S proxy accepts: B = A + a Z codeword
                B = A ^ int(rng.choice(code.SX + code.LX))
            else:
                B = int(rng.choice(code.SX + code.LX)) ^ (int(rng.integers(0, 1 << n)) & A if r < 0.85 else 0)
            hT, hS = T.run(A, Ap, B), S.run(A, Ap, B)
            total += 1; nS += hS; nT += hT
            viol += hS and not hT
    print(f"delta={delta}: {total} random late configurations over {len(pats)} patterns: harmful with S {nS}, "
          f"with T {nT}, harmful with S but not with T {viol}", flush=True)
    ALL_OK &= viol == 0 and nS > 0
    if not brute:
        return
    s = pats[0]
    T, S = Layer(code, s, np.pi / 4), Layer(code, s, np.pi / 2)
    faults = after_faults(n, delta)
    worse, strictly, cases = 0, 0, 0
    for A in range(1, 1 << n):
        for B0 in (0, int(rng.integers(0, 1 << n)) & A):
            cT = next((w for w, x, z in faults if T.run(A, x, B0 ^ z)), delta + 1)
            cS = next((w for w, x, z in faults if S.run(A, x, B0 ^ z)), delta + 1)
            cases += 1
            worse += cT > cS
            strictly += cT < cS
    print(f"delta={delta}: completion weights for all {cases} late patterns (A, B0): T > S in {worse}, "
          f"T < S in {strictly}", flush=True)
    ALL_OK &= worse == 0 and strictly > 0
    # coherent completion: a hook on two qubits of a minimum-weight logical support
    L = next(l for l in code.LX if pc(l) == delta)
    u = sum(1 << q for q in [q for q in range(n) if (L >> q) & 1][:2])
    hT, hS = T.run(u, L ^ u, 0), S.run(u, L ^ u, 0)
    print(f"delta={delta}: hook on two qubits of a logical support, completed by X: harmful with T {hT}, with S {hS}",
          flush=True)
    ALL_OK &= hT and not hS


run(3, 3000, True)
run(5, 1500, False)
print("ALL OK" if ALL_OK else "FAILED")
