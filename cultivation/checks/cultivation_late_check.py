"""State-vector check of the flat-phase lemma and of the exact criterion for late errors (paper/cultivation.tex).

Flat phases. For a CSS code with dim S_Z = (n-1)/2 and a layer U of T and T^dag gates (angles theta_q = +-pi/4),
let phi(w) = prod_{q in w} (-i sgn theta_q). The operator U X^n U^dag maps the code to itself iff phi is constant on
S_Z and on l_Z + S_Z; then phi = 1 on S_Z, and U X^n U^dag acts on the code as (I + c Z_L) X_L / sqrt(2) with
c = phi(l_Z) = +-i.

Late errors. Let X^A act just before the last layer U, X^A' and Z^B after it, on the encoded target tau (the +1
eigenstate of the check), followed by the ideal decoder. With K_A the Z codewords (elements of ker H_X) inside A
and K0_A those that are Z stabilizers, the decoder accepts iff A + A' is in S_X or in l_X + S_X and there is
w <= A with B + w in ker H_X and w orthogonal to K0_A. It then accepts with probability 2^-|A| |K_A| |K0_A| and
outputs Z_L^lambda X_L^mu tau if K_A has no logical (mu = [A + A' in l_X + S_X], lambda = logical class of B + w),
and Z_L^lambda (I + c' Z_L) X_L^mu tau / sqrt(2) with c' = c (-1)^{|(w + A') cap l_A|} if K_A contains a Z logical l_A.
Checked on the [[7,1,3]] and [[19,1,5]] colour codes for every valid T/T^dag pattern tried, on random and on
structured configurations (face hooks, logical supports). Also checks the check-filtering identity and the
decomposition behind the dominance of late errors. Prints ALL OK.
Usage: python cultivation_late_check.py [seed]"""
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
        self.SX = span(self.F)                       # self-dual: S_X and S_Z have the same supports
        self.all = (1 << n) - 1
        self.LX = [self.all ^ s for s in self.SX]
        self.kerHX = self.SX + self.LX                # Z codewords: Z stabilizers and Z logicals
        self.idx = np.arange(1 << n, dtype=np.int64)
        self.par = np.array([pc(i) & 1 for i in range(1 << n)], dtype=np.int8) if n <= 20 else None

    def zsign(self, mask):
        return 1 - 2 * self.par[self.idx & mask]

    def project(self, psi):
        for f in self.F:
            psi = (psi + psi[self.idx ^ f]) / 2          # X_f
            psi = (psi + self.zsign(f) * psi) / 2        # Z_f
        return psi

    def basis(self):
        z = np.zeros(1 << self.n, complex); z[0] = 1
        b0 = self.project(z); b0 /= np.linalg.norm(b0)
        b1 = b0[self.idx ^ self.all]
        return b0, b1


def phases(code, theta):
    ph = np.ones(1 << code.n, complex)
    for q, t in enumerate(theta):
        ph *= np.where((code.idx >> q) & 1, np.exp(1j * t), 1)
    return ph


def phi(w, sgn):
    out = 1
    for q in range(len(sgn)):
        if (w >> q) & 1:
            out *= -1j * sgn[q]
    return out


def check_pattern(code, sgn):
    """returns (is_logical, flat, c, tau as a 2-vector, H on the code)"""
    theta = [np.pi / 4 * s for s in sgn]
    ph = phases(code, theta)
    b0, b1 = code.basis()
    H = np.zeros((2, 2), complex)
    for j, bj in enumerate((b0, b1)):
        v = ph * (np.conj(ph) * bj)[code.idx ^ code.all]     # U X^n U^dag |j_L>
        H[0, j], H[1, j] = np.vdot(b0, v), np.vdot(b1, v)
    logical = np.allclose(H.conj().T @ H, np.eye(2), atol=1e-9)
    vals = [phi(s, sgn) for s in code.SX]
    lvals = [phi(l, sgn) for l in code.LX]
    flat = np.allclose(vals, 1) and np.allclose(lvals, lvals[0])
    c = lvals[0]
    ev, evec = np.linalg.eig(H) if logical else (None, None)
    tau = evec[:, np.argmin(abs(ev - 1))] if logical else None
    return logical, flat, c, tau, H, ph, (b0, b1)


def predict(code, A, Ap, B, sgn, c, tau):
    """acceptance and decoded 2-vector (up to phase) from the criterion"""
    tot = A ^ Ap
    if tot in set(code.SX):
        mu = 0
    elif tot in set(code.LX):
        mu = 1
    else:
        return 0.0, None
    K = [k for k in code.kerHX if k & ~A == 0]
    K0 = [k for k in K if k in set(code.SX)]
    Klog = [k for k in K if k not in set(code.SX)]
    rows = []                               # GF(2) system in w <= A: H_X (B + w) = 0 and w . k = 0 for k in K0
    for f in code.F:
        rows.append((f & A, pc(f & B) & 1))
    for k in K0:
        rows.append((k, 0))
    cand = solve_gf2(rows)
    if cand is None:
        return 0.0, None
    w = cand
    lam = 0 if (B ^ w) in set(code.SX) else 1
    X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1])
    out = np.linalg.matrix_power(Z, lam) @ np.linalg.matrix_power(X, mu) @ tau
    acc = 2.0 ** (-pc(A)) * len(K) * len(K0)
    if Klog:
        lA = Klog[0]
        cp = c * (-1) ** pc((w ^ Ap) & lA)
        out = np.linalg.matrix_power(Z, lam) @ (np.eye(2) + cp * Z) @ np.linalg.matrix_power(X, mu) @ tau / np.sqrt(2)
    return acc, out


def solve_gf2(rows):
    """rows: (mask, rhs); returns one solution mask or None"""
    piv = []
    for m, r in rows:
        for pm, pr, b in piv:
            if (m >> b) & 1:
                m ^= pm; r ^= pr
        if m == 0:
            if r:
                return None
            continue
        b = m.bit_length() - 1
        piv = [(pm ^ m, pr ^ r, pb) if (pm >> b) & 1 else (pm, pr, pb) for pm, pr, pb in piv]
        piv.append((m, r, b))
    return sum(1 << b for m, r, b in piv if r)


def subsets_iter(mask):
    s = mask
    while True:
        yield s
        if s == 0:
            return
        s = (s - 1) & mask


def run_code(delta, ntests):
    global ALL_OK
    code = Code(delta)
    n = code.n
    pats = [[1] * n] if delta == 3 else []
    if delta == 5:
        pats.append([-1 if q == 8 else 1 for q in range(n)])
        pats.append([1 if q == 8 else -1 for q in range(n)])
    # random patterns: keep those whose check is logical, and record the flat-phase equivalence on all
    tried, agree = 0, 0
    for _ in range(40 if delta == 3 else 12):
        s = list(rng.choice([-1, 1], size=n))
        tried += 1
        logical, flat, *_ = check_pattern(code, s)
        agree += logical == flat
        if logical:
            pats.append(s)
    ok_eq = agree == tried
    print(f"delta={delta}: flat phases <=> logical check on {agree}/{tried} random patterns", flush=True)
    ALL_OK &= ok_eq
    nt, bad = 0, 0
    for sgn in pats:
        logical, flat, c, tau, H, ph, (b0, b1) = check_pattern(code, sgn)
        assert logical and flat and abs(abs(c) - 1) < 1e-9 and abs(c.real) < 1e-9
        tauL = tau[0] * b0 + tau[1] * b1
        pre = np.conj(ph) * tauL                                   # U^dag tau_L
        cases = []
        for _ in range(ntests):
            A = int(rng.integers(0, 1 << n))
            if rng.random() < 0.5:                                  # sparse A
                A = sum(1 << int(q) for q in rng.choice(n, size=int(rng.integers(1, delta + 2)), replace=False))
            tgt = int(rng.choice(code.SX + code.LX))
            Ap = A ^ tgt if rng.random() < 0.9 else int(rng.integers(0, 1 << n))
            B = int(rng.choice(code.kerHX))
            if rng.random() < 0.8:
                B ^= int(rng.choice(list(subsets_iter(A))[:4096]))
            cases.append((A, Ap, B))
        # structured: a face hook completed by Z faults; X on a whole logical support with Z inside it
        for f in code.F:
            for L in code.LX:
                if pc(L) == delta:
                    cases.append((f, 0, L & ~f))
                    cases.append((f, 0, L))
        for L in code.LX[:50]:
            if pc(L) == delta:
                for y in list(subsets_iter(L))[:6]:
                    cases.append((L, 0, y))
        for A, Ap, B in cases:
            psi = pre[code.idx ^ A]                                # X^A before the layer
            psi = ph * psi                                          # the layer U
            psi = psi[code.idx ^ Ap] * code.zsign(B)               # X^A' and Z^B after it
            a = np.array([np.vdot(b0, psi), np.vdot(b1, psi)])
            acc = float(np.vdot(a, a).real)
            pacc, pout = predict(code, A, Ap, B, sgn, c, tau)
            ok = abs(acc - pacc) < 1e-9
            if ok and acc > 1e-12:
                fid_true = abs(np.vdot(tau, a)) ** 2 / acc
                fid_pred = abs(np.vdot(tau, pout)) ** 2 / max(np.vdot(pout, pout).real, 1e-30)
                par = abs(abs(np.vdot(pout, a)) ** 2 - np.vdot(pout, pout).real * acc) < 1e-9
                ok = abs(fid_true - fid_pred) < 1e-9 and par
            nt += 1
            if not ok:
                bad += 1
                if bad < 5:
                    print("MISMATCH", delta, sgn, A, Ap, B, acc, pacc)
    print(f"delta={delta}: late criterion agrees on {nt - bad}/{nt} configurations over {len(pats)} patterns",
          flush=True)
    ALL_OK &= bad == 0


def run_filter(delta, ntests):
    """check filtering: U (I + X^n)/2 U^dag Z^b X^a |tau_L> = 1/2 (Z^b X^a + (-1)^|b| phi(a) Z^(b+a) X^a) |tau_L>"""
    global ALL_OK
    code = Code(delta)
    n = code.n
    sgn = [1] * n if delta == 3 else [-1 if q == 8 else 1 for q in range(n)]
    logical, flat, c, tau, H, ph, (b0, b1) = check_pattern(code, sgn)
    tauL = tau[0] * b0 + tau[1] * b1
    worst = 0.0
    for _ in range(ntests):
        a = int(rng.integers(1, 1 << n)); bz = int(rng.integers(0, 1 << n))
        psi = tauL[code.idx ^ a] * code.zsign(bz)                 # Z^b X^a |tau_L>
        v = np.conj(ph) * psi                                       # U^dag
        v = (v + v[code.idx ^ code.all]) / 2                        # (I + X^n)/2, outcome +1 on the fault-free state
        v = ph * v                                                  # U
        w = tauL[code.idx ^ a]
        pred = 0.5 * (code.zsign(bz) * w + (-1) ** pc(bz) * phi(a, sgn) * code.zsign(bz ^ a) * w)
        worst = max(worst, float(np.abs(v - pred).max()))
    ok = worst < 1e-9
    print(f"delta={delta}: check filtering identity on {ntests} random Z^b X^a, max deviation {worst:.1e}", flush=True)
    ALL_OK &= ok


def run_dominance(delta, ntests):
    """late dominance: Post U R Pi U^dag Z^b X^a |tau_L> = sum_{v <= a, |v| = |b| mod 2} cbar_v Post U R Z^(b+v) X^a U^dag
    |tau_L>, with cbar_v = 2^(-|a|/2) conj(phi(v)): an earlier error splits into late insertions on its own qubits"""
    global ALL_OK
    code = Code(delta)
    n = code.n
    sgn = [1] * n if delta == 3 else [-1 if q == 8 else 1 for q in range(n)]
    logical, flat, c, tau, H, ph, (b0, b1) = check_pattern(code, sgn)
    tauL = tau[0] * b0 + tau[1] * b1
    phi_dag = np.conj(ph) * tauL                                    # U^dag |tau_L>, a +1 eigenvector of X^n
    worst, nharm, nfound = 0.0, 0, 0

    def tail(v, A, cz, Ap, Bp):                                     # Post U R applied to v
        v = v[code.idx ^ A] * code.zsign(cz)                        # R = X^A Z^c inside the window
        v = ph * v                                                  # U
        return v[code.idx ^ Ap] * code.zsign(Bp)                    # Post = X^A' Z^B'

    def decode(v):
        return np.array([np.vdot(b0, v), np.vdot(b1, v)])

    def harmful(a2):
        acc = np.vdot(a2, a2).real
        return acc > 1e-12 and abs(np.vdot(tau, a2)) ** 2 / acc < 1 - 1e-9

    for _ in range(ntests):
        a = sum(1 << int(q) for q in rng.choice(n, size=int(rng.integers(1, 4)), replace=False))
        bz = sum(1 << int(q) for q in rng.choice(n, size=int(rng.integers(0, 3)), replace=False))
        A = sum(1 << int(q) for q in rng.choice(n, size=int(rng.integers(0, 4)), replace=False))
        cz = int(rng.integers(0, 1 << n)) if rng.random() < 0.3 else 0
        Ap = A ^ a ^ int(rng.choice(code.SX + code.LX)) if rng.random() < 0.7 else 0
        Bp = int(rng.choice(code.kerHX)) ^ (int(rng.integers(0, 1 << n)) & (A | a))
        early = np.conj(ph) * (tauL[code.idx ^ a] * code.zsign(bz))  # U^dag Z^b X^a |tau_L>
        early = (early + early[code.idx ^ code.all]) / 2            # Pi
        F = tail(early, A, cz, Ap, Bp)
        terms, S = [], np.zeros_like(F)
        sub = a
        while True:
            vv = sub
            if pc(vv) % 2 == pc(bz) % 2:
                cb = 2.0 ** (-pc(a) / 2) * np.conj(phi(vv, sgn))
                L = tail(phi_dag[code.idx ^ a] * code.zsign(bz ^ vv), A, cz, Ap, Bp)
                S += cb * L
                terms.append(decode(L))
            if sub == 0:
                break
            sub = (sub - 1) & a
        worst = max(worst, float(np.abs(F - S).max()))
        if harmful(decode(F)):
            nharm += 1
            nfound += any(harmful(t) for t in terms)
    ok = worst < 1e-9 and nfound == nharm
    print(f"delta={delta}: late dominance identity on {ntests} configurations, max deviation {worst:.1e}; "
          f"{nharm} harmful, each with a harmful late term: {nfound == nharm}", flush=True)
    ALL_OK &= ok


run_code(3, 300)
run_code(5, 60)
run_filter(3, 300)
run_filter(5, 100)
run_dominance(3, 400)
run_dominance(5, 150)
print("ALL OK" if ALL_OK else "FAILED")
