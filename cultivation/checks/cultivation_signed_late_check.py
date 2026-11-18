"""State-vector check of the late-error theorem for Z stabilizers of either sign (paper/cultivation.tex).

On the [[7,1,3]] and [[19,1,5]] colour codes, the code space with Z-stabilizer signs s_f (random) is used. A layer U
of T and T^dag gates has flat phases when phi(f) = s_f on every face, phi(w) = prod_{q in w} (-i sgn theta_q); the
pattern is found by solving phi(f) = s_f over GF(2). Claims tested:
 (1) U X^n U^dag maps the signed code space to itself exactly when phi(f) = s_f on the faces;
 (2) for late configurations (X^A just before U, X^A' and Z^B after it, ideal decoder of the signed code), the
     criterion of the theorem, which does not depend on the signs (late_ilp.exact_harm), decides harm exactly.
Prints ALL OK.  Usage: python cultivation_signed_late_check.py [seed]"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluator", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluator", "scripts"))
from colorcode import color_code_faces
from gf2 import solve
from late_ilp import exact_harm

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rng = np.random.default_rng(seed)
pc = lambda m: bin(m).count("1")
ALL_OK = True


def run(delta, npat, ncfg):
    global ALL_OK
    faces, n = color_code_faces(delta)
    F = [sum(1 << q for q in f) for f in faces]
    idx = np.arange(1 << n, dtype=np.int64)
    par = np.array([pc(i) & 1 for i in range(1 << n)], dtype=np.int8)
    zsign = lambda m: 1 - 2 * par[idx & m]
    ALLQ = (1 << n) - 1

    def project(psi, s):
        for f, sf in zip(F, s):
            psi = (psi + psi[idx ^ f]) / 2
            psi = (psi + sf * zsign(f) * psi) / 2
        return psi

    eq_ok, n_eq, nt, bad = True, 0, 0, 0
    for _ in range(npat):
        s = [int(x) for x in rng.choice([-1, 1], size=len(F))]
        z = np.zeros(1 << n, complex); z[:] = 0
        # a basis of the signed code space: project random states
        b0 = project(rng.normal(size=1 << n) + 0j, s); b0 /= np.linalg.norm(b0)
        b1 = b0[idx ^ ALLQ]                                  # X^n is logical
        b1 = b1 - np.vdot(b0, b1) * b0; b1 /= np.linalg.norm(b1)
        # flat pattern: (-1)^{#T^dag on f} = s_f (-i)^{-|f|}, i.e. #T^dag on f = |f|/2 + [s_f = -1] mod 2
        A = np.array([[(f >> q) & 1 for q in range(n)] for f in F], dtype=np.uint8)
        rhs = np.array([((pc(f) // 2) + (sf < 0)) % 2 for f, sf in zip(F, s)], dtype=np.uint8)
        x = solve(A, rhs)
        if x is None:
            continue
        sgn = [(-1 if x[q] else 1) for q in range(n)]
        # also a random pattern, to test the equivalence (1) in both directions
        for pattern, expect in ((sgn, True), (list(rng.choice([-1, 1], size=n)), None)):
            ph = np.ones(1 << n, complex)
            for q, g in enumerate(pattern):
                ph *= np.where((idx >> q) & 1, np.exp(1j * np.pi / 4 * g), 1)
            H = np.zeros((2, 2), complex)
            for j, bj in enumerate((b0, b1)):
                v = ph * (np.conj(ph) * bj)[idx ^ ALLQ]
                H[0, j], H[1, j] = np.vdot(b0, v), np.vdot(b1, v)
            logical = np.allclose(H.conj().T @ H, np.eye(2), atol=1e-9)
            phi = lambda w: np.prod([(-1j * pattern[q]) for q in range(n) if (w >> q) & 1]) if w else 1
            flat = all(abs(phi(f) - sf) < 1e-9 for f, sf in zip(F, s))
            n_eq += 1
            eq_ok &= logical == flat
            if expect is None:
                continue
            assert logical
            ev, evec = np.linalg.eigh((H + H.conj().T) / 2)
            tau = evec[:, np.argmax(ev)]
            tauL = tau[0] * b0 + tau[1] * b1
            pre = np.conj(ph) * tauL
            for _ in range(ncfg):
                Am = sum(1 << int(q) for q in rng.choice(n, size=int(rng.integers(1, delta + 2)), replace=False))
                tgt = int(rng.choice([0, ALLQ]))
                Ap = (Am ^ tgt) if rng.random() < 0.85 else int(rng.integers(0, 1 << n))
                B = (int(rng.integers(0, 1 << n)) & Am) ^ (ALLQ if rng.random() < 0.5 else 0)
                for f in F:
                    if rng.random() < 0.3:
                        Ap ^= f if rng.random() < 0.5 else 0
                        B ^= f if rng.random() < 0.5 else 0
                psi = ph * pre[idx ^ Am]
                psi = psi[idx ^ Ap] * zsign(B)
                a = np.array([np.vdot(b0, psi), np.vdot(b1, psi)])
                acc = float(np.vdot(a, a).real)
                harm = acc > 1e-12 and abs(np.vdot(tau, a)) ** 2 / acc < 1 - 1e-9
                pred = exact_harm(Am, Ap, B, [[q for q in range(n) if (f >> q) & 1] for f in F], n)
                nt += 1
                bad += harm != pred
    print(f"delta={delta}: logical <=> phi(f) = s_f on {n_eq} patterns: {eq_ok}; late criterion with signed "
          f"stabilizers agrees on {nt - bad}/{nt} configurations", flush=True)
    ALL_OK &= eq_ok and bad == 0 and nt > 0


run(3, 30, 200)
run(5, 4, 120)
print("ALL OK" if ALL_OK else "FAILED")
