"""State-vector check of the final-stage proposition (paper/cultivation.tex), on the [[7,1,3]] colour code.

An ideal correcting decoder measures the stabilizers, applies a minimum-weight correction and unencodes, without
rejecting. Claims tested:
 (i)  Pauli errors on (delta+1)/2 qubits of a minimum-weight logical support, just before the decoder, are corrected
      to a logical error: the output is wrong.
 (ii) after a T layer, any superposition of Pauli errors on at most (delta-1)/2 qubits, here the coherent family of
      X and Z errors on random qubits passed through random T / T^dag layers, is corrected in every syndrome sector
      to the target. With the postselecting decoder the same errors are also harmless, but a weight-(delta+1)/2
      error before it is rejected, so its fault distance can reach delta.
Prints ALL OK.
Usage: python cultivation_final_stage_check.py [seed]"""
import sys, os, itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluator", "src"))
from colorcode import color_code_faces

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rng = np.random.default_rng(seed)
pc = lambda m: bin(m).count("1")
ALL_OK = True

delta = 3
faces, n = color_code_faces(delta)
F = [sum(1 << q for q in f) for f in faces]
idx = np.arange(1 << n, dtype=np.int64)
par = np.array([pc(i) & 1 for i in range(1 << n)], dtype=np.int8)
zsign = lambda m: 1 - 2 * par[idx & m]
ALLQ = (1 << n) - 1


def project(psi, sx, sz):
    """projection onto the syndrome sector (sx: signs of the X stabilizers, sz: of the Z stabilizers)"""
    for k, f in enumerate(F):
        psi = (psi + (-1) ** ((sx >> k) & 1) * psi[idx ^ f]) / 2
        psi = (psi + (-1) ** ((sz >> k) & 1) * zsign(f) * psi) / 2
    return psi


def syndrome(x, z):
    """(signs of X stabilizers, signs of Z stabilizers) for the Pauli X^x Z^z"""
    sx = sum(((pc(f & z) & 1) << k) for k, f in enumerate(F))
    sz = sum(((pc(f & x) & 1) << k) for k, f in enumerate(F))
    return sx, sz


def apply(psi, x, z):
    return psi[idx ^ x] * zsign(z)


# minimum-weight corrections for every syndrome
corr = {}
for w in range(n + 1):
    for qs in itertools.combinations(range(n), w):
        for ps in itertools.product((1, 2, 3), repeat=w):
            x = sum(1 << q for q, p in zip(qs, ps) if p & 1)
            z = sum(1 << q for q, p in zip(qs, ps) if p & 2)
            corr.setdefault(syndrome(x, z), (x, z))

z0 = np.zeros(1 << n, complex); z0[0] = 1
b0 = project(z0, 0, 0); b0 /= np.linalg.norm(b0)
b1 = b0[idx ^ ALLQ]
tau = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
tauL = tau[0] * b0 + tau[1] * b1


def correct_and_decode(psi):
    """returns the list of (sector weight, fidelity) of the correcting decoder"""
    out = []
    for (sx, sz), (cx, cz) in corr.items():
        v = project(psi, sx, sz)
        nv = np.vdot(v, v).real
        if nv < 1e-12:
            continue
        v = apply(v, cx, cz)
        a = np.array([np.vdot(b0, v), np.vdot(b1, v)])
        out.append((nv, abs(np.vdot(tau, a)) ** 2 / nv))
    return out


# (i) weight (delta+1)/2 on a minimum-weight logical support
logicals = [m for m in range(1 << n) if pc(m) == delta and all(pc(m & f) % 2 == 0 for f in F)]
bad_i = 0
for L in logicals:
    qs = [q for q in range(n) if (L >> q) & 1][: (delta + 1) // 2]
    for p in ("X", "Z"):
        m = sum(1 << q for q in qs)
        psi = apply(tauL, m if p == "X" else 0, m if p == "Z" else 0)
        res = correct_and_decode(psi)
        bad_i += all(f < 1 - 1e-9 for _, f in res)
print(f"(i) {(delta + 1) // 2} errors on a minimum-weight logical support: wrong output in {bad_i} of "
      f"{2 * len(logicals)} cases", flush=True)
ALL_OK &= bad_i == 2 * len(logicals)

# (ii) coherent errors on at most (delta-1)/2 qubits after T layers are corrected in every sector
t = (delta - 1) // 2
worst = 1.0
for _ in range(400):
    qs = rng.choice(n, size=t, replace=False)
    psi = tauL.copy()
    for _layer in range(int(rng.integers(1, 4))):
        x = sum(1 << int(q) for q in qs if rng.random() < 0.6)
        z = sum(1 << int(q) for q in qs if rng.random() < 0.4)
        psi = apply(psi, x, z)
        ph = np.ones(1 << n, complex)
        for q in range(n):
            ph *= np.where((idx >> q) & 1, np.exp(1j * np.pi / 4 * rng.choice([-1, 1])), 1)
        U = ph                       # a random layer of T and T^dag gates, then its inverse after more errors
        psi = U * psi
        psi = apply(psi, sum(1 << int(q) for q in qs if rng.random() < 0.5), 0)
        psi = np.conj(U) * psi
    for nv, f in correct_and_decode(psi):
        worst = min(worst, f)
print(f"(ii) superpositions of errors on {t} qubit(s) through T layers: worst sector fidelity {worst:.12f}", flush=True)
ALL_OK &= worst > 1 - 1e-9
print("ALL OK" if ALL_OK else "FAILED")
