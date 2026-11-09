"""Exact checks for the low-nullity witness family (n >= d + nu + 1 for d >= 2).

Arithmetic is in Q(zeta_8) with Fraction coefficients; no floating point.
Construction on n = a + d qubits, block A = first a qubits, block B = last d:
    beta  = (1+zbar)^d |+>^d - (1-zbar)^d |->^d        (unnormalized |+-> = |0>+-|1>)
    psi   = |T>^a (x) beta + (Z|T>)^a (x) Z^d beta
Checks, for every (a, d) with d >= 2, a >= d+1, n <= NMAX:
  1. psi is fixed by Z^n and by X_j X_{j+1} on B, so nu(psi) <= n - d = a.
  2. <Z^e T^n | psi> = 0 for every fault pattern with d <= |e| <= n-d, and the
     fault-free amplitude is nonzero (so the twirled psi is a valid witness).
  3. <X_{A1} X_{B1}> equals the Pell ratio Q/P (d even) or P/(2Q) (d odd), where
     (1+sqrt2)^d = P + Q sqrt2.
  4. For d <= 200 that ratio is a non-dyadic rational exactly when d >= 2, so psi is
     not Clifford equivalent (with stabilizer padding) to any target whose Pauli
     expectations are dyadic, e.g. CCZ^k, CS, CCCZ.
  5. For d = 1 the same construction is dyadic (1/2), consistent with the attained
     first-order bounds.
  6. (Floating point) the qutrit X-basis GHZ witness vanishes exactly on balanced
     fault patterns at n = 3m, so qutrit protocols need n >= floor(3d/2)+1.
"""
from fractions import Fraction as F
from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8

def c(*a): return tuple(F(x) for x in a) + (F(0),) * (4 - len(a))
ZERO, ONE, ZETA = c(0), c(1), c(0, 1)
def add(x, y): return tuple(a + b for a, b in zip(x, y))
def neg(x): return tuple(-a for a in x)
def mul(x, y):
    out = [F(0)] * 4
    for i, a in enumerate(x):
        if a:
            for j, b in enumerate(y):
                if b:
                    k = i + j
                    if k < 4: out[k] += a * b
                    else: out[k - 4] -= a * b
    return tuple(out)
def conj(x): return (x[0], -x[3], -x[2], -x[1])
def power(x, k):
    out = ONE
    for _ in range(k): out = mul(out, x)
    return out
def kron(u, v): return [mul(a, b) for a in u for b in v]
def inner(u, v):
    out = ZERO
    for a, b in zip(u, v):
        if a != ZERO and b != ZERO: out = add(out, mul(conj(a), b))
    return out
def popcount(x): return bin(x).count("1")

def apply_x(v, mask): return [v[b ^ mask] for b in range(len(v))]
def apply_z(v, mask): return [neg(a) if popcount(b & mask) & 1 else a for b, a in enumerate(v)]

def fault_vec(e, n):  # Z^e |T>^n, unnormalized; qubit 0 is the most significant bit
    v = [ONE]
    for j in range(n):
        v = kron(v, [ONE, neg(ZETA) if e >> (n - 1 - j) & 1 else ZETA])
    return v

def witness(a, d):
    zbar = conj(ZETA)
    u, w = power(add(ONE, zbar), d), power(add(ONE, neg(zbar)), d)
    plus, minus = [ONE], [ONE]
    for _ in range(d):
        plus, minus = kron(plus, [ONE, ONE]), kron(minus, [ONE, neg(ONE)])
    beta = [add(mul(u, p), neg(mul(w, m))) for p, m in zip(plus, minus)]
    betap = apply_z(beta, (1 << d) - 1)
    TA, ZTA = fault_vec(0, a), fault_vec((1 << a) - 1, a)
    return [add(x, y) for x, y in zip(kron(TA, beta), kron(ZTA, betap))]

def pell(d):
    P, Q = 1, 0
    for _ in range(d): P, Q = P + 2 * Q, P + Q
    return P, Q

def pell_ratio(d):
    P, Q = pell(d)
    return F(Q, P) if d % 2 == 0 else F(P, 2 * Q)

def is_dyadic(f): return f.denominator & (f.denominator - 1) == 0

def check(a, d):
    n = a + d
    psi = witness(a, d)
    assert apply_z(psi, (1 << n) - 1) == psi, "Z^n"
    for j in range(d - 1):
        assert apply_x(psi, 0b11 << j) == psi, ("XX", j)
    zero_amp = None
    for e in range(1 << n):
        amp = inner(fault_vec(e, n), psi)
        if e == 0: zero_amp = amp
        if d <= popcount(e) <= n - d:
            assert amp == ZERO, ("band", e)
    assert zero_amp != ZERO
    norm = inner(psi, psi)
    xx = inner(psi, apply_x(psi, (1 << (n - 1)) | (1 << (d - 1))))  # X on A-qubit 0 and B-qubit 0
    # both are rational multiples of sqrt2 or rationals; compare ratio exactly
    r = pell_ratio(d)
    assert xx == tuple(r * t for t in norm), ("pell", a, d, xx, norm)
    return {"a": a, "d": d, "n": n, "nu_upper": a, "XX_expectation": str(r), "non_dyadic": not is_dyadic(r)}

def qutrit_ghz_check(mmax=3):
    """Numerical cross-check of the analytic qutrit identity (floating point).

    G = sum_j zeta3^j |x_j>^n with Z|x_j> = |x_{j+1}> is a qutrit stabilizer state.
    Its amplitude on a fault pattern with symbol counts (m,m,m) factors as
    (prod_a f(a))^m (1+zeta3+zeta3^2) = 0, and the fault-free amplitude is nonzero
    because |f(0)| > |f(1)|+|f(2)|. Hence n=3m is excluded at d=2m.
    """
    import cmath, itertools
    z9, z3 = cmath.exp(2j * cmath.pi / 9), cmath.exp(2j * cmath.pi / 3)
    T3 = [1 / 3 ** .5, z9 / 3 ** .5, z9 ** -1 / 3 ** .5]
    fb = [[T3[k] * z3 ** (a * k) for k in range(3)] for a in range(3)]      # Z^a |T3>
    xt = [[z3 ** (j * k) / 3 ** .5 for k in range(3)] for j in range(3)]    # Z xt_j = xt_{j+1}
    ov = [[sum(fb[a][k].conjugate() * xt[j][k] for k in range(3)) for j in range(3)] for a in range(3)]
    f = [abs(ov[0][j]) for j in range(3)]
    assert f[0] > f[1] + f[2]
    rows = []
    for m in range(1, mmax + 1):
        n = 3 * m
        zero_classes = set(); fault_free = None
        for e in itertools.product(range(3), repeat=n):
            amp = sum(z3 ** j * cmath.exp(sum(cmath.log(ov[a][j]) for a in e)) for j in range(3))
            counts = tuple(sorted((e.count(s) for s in range(3)), reverse=True))
            if e == (0,) * n: fault_free = abs(amp)
            if abs(amp) < 1e-12: zero_classes.add(counts)
        assert zero_classes == {(m, m, m)}, zero_classes
        assert fault_free > 1e-6
        rows.append({"n": n, "d_excluded": 2 * m, "zero_count_classes": [list(c) for c in zero_classes]})
    return rows

def main():
    rows = []
    for d in range(2, NMAX):
        for a in range(d + 1, NMAX - d + 1):
            rows.append(check(a, d)); print(rows[-1], flush=True)
    assert pell_ratio(1) == F(1, 2)
    for d in range(2, 201):
        assert not is_dyadic(pell_ratio(d)), d
    qrows = qutrit_ghz_check()
    print("qutrit GHZ:", qrows)
    report = {"family_checks": rows, "pell_ratio_non_dyadic_for_2<=d<=200": True,
              "d1_ratio": "1/2 (dyadic)", "qutrit_ghz_numerical": qrows}
    out = ROOT / "results" / "low_nullity_witness_checks.json"
    out.write_text(json.dumps(report, indent=2))
    print("all checks passed; wrote", out)

if __name__ == "__main__":
    main()
