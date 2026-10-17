"""Worked example of App. "A worked example" (exact, sympy).

Prints every number quoted in the appendix:
  Part 1  three T states -> CS: code Z1Z2Z3=+1, all projected amplitudes, Gram entries, error polynomial.
  Part 2  one candidate [[7,3]] code from the n=7, d=2 search and the exact minors that reject it.
  Part 3  the eight-input cube factory: Gram entries of the tolerated patterns and the harmful count.
Gram entries use  <phi_e| i^k X^x Z^z |phi_f> = [k' subset x] 2^{-|x|/2} i^{k + 2|e&x| - |k'|},  k' = z^e^f,
and are cross-checked against dense linear algebra.
Usage (from scripts/): python worked_example.py
"""
import itertools
import numpy as np
import sympy as sp

I, R2 = sp.I, sp.sqrt(2)
w = sp.exp(sp.I * sp.pi / 4)
pc = lambda v: bin(v).count("1")

# ---------- Pauli strings -> (k, x, z) meaning i^k X^x Z^z, qubit j = bit j (leftmost letter = qubit 0) ----------
def parse(s):
    sign = -1 if s[0] == "-" else 1
    s = s.lstrip("+-")
    k, x, z = (0 if sign > 0 else 2), 0, 0
    for j, c in enumerate(s):
        if c in "XY": x |= 1 << j
        if c in "ZY": z |= 1 << j
        if c == "Y": k += 1           # Y = i X Z
    return (k % 4, x, z)

def mul(P, Q):
    (k1, x1, z1), (k2, x2, z2) = P, Q
    return ((k1 + k2 + 2 * pc(z1 & x2)) % 4, x1 ^ x2, z1 ^ z2)

def group(gens):
    els = [(0, 0, 0)]
    for g in gens:
        els += [mul(e, g) for e in els]
    return els

def matel(e, f, P):
    k, x, z = P
    kk = z ^ e ^ f
    if kk & ~x: return sp.Integer(0)
    return R2 ** (-pc(x)) * I ** ((k + 2 * pc(e & x) - pc(kk)) % 4)

def gram(gens, e, f):
    return sp.nsimplify(sp.expand(sum(matel(e, f, P) for P in group(gens)) / 2 ** len(gens)))

def minor(gens, e):
    g00, g0e, gee = gram(gens, 0, 0), gram(gens, 0, e), gram(gens, e, e)
    return sp.simplify(g00 * gee - g0e * sp.conjugate(g0e)), (g00, g0e, gee)

# ---------- dense cross-check ----------
def dense_proj(gens, n):
    D = 1 << n; idx = np.arange(D); M = np.eye(D, dtype=complex)
    for (k, x, z) in gens:
        P = np.zeros((D, D), complex)
        P[idx ^ x, idx] = (1j ** k) * (-1.0) ** np.array([pc(b & z) for b in idx])
        M = M @ (np.eye(D) + P) / 2
    return M
def dense_phi(e, n):
    v = np.array([1.0 + 0j])
    for j in reversed(range(n)):
        v = np.kron(v, np.array([1, -np.exp(1j*np.pi/4) if (e >> j) & 1 else np.exp(1j*np.pi/4)]) / np.sqrt(2))
    return v
def check_dense(gens, n, pairs):
    Pm = dense_proj(gens, n)
    for e, f in pairs:
        ex = complex(gram(gens, e, f)); de = np.vdot(dense_phi(e, n), Pm @ dense_phi(f, n))
        assert abs(ex - de) < 1e-12, (e, f, ex, de)

def bits(e, n): return "".join(str((e >> j) & 1) for j in range(n))

# =====================================================================================================
print("PART 1: three T states -> CS")
n = 3; gens = [parse("ZZZ")]
for x in range(8):
    if pc(x) % 2 == 0:
        print(f"  |{bits(x,3)}>  amplitude of Pi_C phi_000: w^{pc(x)}/(2 sqrt2)")
for e in [0b111, 0b001]:
    m, (g00, g0e, gee) = minor(gens, e)
    print(f"  e={bits(e,3)}: G00={g00}, G0e={g0e}, Gee={gee}, minor={m}")
check_dense(gens, 3, [(0, 0), (0, 7), (7, 7), (0, 1), (1, 1)])
p = sp.symbols("p")
harmless = [e for e in range(8) if minor(gens, e)[0] == 0]
acc = {e: gram(gens, e, e) for e in range(8)}
num = sum(acc[e] * p ** pc(e) * (1 - p) ** (3 - pc(e)) for e in range(8) if e not in harmless)
den = sum(acc[e] * p ** pc(e) * (1 - p) ** (3 - pc(e)) for e in range(8))
print("  harmless patterns:", [bits(e, 3) for e in harmless], " eps(p) =", sp.factor(sp.simplify(num / den)))

# =====================================================================================================
print("\nPART 2: a rejected candidate at n=7, d=2")
n = 7
cand = ["XXXXXXI", "IIIIIIZ", "ZIIIIZI", "IZZZZII"]   # first code of X-rank one surviving the pruning
gens = [parse(s) for s in cand]
H = [0] + [1 << j for j in range(n)] + [((1 << n) - 1) ^ (1 << j) for j in range(n)] + [(1 << n) - 1]
# no-hidden-logical check
Xs = [g[1] for g in gens]; Zgroup = {P[2] for P in group(gens) if P[1] == 0}
for e in H[1:]:
    comm = all(pc(x & e) % 2 == 0 for x in Xs)
    assert (not comm) or (e in Zgroup), ("hidden logical", bits(e, n))
print("  passes the no-hidden-logical test")
for signs in itertools.product([1, -1], repeat=len(cand)):
    g = [((k + (0 if s > 0 else 2)) % 4, x, z) for s, (k, x, z) in zip(signs, gens)]
    g00 = gram(g, 0, 0)
    if g00 == 0:
        print(f"  signs {signs}: G00 = 0 (fault-free input rejected)"); continue
    fails = []
    for e in H[1:]:
        m, vals = minor(g, e)
        if m != 0: fails.append((bits(e, n), m, vals))
    first = fails[0]
    print(f"  signs {signs}: G00={g00}; {len(fails)} of {len(H)-1} minors nonzero; e.g. e={first[0]}: "
          f"G0e={first[2][1]}, Gee={first[2][2]}, minor={first[1]}")
check_dense(gens, n, [(0, 0), (0, 1), (1, 1), (0, 127), (127, 127)])

# =====================================================================================================
print("\nPART 3: the eight-input cube factory")
n = 8
# vertices v=0..7 of the cube; X on even-weight vertices, Y on odd (T-dagger inputs written as S^dag T)
xy = "".join("X" if pc(v) % 2 == 0 else "Y" for v in range(8))
faces = ["".join("Z" if ((v >> i) & 1) == val else "I" for v in range(8)) for i, val in [(0, 0), (1, 0), (2, 0), (0, 1)]]
cube = [xy] + faces
print("  stabilizers:", cube)
gens = [parse(s) for s in cube]
g00 = gram(gens, 0, 0); print("  G00 =", g00)
for e in [1 << 0, 0xFF ^ 1, 0xFF]:
    print(f"  e={bits(e,8)}: G0e={gram(gens,0,e)}, Gee={gram(gens,e,e)}")
harm2 = sum(1 for a, b in itertools.combinations(range(8), 2) if minor(gens, (1 << a) | (1 << b))[0] != 0)
print("  harmful weight-2 patterns:", harm2, " (leading error 28 p^2 with acceptance G00/G00)")
check_dense(gens, 8, [(0, 0), (0, 1), (1, 1), (0, 255), (255, 255)])
print("\nall dense cross-checks passed")
