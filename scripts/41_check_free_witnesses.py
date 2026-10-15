"""Exact checks for the free-witness input bounds (Lemma 2, Theorem 3).

Arithmetic is in Q(zeta_8) with Fraction coefficients; no floating point.
Checks:
  1. The erasure witness I/2 and the correlated-pair witness (rho_00+rho_11)/2 are
     diagonal in the fault basis and equal explicit mixtures of stabilizer states.
  2. Fault vectors Z^e|T>^n are orthonormal (n<=4), so witnesses are pattern mixtures.
  3. W=(X+Y)/sqrt2 fixes |T>, negates Z|T>; the Y-basis GHZ state (|+i>^N-|-i>^N)/sqrt2
     has fault weights vanishing exactly at |e|=N/2 (N<=8), giving n>=2d+1 for all d.
  3b. The pair-witness weight set {0,2,...,2d} avoids d for every odd d (d<=101),
     and contains d for every even d.
  4. CCZ has no nonidentity signed Pauli stabilizer, so nu(CCZ)=3.
  5. The resulting CCZ and CCZ^k input bounds table.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

# Q(zeta_8): tuple (a0,a1,a2,a3) meaning sum a_j zeta^j, zeta^4=-1.
def c(*a): return tuple(F(x) for x in a) + (F(0),) * (4 - len(a))
ZERO, ONE, ZETA, I = c(0), c(1), c(0, 1), c(0, 0, 1)
def add(x, y): return tuple(a + b for a, b in zip(x, y))
def neg(x): return tuple(-a for a in x)
def scal(x, s): return tuple(a * s for a in x)
def mul(x, y):
    out = [F(0)] * 4
    for i, a in enumerate(x):
        if a:
            for j, b in enumerate(y):
                k = i + j
                if k < 4: out[k] += a * b
                else: out[k - 4] -= a * b
    return tuple(out)
def conj(x):  # zeta^j -> zeta^{-j} = -zeta^{4-j}
    return (x[0], -x[3], -x[2], -x[1])
def sigma(x):  # zeta -> -zeta, i.e. sqrt2 -> -sqrt2 and i fixed
    return (x[0], -x[1], x[2], -x[3])
def zpow(k):
    k %= 8
    return c(*([0] * k + [1])) if k < 4 else neg(c(*([0] * (k - 4) + [1])))

def kron(u, v): return [mul(a, b) for a in u for b in v]
def outer(u, v): return [[mul(a, conj(b)) for b in v] for a in u]
def madd(A, B): return [[add(a, b) for a, b in zip(r, s)] for r, s in zip(A, B)]
def mscal(A, s): return [[scal(a, s) for a in r] for r in A]
def inner(u, v): return sum_((mul(conj(a), b) for a, b in zip(u, v)))
def sum_(it):
    out = ZERO
    for x in it: out = add(out, x)
    return out

# |T> = (|0> + zeta|1>)/sqrt2; we carry unnormalized vectors and rational norms.
T = [ONE, ZETA]                     # norm^2 = 2
def fault_vec(e, n):                # Z^e |T>^{(x)n}, unnormalized (norm^2 = 2^n)
    v = [ONE]
    for j in range(n):
        bit = e >> (n - 1 - j) & 1
        v = kron(v, [ONE, ZETA if not bit else neg(ZETA)])
    return v

def check_orthonormal(nmax=4):
    for n in range(1, nmax + 1):
        for e in range(1 << n):
            for f in range(1 << n):
                g = inner(fault_vec(e, n), fault_vec(f, n))
                assert g == (c(2 ** n) if e == f else ZERO), (n, e, f)
    return True

def check_sigma_swaps_faults(nmax=4):
    # sigma maps the rank-one projector of Z^e|T>^n to that of Z^{ebar}|T>^n.
    for n in range(1, nmax + 1):
        full = (1 << n) - 1
        for e in range(1 << n):
            P = outer(fault_vec(e, n), fault_vec(e, n))
            Q = outer(fault_vec(e ^ full, n), fault_vec(e ^ full, n))
            assert [[sigma(a) for a in r] for r in P] == Q
    return True

def check_pair_witness():
    # (rho_00 + rho_11)/2 with rho_e = |Z^e TT><Z^e TT| / 4
    lhs = mscal(madd(outer(fault_vec(0, 2), fault_vec(0, 2)),
                     outer(fault_vec(3, 2), fault_vec(3, 2))), F(1, 8))
    a = [ONE, ZERO, ZERO, I]        # (|00> + i|11>)/sqrt2, norm^2 2
    b = [ZERO, ONE, ONE, ZERO]      # (|01> + |10>)/sqrt2,  norm^2 2
    rhs = mscal(madd(outer(a, a), outer(b, b)), F(1, 4))
    assert lhs == rhs
    # a and b are stabilizer states: stabilizers {Y X, X Y}-type checked directly
    X = [[ZERO, ONE], [ONE, ZERO]]; Y = [[ZERO, neg(I)], [I, ZERO]]; Zm = [[ONE, ZERO], [ZERO, neg(ONE)]]
    def kronm(A, B):
        return [[mul(A[i][j], B[k][l]) for j in range(2) for l in range(2)] for i in range(2) for k in range(2)]
    def apply(M, v): return [sum_(mul(M[i][j], v[j]) for j in range(4)) for i in range(4)]
    assert apply(kronm(Zm, Zm), a) == a and apply(kronm(X, Y), a) == a     # a: <ZZ, XY>
    assert apply(kronm(Zm, Zm), b) == [neg(t) for t in b] and apply(kronm(X, X), b) == b  # b: <-ZZ, XX>
    # diagonal in the fault basis with weights 1/2 on e=00,11 and 0 on 01,10
    weights = {}
    for e in range(4):
        v = fault_vec(e, 2)
        val = sum_(mul(conj(v[i]), sum_(mul(lhs[i][j], v[j]) for j in range(4))) for i in range(4))
        weights[format(e, '02b')] = val
    # rho_e has norm^2 4, so phi_e = <v|W|v>/4
    phi = {k: scal(v, F(1, 4)) for k, v in weights.items()}
    assert phi == {'00': c(F(1, 2)), '01': ZERO, '10': ZERO, '11': c(F(1, 2))}
    return {k: str(v[0]) for k, v in phi.items()}

def check_erasure_witness():
    # I/2 = (rho_0 + rho_1)/2 = (|0><0| + |1><1|)/2
    lhs = mscal(madd(outer(fault_vec(0, 1), fault_vec(0, 1)), outer(fault_vec(1, 1), fault_vec(1, 1))), F(1, 4))
    assert lhs == mscal([[ONE, ZERO], [ZERO, ONE]], F(1, 2))
    return True

def check_parity(dmax=101):
    for d in range(1, dmax + 1):
        wts = set(range(0, 2 * d + 1, 2))
        assert (d in wts) == (d % 2 == 0)
    return True

def ccz_nullity():
    amp = [ONE if (x & 7) != 7 else neg(ONE) for x in range(8)]   # unnormalized
    stabs = 0
    for xa, zb in product(range(8), repeat=2):
        # (X^a Z^b |psi>)_y = (-1)^{b.(y+a)} psi_{y+a}
        v = [scal(amp[y ^ xa], (-1) ** bin(zb & (y ^ xa)).count('1')) for y in range(8)]
        for ph in (ONE, neg(ONE), I, neg(I)):
            if [mul(ph, t) for t in v] == amp: stabs += 1
    assert stabs == 1  # identity only
    return 3

def check_W_twirl():
    # W = (X+Y)/sqrt2 = [[0, conj(zeta)], [zeta, 0]] fixes |T> and negates Z|T>.
    Wm = [[ZERO, conj(ZETA)], [ZETA, ZERO]]
    def ap(M, v): return [sum_(mul(M[i][j], v[j]) for j in range(2)) for i in range(2)]
    assert ap(Wm, fault_vec(0, 1)) == fault_vec(0, 1)
    assert ap(Wm, fault_vec(1, 1)) == [neg(t) for t in fault_vec(1, 1)]
    # W^2 = I, and W is Clifford: W X W = Y, W Z W = -Z
    X = [[ZERO, ONE], [ONE, ZERO]]; Y = [[ZERO, neg(I)], [I, ZERO]]; Zm = [[ONE, ZERO], [ZERO, neg(ONE)]]
    def mm(A, B): return [[sum_(mul(A[i][k], B[k][j]) for k in range(2)) for j in range(2)] for i in range(2)]
    assert mm(Wm, Wm) == [[ONE, ZERO], [ZERO, ONE]]
    assert mm(mm(Wm, X), Wm) == Y and mm(mm(Wm, Zm), Wm) == [[neg(a) for a in r] for r in Zm]
    return True

def ghz_y(N):
    # (|+i>^N - |-i>^N)/sqrt2 is proportional to sum_{|x| odd} i^{|x|} |x>
    return [zpow(2 * bin(x).count('1')) if bin(x).count('1') % 2 else ZERO for x in range(1 << N)]

def check_ghz_witness(Nmax=8):
    out = {}
    for N in range(2, Nmax + 1, 2):
        g = ghz_y(N)
        # stabilizer generators: Y_j Y_{j+1} (j=1..N-1) and -Z^{(x)N}
        def apply_pauli(xm, zm, ph, v):  # ph * X^xm Z^zm, with Y = i X Z
            return [mul(ph, scal(v[y ^ xm], (-1) ** bin(zm & (y ^ xm)).count('1'))) for y in range(1 << N)]
        for j in range(N - 1):
            m = (1 << (N - 1 - j)) | (1 << (N - 2 - j))   # Y_j Y_{j+1} = (iXZ)(iXZ) = -X^m Z^m
            assert apply_pauli(m, m, neg(ONE), g) == g
        assert apply_pauli(0, (1 << N) - 1, neg(ONE), g) == g
        # fault-basis overlaps: zero exactly at |e| = N/2, nonzero at e = 0
        zero_weights = set()
        for e in range(1 << N):
            ov = inner(fault_vec(e, N), g)
            w = bin(e).count('1')
            if ov == ZERO: zero_weights.add(w)
            else: assert w != N // 2
        assert zero_weights == {N // 2}
        assert inner(fault_vec(0, N), g) != ZERO
        out[f'N={N}'] = 'q_e=0 exactly for |e|=N/2; q_0>0'
    return out

def bound(d, nu): return max(2 * d + 1, d + nu)  # rational targets: GHZ witness + Lemma 2 with Beverland Thm 5.3

def main():
    rec = dict(
        fault_vectors_orthonormal_n_le_4=check_orthonormal(),
        sigma_swaps_complementary_faults_n_le_4=check_sigma_swaps_faults(),
        erasure_witness=check_erasure_witness(),
        pair_witness_fault_weights=check_pair_witness(),
        pair_witness_avoids_odd_d_up_to_101=check_parity(),
        W_twirl_fixes_T_negates_ZT=check_W_twirl(),
        ghz_y_witness=check_ghz_witness(),
        nu_CCZ=ccz_nullity(),
        ccz_bounds={f'k={k}': {f'd={d}': bound(d, 3 * k) for d in range(1, 7)} for k in (1, 2, 3, 4)},
        status='PASS')
    out = ROOT / 'results' / 'free_witness_checks.json'
    out.write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps(rec, indent=2))

if __name__ == '__main__':
    main()
