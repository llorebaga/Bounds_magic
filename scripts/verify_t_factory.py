"""Exact certificate for a single-T factory given as a signed [[n,1]] stabilizer code (kernel record format).
Checks, in exact arithmetic over Q(sqrt2, i) (sympy):
  1. G_00 > 0 and every single-fault minor G_00 G_ee - |G_0e|^2 vanishes  (suppression order >= 2);
  2. the encoded state Pi_C phi_0 is Clifford-equivalent to |T>: logical Bloch moduli {1/sqrt2, 1/sqrt2, 0};
  3. the exact error polynomial eps(p) = sum_e h_e pi_e / sum_e a_e pi_e with a_e = G_ee and
     h_e = G_ee - |G_0e|^2 / G_00 (the part of Pi_C phi_e orthogonal to the target), and its leading term.
Usage: python verify_t_factory.py records.json index n
"""
import sys, json, itertools
import sympy as sp
from common import nullspace, rref, in_span_rref

R2 = sp.sqrt(2)
def mul(P, Q):  # (i^k1 X^x1 Z^z1)(i^k2 X^x2 Z^z2)
    (k1, x1, z1), (k2, x2, z2) = P, Q
    return ((k1 + k2 + 2 * bin(z1 & x2).count("1")) % 4, x1 ^ x2, z1 ^ z2)
def group(gens):
    els = [(0, 0, 0)]
    for g in gens: els += [mul(e, g) for e in els]
    return els
def matel(e, f, P):  # <phi_e| i^k X^x Z^z |phi_f>
    k, x, z = P; kk = z ^ e ^ f
    if kk & ~x: return sp.Integer(0)
    return R2 ** (-bin(x).count("1")) * sp.I ** ((k + 2 * bin(e & x).count("1") - bin(kk).count("1")) % 4)

recs = json.load(open(sys.argv[1])); r = recs[int(sys.argv[2])]; n = int(sys.argv[3])
I = sp.I
pc = lambda v: bin(v).count("1")

def to_kxz(x, z):  # Hermitian canonical i^{|x&z|} X^x Z^z
    return (pc(x & z) % 4, x, z)

gens = []
for i, (ax, c) in enumerate(zip(r["AX"], r["c"])):
    k, x, z = to_kxz(ax, c)
    if (r["signs"] >> i) & 1: k = (k + 2) % 4
    gens.append((k, x, z))
for j, b in enumerate(r["BZ"]):
    k, x, z = to_kxz(0, b)
    if (r["signs"] >> (len(r["AX"]) + j)) & 1: k = (k + 2) % 4
    gens.append((k, x, z))
els = group(gens); m = len(gens)

def gram(e, f):
    return sp.nsimplify(sp.expand(sum(matel(e, f, P) for P in els) / 2 ** m))

def pstr(P):
    k, x, z = P
    return ("-" if k in (2, 3) and False else "") + "".join("IXZY"[((x >> j) & 1) + 2 * ((z >> j) & 1)] for j in range(n))

print("stabilizer generators (sign, Pauli):")
for (k, x, z) in gens:
    herm_k = pc(x & z) % 4
    print("  ", "+" if k == herm_k else "-", pstr((k, x, z)))

g00 = gram(0, 0); print("G00 =", g00)
assert g00 != 0
for j in range(n):
    e = 1 << j
    mnr = sp.simplify(g00 * gram(e, e) - gram(0, e) * sp.conjugate(gram(0, e)))
    assert mnr == 0, ("single fault", j, mnr)
print("all", n, "single-fault minors vanish exactly: suppression order >= 2")

# logical Paulis (commute with S, independent modulo S)
D = 1 << n
cons = [(z) | (x << n) for (k, x, z) in gens]
ns = nullspace(cons, 2 * n)
S_rows = rref([x | (z << n) for (k, x, z) in gens]); span = list(S_rows); L = []
for w in ns:
    if not in_span_rref(w, rref(span)): L.append(w); span.append(w)
    if len(L) == 2: break
L1 = to_kxz(L[0] & (D - 1), L[0] >> n); L2 = to_kxz(L[1] & (D - 1), L[1] >> n)
L3 = mul(L1, L2); L3 = to_kxz(L3[1], L3[2])
def expv(P):  # <phi0|Pi_C P Pi_C|phi0> / G00 = <phi0| Pi_C P |phi0>/G00 (P commutes with Pi_C)
    return sp.nsimplify(sp.expand(sum(matel(0, 0, mul(S, P)) for S in els) / 2 ** m / g00))
bloch = sorted([sp.simplify(sp.Abs(expv(P))) for P in (L1, L2, L3)], key=lambda t: float(t))
print("logical Bloch moduli:", bloch)
assert bloch[0] == 0 and sp.simplify(bloch[1] - 1 / sp.sqrt(2)) == 0 and sp.simplify(bloch[2] - 1 / sp.sqrt(2)) == 0
print("encoded state is Clifford-equivalent to |T>")

p = sp.symbols("p")
num = 0; den = 0; harm_by_w = {}
for e in range(D):
    a = gram(e, e)
    if a == 0: continue
    g0e = gram(0, e)
    h = sp.nsimplify(sp.simplify(a - g0e * sp.conjugate(g0e) / g00))
    w = pc(e)
    den += a * p ** w * (1 - p) ** (n - w)
    num += h * p ** w * (1 - p) ** (n - w)
    if h != 0: harm_by_w[w] = harm_by_w.get(w, 0) + 1
eps = sp.simplify(num / den)
print("harmful patterns by weight:", dict(sorted(harm_by_w.items())))
print("acceptance at p=0:", g00)
print("leading error: eps =", sp.nsimplify(sp.series(eps, p, 0, 3).removeO()))
