"""Exact check of Proposition 11: six T states projected onto Z^6=+1, X^6=-1 give X^{(x)4}|CCCZ>.

Projecting |T>^6 onto the [[6,4,2]] code with Z^{(x)6}=+1 and X^{(x)6}=-1 succeeds with probability 1/4.
On the representatives x = (y, y1+...+y4, 0), y in F2^4, of the code basis |x> - |xbar>, the amplitudes are
proportional to (-1)^{e3(y)+e4(y)}, which equals X^{(x)4}|CCCZ> up to the Clifford phases (-1)^{e1+e2+1}.
All arithmetic is exact (sympy, amplitudes in Q(omega)).
"""
import itertools
import sympy as sp

w = sp.exp(sp.I * sp.pi / 4)
n = 6
xs = list(itertools.product((0, 1), repeat=n))
T6 = {x: w ** sum(x) / sp.Integer(8) for x in xs}            # |T>^6 = 2^{-3} sum_x w^{|x|} |x>


def proj(state):
    # (I + Z^6)/2 keeps even weight; (I - X^6)/2 maps |x> to (|x> - |xbar>)/2
    out = {}
    for x, a in state.items():
        if sum(x) % 2:
            continue
        xb = tuple(1 - b for b in x)
        out[x] = out.get(x, 0) + a / 2
        out[xb] = out.get(xb, 0) - a / 2
    return {x: sp.nsimplify(sp.expand(a)) for x, a in out.items()}


psi = proj(T6)
prob = sp.simplify(sum(sp.expand(a * sp.conjugate(a)) for a in psi.values()))
print("acceptance probability:", prob)
assert sp.simplify(prob - sp.Rational(1, 4)) == 0


def e(k, y):
    return sum(sp.prod([y[i] for i in c]) for c in itertools.combinations(range(4), k)) % 2


ys = list(itertools.product((0, 1), repeat=4))
amps = {}
for y in ys:
    x = tuple(y) + (sum(y) % 2, 0)
    amps[y] = sp.simplify(psi[x])
a0 = amps[(0, 0, 0, 0)]
assert a0 != 0
ok_moduli = all(sp.simplify(sp.Abs(amps[y]) - sp.Abs(a0)) == 0 for y in ys)
ok_signs = all(sp.simplify(amps[y] / a0 - (-1) ** (e(3, y) + e(4, y))) == 0 for y in ys)
print("equal moduli:", ok_moduli, "  amplitudes proportional to (-1)^{e3+e4}:", ok_signs)
assert ok_moduli and ok_signs

# (-1)^{e3+e4}(y) = (-1)^{e4(y+1)} * (-1)^{e1+e2+1}: X^{(x)4}|CCCZ> up to the Clifford phases (-1)^{e1+e2+1}
ok_cccz = all(((e(3, y) + e(4, y)) - (e(4, tuple(1 - b for b in y)) + e(1, y) + e(2, y) + 1)) % 2 == 0 for y in ys)
print("equals X^{(x)4}|CCCZ> up to Clifford phases (-1)^{e1+e2+1}:", ok_cccz)
assert ok_cccz
print("Proposition 11 construction verified")
