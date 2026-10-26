"""Proposition 7 (nine-input T factory): explicit logical frame, acceptance and error polynomials.

Checks that Xbar=-IIIIZIZXI, Ybar=-IIIXIXYII, Zbar=-IIIXZXXXI commute with the eight stabilizers and
satisfy Xbar Ybar = i Zbar on the code, that the projected ideal input has expectations (1/sqrt2, 1/sqrt2, 0),
and prints the harmful-pattern counts, 256 A_w, 256 H_w and A(p) = D(p)/256, eps(p) = N(p)/D(p).
Usage: python verify_t9_frame.py
"""
import numpy as np, itertools, sympy as sp
I = np.eye(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1]); Y = 1j * X @ Z
P = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
def op(s):
    sign = -1 if s[0] == '-' else 1
    s = s.lstrip('+-'); M = np.array([[1]])
    for c in s: M = np.kron(M, P[c])
    return sign * M
gens = ['-XZIIZIIIX', '+XIZIZIIXI', '+ZXZZIIXZI', '+ZZXIZXIIZ', '-YXYIXIIZZ', '-YYXXIIIII', '+IZIZZIZII', '+IIZZZZIII']
G = [op(g) for g in gens]
n = 9; D = 2 ** n
Pi = np.eye(D)
for g in G: Pi = Pi @ (np.eye(D) + g) / 2
w = np.exp(1j * np.pi / 4)
T = np.array([1, w]) / np.sqrt(2)
def phi(e):
    v = np.array([1])
    for b in e: v = np.kron(v, Z @ T if b else T)
    return v
v0 = Pi @ phi([0] * n); A0 = np.vdot(v0, v0).real
print('A0', A0, A0 * 128)
frame = {'X': '-IIIIZIZXI', 'Y': '-IIIXIXYII', 'Z': '-IIIXZXXXI'}
L = {k: op(s) for k, s in frame.items()}
for k, M in L.items():
    print(k, 'commutes with all gens:', all(np.allclose(M @ g, g @ M) for g in G),
          'expectation', np.round(np.vdot(v0, M @ v0) / A0, 6))
print('XY = iZ on code:', np.allclose(Pi @ L['X'] @ L['Y'] @ Pi, 1j * Pi @ L['Z'] @ Pi))
# acceptance and damage per weight, with exact output fidelity to the decoded T
t = v0 / np.sqrt(A0)
Aw = [0.0] * (n + 1); Hw = [0.0] * (n + 1); harm = [0] * (n + 1)
for e in itertools.product([0, 1], repeat=n):
    v = Pi @ phi(e); a = np.vdot(v, v).real; f = abs(np.vdot(t, v)) ** 2
    k = sum(e); Aw[k] += a; Hw[k] += a - f
    if a - f > 1e-12: harm[k] += 1
print('harm', harm); print('256A', np.round(np.array(Aw) * 256, 6)); print('256H', np.round(np.array(Hw) * 256, 6))
p = sp.symbols('p')
Ap = sum(sp.Rational(round(Aw[k] * 256)) * p ** k * (1 - p) ** (n - k) for k in range(n + 1))
Hp = sum(sp.Rational(round(Hw[k] * 256)) * p ** k * (1 - p) ** (n - k) for k in range(n + 1))
print('D(p)=', sp.expand(Ap)); print('N(p)=', sp.expand(Hp)); print('eps series', sp.series(Hp / Ap, p, 0, 4))
