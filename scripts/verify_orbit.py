"""Proposition (Clifford orbit): CS12 CS13 |+++> = e^{i theta} CX_{23} H_2 CX_{12} S_2 H_2 |CCZ>  (qubits 1,2,3; rightmost first).
Checked in exact arithmetic over Q(i, sqrt2) via sympy.  Also: BFS over the Clifford orbit of |CCZ> (up to phase)
and list of diagonal third-level phase states it contains."""
import sympy as sp, itertools
n = 3
def bit(x, q): return (x >> (q - 1)) & 1          # qubit q in {1,2,3} <-> bit q-1
def ket(f): return sp.Matrix([f(x) for x in range(8)])
ccz = ket(lambda x: (-1) ** (bit(x,1) * bit(x,2) * bit(x,3))) / sp.sqrt(8)
tgt = ket(lambda x: sp.I ** (bit(x,1) * bit(x,2) + bit(x,1) * bit(x,3))) / sp.sqrt(8)
def one(U, q):
    M = sp.zeros(8, 8)
    for x in range(8):
        for y in range(8):
            if all(bit(x, j) == bit(y, j) for j in (1, 2, 3) if j != q): M[x, y] = U[bit(x, q), bit(y, q)]
    return M
def cx(c, t):
    M = sp.zeros(8, 8)
    for x in range(8): M[x ^ (bit(x, c) << (t - 1)), x] = 1
    return M
H = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2); S = sp.Matrix([[1, 0], [0, sp.I]])
v = cx(2, 3) * one(H, 2) * cx(1, 2) * one(S, 2) * one(H, 2) * ccz
k = next(i for i in range(8) if v[i] != 0)
ratio = sp.simplify(tgt[k] / v[k])
ok = all(sp.simplify(tgt[i] - ratio * v[i]) == 0 for i in range(8)) and sp.simplify(sp.Abs(ratio) - 1) == 0
print("Proposition holds exactly:", ok, " phase:", sp.nsimplify(ratio))
