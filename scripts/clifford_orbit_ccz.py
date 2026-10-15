"""Clifford orbit of |CCZ> (Proposition 3 and its discussion).

1. Breadth-first search over the 3-qubit Clifford group (generators H_q, S_q, CX_ab) acting on |CCZ>,
   states compared up to global phase.  Reports the orbit size (69,120).
2. Lists all diagonal third-level phase states w^{f(x)} (f = linear + 2*quadratic + 4*cubic, mod 8)
   in the orbit and summarizes their (T-parities, CS-parities, CCZ) patterns.
3. Finds a shortest circuit mapping |CCZ> to CS12 CS13|+++> (exact check: verify_orbit.py).
Usage: python clifford_orbit_ccz.py
"""
import itertools
from collections import deque
import numpy as np

n, D = 3, 8
b = lambda x, i: (x >> i) & 1
def state(f): return np.array([np.exp(1j * np.pi / 4 * (f(x) % 8)) for x in range(D)]) / np.sqrt(D)
I2 = np.eye(2); H = np.array([[1, 1], [1, -1]]) / np.sqrt(2); S = np.diag([1, 1j])
def one(U, q):
    M = np.array([[1]], complex)
    for j in reversed(range(n)): M = np.kron(M, U if j == q else I2)
    return M
def cnot(c, t):
    M = np.zeros((D, D))
    for x in range(D): M[x ^ (b(x, c) << t), x] = 1
    return M
gens = [(f'H{q}', one(H, q)) for q in range(n)] + [(f'S{q}', one(S, q)) for q in range(n)] + \
       [(f'CX{c}{t}', cnot(c, t)) for c in range(n) for t in range(n) if c != t]
def key(v):
    k = np.argmax(np.abs(v) > 1e-9); v = v * np.conj(v[k]) / abs(v[k])
    return tuple(np.round(v.real, 6)) + tuple(np.round(v.imag, 6))

ccz = state(lambda x: 4 * (b(x, 0) & b(x, 1) & b(x, 2)))
target = key(state(lambda x: 2 * (b(x, 0) & b(x, 1)) + 2 * (b(x, 0) & b(x, 2))))
seen = {key(ccz): []}; q = deque([ccz]); path_to_target = None
while q:
    v = q.popleft(); pv = seen[key(v)]
    for name, G in gens:
        w = G @ v; k = key(w)
        if k not in seen:
            seen[k] = pv + [name]; q.append(w)
            if k == target and path_to_target is None: path_to_target = seen[k]
print("orbit size (up to phase):", len(seen))
print("shortest circuit CCZ -> CS12CS13 (applied left to right, qubits 0,1,2):", path_to_target)

hits = []
for lin in itertools.product(range(8), repeat=3):
    for quad in itertools.product(range(4), repeat=3):
        for cub in (0, 1):
            f = lambda x: (lin[0] * b(x, 0) + lin[1] * b(x, 1) + lin[2] * b(x, 2)
                           + 2 * (quad[0] * b(x, 0) * b(x, 1) + quad[1] * b(x, 0) * b(x, 2) + quad[2] * b(x, 1) * b(x, 2))
                           + 4 * cub * b(x, 0) * b(x, 1) * b(x, 2))
            if key(state(f)) in seen: hits.append((lin, quad, cub))
print("diagonal third-level states in the orbit:", len(hits))
print("(T parities, CS parities, CCZ) patterns:",
      sorted(set((tuple(l % 2 for l in lin), tuple(c % 2 for c in quad), cub) for lin, quad, cub in hits)))
