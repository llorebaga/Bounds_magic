"""Positive control: the [[8,3,2]] cube code with transversal T/T-dagger is the known 8-input
quadratic CCZ factory. In the all-T input model, T-dagger inputs are S^dag T, so the code is
conjugated by S on odd vertices. Check: Pi_C phi_0 != 0, Gram over H (d=2, plus Galois
complements) is rank one, and the decoded state has the Pauli spectrum of CCZ (x) |0>^5."""
import sys, numpy as np
from common import *

n = 8
verts = list(range(8))  # vertex v = (v0,v1,v2) bits; qubit index = v
odd = [v for v in verts if popcount(v) % 2]
faces = []
for i in range(3):
    for val in (0, 1):
        faces.append(sum(1 << v for v in verts if ((v >> i) & 1) == val))
# stabilizers: X^8 and 4 independent Z faces (x_i=0 for i=0,1,2, and x_0=1)
stabs = [(1, herm(0xFF, 0))] + [(1, herm(0, f)) for f in (faces[0], faces[2], faces[4], faces[1])]
# conjugate by S on odd vertices: X -> Y on those qubits (S X S^dag = Y)
def conjS(P, mask):
    p, x, z = P
    # Y = i X Z ; X_j -> Y_j for j in mask&x : X^x Z^z -> X^x Z^(z ^ (x&mask)) times i^{|x&mask|} with reordering
    sel = x & mask
    # S X S^dag = Y = i X Z. For each selected qubit, X_j -> i X_j Z_j (Z_j placed right after X_j; it
    # commutes with other qubits' operators). Then X_j Z_j Z_j^{z_j} keeps order.
    return ((p + popcount(sel)) % 4, x, z ^ sel)
oddmask = sum(1 << v for v in odd)
stabs = [(s, conjS(P, oddmask)) for s, P in stabs]
Pi = code_projector(stabs, n)
print("rank", round(np.trace(Pi).real))
H = harmless_set(n, 2)
B = np.array([phi(e, n) for e in H]).T
G = B.conj().T @ Pi @ B
sv = np.linalg.svd(G, compute_uv=False)
print("G00", G[0, 0].real, "singular values", np.round(sv[:4], 8))
v = Pi @ phi(0, n); v /= np.linalg.norm(v)
# Pauli spectrum compare
D = 1 << n; idx = np.arange(D)
Hm = (-1.0) ** popc_arr(idx[:, None] & idx[None, :]); XB = idx[:, None] ^ idx[None, :]
spec = lambda u: np.sort(np.round(np.abs((np.conj(u[XB]) * u[None, :]) @ Hm).ravel(), 6))
c = np.array([(-1) ** (((i >> 2) & (i >> 1) & i) & 1) for i in range(8)]) / np.sqrt(8)
ref = c.astype(complex)
for _ in range(n - 3): ref = np.kron(ref, [1, 0])
print("decoded spectrum == CCZ:", np.allclose(spec(v), spec(ref)))
