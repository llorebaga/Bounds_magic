"""Sec. S5: the six-cube-block quartic CCZ factory flattened into a single generalized triorthogonal matrix.

Each cube block (a, b, c) becomes the eight T columns ((u1 a + u2 b + u3 c), e_i), u in F_2^3, with one check row per block.
Checks the overlap parity conditions (any overlap of up to three rows that involves a check row is even; the logical
overlaps give CCZ) and counts undetected logical errors at weights 1-4 (expected 0, 0, 0, 864).
Usage: python verify_quartic_flat.py
"""
import itertools
import numpy as np

blocks = """01111110 00001111 10100111
11101000 10110111 11111001
10101111 11011110 10110110
11000001 11100111 10010111
00010000 10110001 11110110
11010111 00010001 01011001"""
B = [[np.array([int(c) for c in w]) for w in line.split()] for line in blocks.splitlines()]
k, s_out, nb = 3, 5, len(B)
cols = []
for i, (a, b, c) in enumerate(B):
    for u in itertools.product([0, 1], repeat=3):
        outer = (u[0] * a + u[1] * b + u[2] * c) % 2
        e = np.zeros(nb, dtype=int); e[i] = 1
        cols.append(np.concatenate([outer, e]))
G = np.array(cols).T  # rows: 3 logical, 5 outer checks, 6 block checks
N, n = G.shape
L = range(k); S = range(k, N)
print("matrix", G.shape, "rank", np.linalg.matrix_rank(G.astype(float)))
# rank over F2
def rank2(M):
    M = M.copy() % 2; r = 0
    for c in range(M.shape[1]):
        piv = [i for i in range(r, M.shape[0]) if M[i, c]]
        if not piv: continue
        M[[r, piv[0]]] = M[[piv[0], r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]: M[i] ^= M[r]
        r += 1
    return r
print("F2 rank of rows", rank2(G), "check rows rank", rank2(G[k:]))
# generalized triorthogonality: overlaps involving a check row even;
# logical-only: singles and pairs even (no T, no CS), triple 123 odd (CCZ)
bad = []
for R in itertools.chain(*(itertools.combinations(range(N), t) for t in (1, 2, 3))):
    ov = int(np.prod(G[list(R)], axis=0).sum())
    if any(r in S for r in R):
        if ov % 2: bad.append(("check", R, ov))
    else:
        want = 1 if len(R) == 3 else 0
        if ov % 2 != want: bad.append(("logical", R, ov))
print("parity violations:", len(bad), bad[:5])
# harmful patterns by weight
Gl, Gs = G[:k], G[k:]
for w in range(1, 5):
    cnt = 0
    for J in itertools.combinations(range(n), w):
        idx = list(J)
        if not (Gs[:, idx].sum(axis=1) % 2).any() and (Gl[:, idx].sum(axis=1) % 2).any():
            cnt += 1
    print("weight", w, "harmful", cnt)
