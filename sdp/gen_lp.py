"""Level-k LP relaxation for diagonal targets tau = U|+>^r, U in {T, CS, CCZ}.

The Cliffords U X^a U^dagger fix |tau> and twirling the output by them dephases it in the orthonormal
basis {Z^a |tau>}, a in F2^r (for CCZ, U X_1 U^dagger = X_1 CZ_23). Averaging over input permutations
and over the qubit permutations fixing tau, the protocol becomes
    Lambda(rho) = sum_e <phi_e|rho|phi_e> sum_w lam[|e|, w] P_w,   P_w = sum_{|a| = w} Z^a|tau><tau|Z^a / C(r,w),
with lam >= 0, lam[0,0] = 1, lam[v,w] = 0 for w > 0 at harmless weights v (v < d, and v > n-d for
rational targets by Theorem 2). Level-k constraint: for every stabilizer state s on n-k inputs,
    g_s(j,w) = C(k,j) sum_v m_v(s) lam[j+v, w]  in  cone{ agg(s') : s' stabilizer on k+r qubits },
agg(s')(j,w) = sum_{|e_K|=j, |a|=w} |<phi_{e_K} (x) Z^a tau | s'>|^2. Generators are exact enumerations.
"""
import itertools, math, sys, time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.spatial import ConvexHull

from stab_enum import subspaces, bitrows
from t_lp import conditioning_set

OMEGA = np.exp(1j * np.pi / 4)
TARGETS = {
    "T":   (1, lambda y: OMEGA ** y[:, 0], False),
    "CS":  (2, lambda y: 1j ** (y[:, 0] * y[:, 1]), True),
    "CCZ": (3, lambda y: (-1.0) ** (y[:, 0] * y[:, 1] * y[:, 2]), True),
}


def generators(target, k):
    """Distinct aggregated vectors (index j*(r+1)+w) of all stabilizer states on k+r qubits."""
    fn = f"gen_{target}_k{k}.npy"
    try:
        return np.load(fn)
    except FileNotFoundError:
        pass
    r, u, _ = TARGETS[target]
    N = k + r
    Xall = bitrows(N)
    E = Xall.copy()                                  # patterns (e_K, a) in the same bit order
    # phase per computational basis state: conj(omega)^{|x_K|} * conj(u(x_out))
    ph = np.conj(OMEGA) ** Xall[:, :k].sum(axis=1) * np.conj(u(Xall[:, k:]))
    jw = Xall[:, :k].sum(axis=1) * (r + 1) + Xall[:, k:].sum(axis=1)   # aggregation index of pattern
    Agg = np.zeros((2 ** N, (k + 1) * (r + 1)))
    Agg[np.arange(2 ** N), jw] = 1.0
    found = {}
    for t in range(N + 1):
        U = bitrows(t)
        pairs = [(a, b) for a in range(t) for b in range(a + 1, t)]
        L = bitrows(t)
        lu = ((L @ U.T)[:, None, :] + 2 * (L @ U.T)[None, :, :]).reshape(-1, len(U))
        Q = bitrows(len(pairs))
        qu = Q @ np.stack([U[:, a] * U[:, b] for a, b in pairs], axis=1).T if pairs else np.zeros((1, len(U)), int)
        phases = (1j ** ((lu[:, None, :] + 2 * qu[None, :, :]) % 4)).reshape(-1, len(U))
        for B, piv in subspaces(N, t):
            nonpiv = [j for j in range(N) if j not in piv]
            for cb in itertools.product((0, 1), repeat=len(nonpiv)):
                c = np.zeros(N, dtype=np.int64); c[nonpiv] = cb
                X = (U @ B.astype(np.int64) + c) % 2
                idx = X @ (1 << np.arange(N - 1, -1, -1))
                Mx = ph[idx][:, None] * (-1.0) ** ((X @ E.T) % 2)
                amp = phases @ Mx * 2.0 ** (-(N + t) / 2)
                D = (np.abs(amp) ** 2) @ Agg
                for row in np.round(D, 11):
                    found.setdefault(row.tobytes(), row)
    G = np.array(list(found.values()))
    assert np.allclose(G.sum(axis=1), 1)
    np.save(fn, G)
    return G


_F = {}
def facets(target, k):
    if (target, k) in _F:
        return _F[(target, k)]
    G = generators(target, k)
    c = G.mean(axis=0)
    U_, S, Vt = np.linalg.svd(G - c, full_matrices=False)
    B = Vt[: int((S > 1e-9).sum())]
    hull = ConvexHull((G - c) @ B.T, qhull_options="Qx")
    F = np.array([-(B.T @ e[:-1]) - (e[-1] - e[:-1] @ (B @ c)) for e in hull.equations])
    F = np.unique(np.round(F, 12), axis=0)
    assert (G @ F.T >= -1e-9).all()
    _F[(target, k)] = F
    return F


def contraction(n, k, r, mv):
    A = np.zeros(((k + 1) * (r + 1), (n + 1) * (r + 1)))
    for j in range(k + 1):
        for w in range(r + 1):
            for v, x in enumerate(mv):
                A[j * (r + 1) + w, (j + v) * (r + 1) + w] += math.comb(k, j) * x
    return A


def solve(target, n, d, levels, n_random=1500, seed=0):
    r, _, rational = TARGETS[target]
    rng = np.random.default_rng(seed)
    blocks = []
    for k in levels:
        if k > n:
            continue
        ms = conditioning_set(n - k, rng, n_random) if n > k else np.array([[1.0]])
        blocks.append((facets(target, k), np.array([contraction(n, k, r, mv) for mv in ms])))
    m = gp.Model(); m.Params.OutputFlag = 0; m.Params.NumericFocus = 3
    x = m.addMVar((n + 1) * (r + 1), lb=0.0)
    harmless = [v for v in range(n + 1) if v < d or (rational and v > n - d)]
    for v in harmless:
        for w in range(1, r + 1):
            m.addConstr(x[v * (r + 1) + w] == 0)
    m.addConstr(x.sum() == 1)
    m.setObjective(x[0], GRB.MAXIMIZE)
    for _ in range(5000):
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return 0.0
        lam = x.X; new = 0
        for F, As in blocks:
            V = np.einsum("fr,mrc,c->mf", F, As, lam)
            bad = np.argwhere(V < -1e-8)
            for i in np.argsort(V[bad[:, 0], bad[:, 1]])[:800]:
                mi, fi = bad[i]; m.addConstr((F[fi] @ As[mi]) @ x >= 0); new += 1
        if new == 0:
            return m.ObjVal
    raise RuntimeError


if __name__ == "__main__":
    target = sys.argv[1]
    levels = tuple(int(a) for a in sys.argv[2].split(","))
    ds = [int(a) for a in sys.argv[3].split(",")]
    for k in levels:
        t0 = time.time(); G = generators(target, k); F = facets(target, k)
        print(f"{target} k={k}: {len(G)} generators, {len(F)} facets [{time.time()-t0:.0f}s]", flush=True)
    for d in ds:
        out = []
        for n in range(d, 4 * d + 3):
            val = solve(target, n, d, levels)
            out.append(f"{n}:{val:.2g}")
            if val > 1e-6:
                break
        print(f"{target} d={d} levels={levels}: max lam00 by n -> {' '.join(out)}", flush=True)
