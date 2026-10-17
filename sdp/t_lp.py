"""Convex relaxation (LP) for distilling one T state with suppression order d from n noisy inputs.

Reduction (see README): twirling inputs and output by W = (X+Y)/sqrt2 and averaging over input
permutations, any stabilizer protocol becomes
    Lambda(rho) = sum_e <phi_e|rho|phi_e> ( lam[|e|,0] |T><T| + lam[|e|,1] Z|T><T|Z ),
with lam >= 0, lam[v,1] = 0 for v < d (harmless weights), lam[0,0] > 0 (fault-free input accepted),
and Choi operator J = sum_{e,b} lam[|e|,b] |phi_e,b><phi_e,b| in the cone of stabilizer states on n+1 qubits.

Level-k relaxation: for every stabilizer state s on n-k of the inputs, contracting J with s gives
an operator on the other k inputs and the output that must again lie in the stabilizer cone:
    G_s(j,b) = C(k,j) sum_v m_v(s) lam[j+v, b]   in   cone{ agg(d(s')) : s' stabilizer on k+1 qubits },
where m_v(s) is the T-basis weight distribution of s and agg sums d(s') over patterns with
|e_K| = j on the k inputs and e_out = b. The generator set is exact (all stabilizer states on
k+1 qubits), and any family of conditioning states s gives valid constraints. Infeasibility therefore
proves that no stabilizer protocol exists.
"""
import itertools
import math
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from stab_enum import enumerate_distributions, OMEGA


def weights(N):
    return np.array([bin(e).count("1") for e in range(2 ** N)])


def load_dist(N):
    try:
        return np.load(f"dist_N{N}.npy")
    except FileNotFoundError:
        D, _ = enumerate_distributions(N)
        np.save(f"dist_N{N}.npy", D)
        return D


def bit_index_order(N):
    """Patterns are indexed by itertools.product order: e[0] is the most significant bit."""
    E = np.array(list(itertools.product((0, 1), repeat=N)), dtype=np.int64)
    return E


def generators(k):
    """Aggregated generators on k inputs + 1 output: vectors indexed by (j, b), flattened j*2+b."""
    N = k + 1
    D = load_dist(N)
    E = bit_index_order(N)
    j = E[:, :k].sum(axis=1)
    b = E[:, k]
    A = np.zeros((2 ** N, (k + 1) * 2))
    A[np.arange(2 ** N), j * 2 + b] = 1.0
    G = D @ A
    G = np.unique(np.round(G, 12), axis=0)
    return G


def weight_dists_exact(N):
    """T-basis weight distributions m(s) of all stabilizer states on N qubits (distinct)."""
    D = load_dist(N)
    E = bit_index_order(N)
    w = E.sum(axis=1)
    A = np.zeros((2 ** N, N + 1))
    A[np.arange(2 ** N), w] = 1.0
    return np.unique(np.round(D @ A, 12), axis=0)


def random_stab_weight_dist(N, rng, t=None):
    """Weight distribution of a random stabilizer state in Dehaene-De Moor form."""
    if t is None:
        t = rng.integers(0, N + 1)
    while True:
        B = rng.integers(0, 2, size=(t, N))
        if np.linalg.matrix_rank(B) == t if t else True:
            # rank over the reals is a proxy; check over F2 below
            break
    # F2 rank check
    def f2rank(M):
        M = M.copy() % 2
        r = 0
        for c in range(M.shape[1]):
            piv = [i for i in range(r, M.shape[0]) if M[i, c]]
            if not piv:
                continue
            M[[r, piv[0]]] = M[[piv[0], r]]
            for i in range(M.shape[0]):
                if i != r and M[i, c]:
                    M[i] ^= M[r]
            r += 1
        return r
    while t and f2rank(B) < t:
        B = rng.integers(0, 2, size=(t, N))
    c = rng.integers(0, 2, size=N)
    U = np.array(list(itertools.product((0, 1), repeat=t)), dtype=np.int64).reshape(2 ** t, t)
    X = (U @ B + c) % 2
    l = rng.integers(0, 4, size=t)
    Qm = np.triu(rng.integers(0, 2, size=(t, t)), 1)
    expo = (U @ l + 2 * np.einsum("ui,ij,uj->u", U, Qm, U)) % 4
    amp_x = (1j ** expo) * np.conj(OMEGA) ** X.sum(axis=1)
    E = np.array(list(itertools.product((0, 1), repeat=N)), dtype=np.int64)
    amp = ((-1.0) ** ((X @ E.T) % 2)).T @ amp_x * 2.0 ** (-(N + t) / 2)
    d = np.abs(amp) ** 2
    m = np.bincount(E.sum(axis=1), weights=d, minlength=N + 1)
    return m


def ghz_type(N):
    """Weight distribution of |G_N> = (|+i>^N - |-i>^N)/sqrt2 (even N) from the paper."""
    alpha = (1 + OMEGA) / 2
    beta = (1 - OMEGA) / 2
    m = np.zeros(N + 1)
    for w in range(N + 1):
        amp = (alpha ** (N - w) * beta ** w - beta ** (N - w) * alpha ** w) / math.sqrt(2)
        m[w] = math.comb(N, w) * abs(amp) ** 2
    return m / m.sum()


def conditioning_set(R, rng, n_random=2000):
    """Weight distributions of stabilizer states on R qubits used as conditioning states."""
    ms = []
    if R <= 5:
        ms.extend(weight_dists_exact(R))
    else:
        # products of exact smaller sets (convolutions) and random states
        small = {r: weight_dists_exact(r) for r in range(1, 6)}
        for _ in range(n_random):
            parts = []
            left = R
            while left > 0:
                r = int(rng.integers(1, min(5, left) + 1))
                parts.append(small[r][rng.integers(len(small[r]))])
                left -= r
            m = np.array([1.0])
            for p in parts:
                m = np.convolve(m, p)
            ms.append(m)
        for _ in range(n_random):
            ms.append(random_stab_weight_dist(R, rng))
        if R % 2 == 0:
            ms.append(ghz_type(R))
    ms = np.unique(np.round(np.array(ms), 12), axis=0)
    return ms


def solve(n, d, k, rng=None, n_random=2000, verbose=False, extra_levels=()):
    """Return (feasible, lam) for the level-k relaxation (plus any extra levels)."""
    rng = rng or np.random.default_rng(0)
    m = gp.Model()
    m.Params.OutputFlag = 1 if verbose else 0
    lam = m.addVars(n + 1, 2, lb=0.0, name="lam")
    for v in range(min(d, n + 1)):
        m.addConstr(lam[v, 1] == 0)
    m.addConstr(lam[0, 0] == 1)
    for kk in (k,) + tuple(extra_levels):
        if kk > n:
            continue
        G = generators(kk)
        R = n - kk
        ms = conditioning_set(R, rng, n_random) if R > 0 else np.array([[1.0]])
        for mvec in ms:
            mu = m.addVars(len(G), lb=0.0)
            for j in range(kk + 1):
                for b in range(2):
                    lhs = gp.quicksum(math.comb(kk, j) * mvec[v] * lam[j + v, b]
                                      for v in range(R + 1) if mvec[v] > 1e-15)
                    rhs = gp.quicksum(G[i, j * 2 + b] * mu[i] for i in range(len(G)) if G[i, j * 2 + b] > 1e-15)
                    m.addConstr(lhs == rhs)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        return True, np.array([[lam[v, b].X for b in range(2)] for v in range(n + 1)])
    return False, None


if __name__ == "__main__":
    import sys
    d = int(sys.argv[1])
    k = int(sys.argv[2])
    for n in range(2, 16):
        feas, lam = solve(n, d, k)
        print(f"d={d} k={k} n={n}: {'feasible' if feas else 'INFEASIBLE (no protocol)'}", flush=True)
