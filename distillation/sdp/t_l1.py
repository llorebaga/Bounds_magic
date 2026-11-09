"""Pauli-l1 (stabilizer-norm) relaxation for one T state.

A stabilizer mixture J on N qubits satisfies sum_P |Tr(J P)| <= 2^N Tr(J), since a pure stabilizer
state has exactly 2^N Pauli expectations equal to +-1 and the rest zero. For the symmetric,
T-basis-diagonal Choi operator J = sum_{e,b} lam[|e|,b] |phi_e,b><phi_e,b| (n inputs, 1 output),
write |phi_x><phi_x| = prod_j (I + (-1)^{x_j} W_j)/2 with W = (X+Y)/sqrt2. Then
    Tr(J P) = c(k,o) 2^{-(k+o)/2}   for each of the C(n,k) 2^{k+o} Paulis with X/Y letters on
                                    k inputs (and on the output if o = 1), identity elsewhere,
    c(k,o)  = sum_{v,b} lam[v,b] K_k(v) (-1)^{b o},    K_k(v) Krawtchouk,
and every other Pauli expectation vanishes. The constraint is linear in lam with absolute values.
It can be combined with the level-k constraints of t_lp.py.
"""
import math
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from t_lp import generators, conditioning_set


def kraw(n, k, v):
    return sum((-1) ** i * math.comb(k, i) * math.comb(n - k, v - i) for i in range(0, min(k, v) + 1))


def add_l1(model, lam, n):
    absv = {}
    total = 0
    for k in range(n + 1):
        for o in range(2):
            c = gp.quicksum(lam[v, b] * kraw(n, k, v) * (-1) ** (b * o) for v in range(n + 1) for b in range(2))
            if (k, o) == (0, 0):
                c00 = c
                continue
            a = model.addVar(lb=0.0)
            model.addConstr(a >= c)
            model.addConstr(a >= -c)
            total += math.comb(n, k) * 2 ** ((k + o) / 2) * a
    model.addConstr(total <= (2 ** (n + 1) - 1) * c00)


def solve(n, d, use_l1=True, k=None, n_random=400, rng=None):
    rng = rng or np.random.default_rng(0)
    m = gp.Model()
    m.Params.OutputFlag = 0
    lam = m.addVars(n + 1, 2, lb=0.0)
    for v in range(min(d, n + 1)):
        m.addConstr(lam[v, 1] == 0)
    m.addConstr(lam[0, 0] == 1)
    if use_l1:
        add_l1(m, lam, n)
    if k is not None and k <= n:
        G = generators(k)
        R = n - k
        ms = conditioning_set(R, rng, n_random) if R > 0 else np.array([[1.0]])
        for mvec in ms:
            mu = m.addVars(len(G), lb=0.0)
            for j in range(k + 1):
                for b in range(2):
                    lhs = gp.quicksum(math.comb(k, j) * mvec[v] * lam[j + v, b] for v in range(R + 1))
                    rhs = gp.quicksum(G[i, j * 2 + b] * mu[i] for i in range(len(G)))
                    m.addConstr(lhs == rhs)
    m.optimize()
    return m.Status == GRB.OPTIMAL


if __name__ == "__main__":
    for d in range(2, 7):
        n = d
        while not solve(n, d):
            n += 1
        print(f"T d={d}: Pauli-l1 relaxation first feasible n = {n}", flush=True)
