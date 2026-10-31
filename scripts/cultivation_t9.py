"""Nine T gates suffice for fault distance two (paper/cultivation.tex, Proposition 18 and Sec. S5).

The circuit prepares nine |+> states, applies T to each, and measures eight weight-four stabilizers of the
nine-input factory code of paper/paper.tex (Proposition 7 there; a [[9,1,3]] code) with one ancilla each, the
first with a flag; the output stays encoded and is decoded ideally. An integer program over the fault map of
Theorem 1 chooses the stabilizers, their order, the order of the qubits in each measurement and the flagged
measurements; the result is then verified by enumerating every single and every pair of faults, with the fault
map and with direct state-vector simulation. Finally, the coefficient of p^2 in the logical error per accepted run
is computed under the uniform depolarizing model of the cultivation paper, without and with idle noise.

Usage: python cultivation_t9.py [--mip]     (--mip re-solves the integer program with Gurobi)
Writes results/cultivation/t9_circuit.json.
"""
import itertools
import json
import os
import sys
import time

import numpy as np

import cultiv as cv

N = 9
FACTORY = ["-XZIIZIIIX", "+XIZIZIIXI", "+ZXZZIIXZI", "+ZZXIZXIIZ",
           "-YXYIXIIZZ", "-YYXXIIIII", "+IZIZZIZII", "+IIZZZZIII"]
# solution of the integer program (stabilizer, qubit order, flag), in the order of measurement
PLAN = [("-IYIXIXIIY", (8, 5, 3, 1), 1), ("-IZZIIIIXX", (8, 2, 7, 1), 0), ("-IIIIIZZXX", (6, 8, 7, 5), 0),
        ("-YYXXIIIII", (2, 0, 3, 1), 0), ("+IIIXXIIYY", (7, 4, 3, 8), 0), ("+XIIZIZIXI", (5, 0, 3, 7), 0),
        ("-XZIIZIIIX", (4, 0, 1, 8), 0), ("-IIXXIIYYI", (6, 2, 3, 7), 0)]

_TAB = {("I", "I"): ("I", 1), ("I", "X"): ("X", 1), ("I", "Y"): ("Y", 1), ("I", "Z"): ("Z", 1),
        ("X", "I"): ("X", 1), ("X", "X"): ("I", 1), ("X", "Y"): ("Z", 1j), ("X", "Z"): ("Y", -1j),
        ("Y", "I"): ("Y", 1), ("Y", "X"): ("Z", -1j), ("Y", "Y"): ("I", 1), ("Y", "Z"): ("X", 1j),
        ("Z", "I"): ("Z", 1), ("Z", "X"): ("Y", 1j), ("Z", "Y"): ("X", -1j), ("Z", "Z"): ("I", 1)}


def pmul(a, b):
    """product of two commuting signed Pauli strings"""
    s, ph = "", (-1 if a[0] == "-" else 1) * (-1 if b[0] == "-" else 1)
    for x, y in zip(a[1:], b[1:]):
        c, p = _TAB[(x, y)]
        s += c
        ph *= p
    return ("+" if np.real(ph) > 0 else "-") + s


def group(gens):
    G = {"+" + "I" * N}
    for g in gens:
        G |= {pmul(h, g) for h in G}
    return G


def sym(s):
    x = sum(1 << q for q, c in enumerate(s[1:]) if c in "XY")
    z = sum(1 << q for q, c in enumerate(s[1:]) if c in "YZ")
    return x | (z << N)


def anti(a, b):
    m = (1 << N) - 1
    return (bin(a & m & (b >> N)).count("1") + bin((a >> N) & b & m).count("1")) & 1


def circuit(plan):
    """ops, ideal op indices; qubits 0-8 data, 9 ancilla, 10 flag"""
    ops = [("R", q, "X") for q in range(N)] + [("T", q, 1) for q in range(N)]
    for g, order, flag in plan:
        P = g[1:]
        ops.append(("R", 9, "X"))
        for m, q in enumerate(order):
            if flag and m == 1:
                ops += [("R", 10, "Z"), ("U2", "CX", 9, 10)]
            if flag and m == len(order) - 1:
                ops.append(("U2", "CX", 9, 10))
            ops.append(("U2", "C" + P[q], 9, q))
        ops.append(("M", "X", (9,), 1 if g[0] == "+" else -1, True))
        if flag:
            ops.append(("M", "Z", (10,), 1, True))
    ideal = set()
    for g, _, _ in plan:
        sup = [q for q in range(N) if g[1 + q] != "I"]
        ideal.add(len(ops))
        ops.append(("M", "".join(g[1 + q] for q in sup), tuple(sup), 1 if g[0] == "+" else -1, False))
    return ops, ideal


# ---------------------------------------------------------------- integer program
class InputHarm:
    """harmlessness of a Pauli on the nine T inputs (collapse condition, state vectors)"""

    def __init__(self):
        T1 = np.array([1, cv.W8]) / np.sqrt(2)
        v = T1
        for _ in range(N - 1):
            v = np.kron(v, T1)
        self.T = v
        self.ref = self.proj(v)
        self.cache = {}

    def apply(self, v, ps):
        v = v.reshape([2] * N)
        for q, c in enumerate(ps):
            if c != "I":
                v = np.moveaxis(np.tensordot(cv.P1[c], v, axes=([1], [q])), 0, q)
        return v.reshape(-1)

    def proj(self, v):
        for g in FACTORY:
            v = (v + (-1 if g[0] == "-" else 1) * self.apply(v, g[1:])) / 2
        return v

    def __call__(self, R):
        if R not in self.cache:
            ps = "".join("IXZY"[((R >> q) & 1) + 2 * ((R >> (N + q)) & 1)] for q in range(N))
            self.cache[R] = cv.parallel(self.proj(self.apply(self.T, ps)), self.ref)
        return self.cache[R]


def dangers(g, order, flag, harm):
    """residual data Paulis R of single faults in the measurement of g that are harmful unless R anticommutes
    with a stabilizer measured earlier (Sec. S5)"""
    ops, _ = circuit([(g, order, flag)])
    ops = ops[2 * N:-1]
    meas = [i for i, op in enumerate(ops) if op[0] == "M"]
    sk = cv.Skeleton.__new__(cv.Skeleton)
    sk.ops, sk.nq, sk.tgates, sk.proj_of_t = ops, 11, [], {}
    sk.proj_of_op = {i: k for k, i in enumerate(meas)}
    gv = sym(g)
    out = set()
    for i, fs in cv.locations(ops):
        for f in fs:
            fl, res = sk.fault_effect(i, f)
            if flag and (fl >> 1) & 1:
                continue                                   # flag raised: rejected
            phi = fl & 1                                   # the measurement of g is flipped
            R = (res.x & ((1 << N) - 1)) | ((res.z & ((1 << N) - 1)) << N)
            if phi == anti(R, gv) and not harm(R):
                out.add(R)
    return out


def solve_mip():
    import gurobipy as gp
    from gurobipy import GRB
    harm = InputHarm()
    W4 = sorted(s for s in group(FACTORY) if sum(c != "I" for c in s[1:]) == 4)
    V = [sym(s) for s in W4]
    D = {}
    for i, g in enumerate(W4):
        sup = [q for q in range(N) if g[1 + q] != "I"]
        for pi in itertools.permutations(sup):
            for fl in (0, 1):
                D[(i, pi, fl)] = dangers(g, pi, fl, harm)
    circuits = []                                          # minimal dependent sets of the binary matroid
    for r in range(2, N + 1):
        for S in itertools.combinations(range(len(V)), r):
            x = 0
            for i in S:
                x ^= V[i]
            if x == 0 and not any(set(C) < set(S) for C in circuits):
                circuits.append(S)
    n = len(V)
    m = gp.Model()
    m.Params.OutputFlag = 0
    s = m.addVars(n, vtype=GRB.BINARY)
    y = m.addVars([(i, j) for i in range(n) for j in range(n) if i != j], vtype=GRB.BINARY)
    keys = list(D)
    z = m.addVars(len(keys), vtype=GRB.BINARY)
    m.addConstr(s.sum() == N - 1)
    for C in circuits:
        m.addConstr(gp.quicksum(s[i] for i in C) <= len(C) - 1)
    for i in range(n):
        m.addConstr(gp.quicksum(z[k] for k, key in enumerate(keys) if key[0] == i) == s[i])
        for j in range(n):
            if i != j:
                m.addConstr(y[i, j] <= s[i])
                m.addConstr(y[i, j] <= s[j])
                if i < j:
                    m.addConstr(y[i, j] + y[j, i] >= s[i] + s[j] - 1)
                    m.addConstr(y[i, j] + y[j, i] <= 1)
    for i, j, k in itertools.permutations(range(n), 3):
        if i < j and i < k:
            m.addConstr(y[i, j] + y[j, k] + y[k, i] <= 2)
    for k, (i, pi, fl) in enumerate(keys):
        for R in D[(i, pi, fl)]:
            m.addConstr(z[k] <= gp.quicksum(y[j, i] for j in range(n) if j != i and anti(R, V[j])))
    m.setObjective(gp.quicksum(z[k] for k, key in enumerate(keys) if key[2] == 1), GRB.MINIMIZE)
    m.optimize()
    sel = [i for i in range(n) if s[i].X > .5]
    order = sorted(sel, key=lambda i: sum(y[j, i].X > .5 for j in sel if j != i))
    plan = []
    for i in order:
        k = [k for k, key in enumerate(keys) if key[0] == i and z[k].X > .5][0]
        plan.append((W4[i], keys[k][1], keys[k][2]))
    info = dict(weight4_elements=len(W4), matroid_circuits=len(circuits), variables=m.NumVars,
                constraints=m.NumConstrs, min_flags=int(round(m.ObjVal)), seconds=round(m.Runtime, 1))
    return plan, info


# ---------------------------------------------------------------- verification
def exact_proof(plan):
    """the computational step of the proof of Proposition 18, in exact F2 arithmetic: every single fault in a
    measurement either raises the flag, or leaves a data Pauli R that anticommutes with an earlier stabilizer, or
    flips the outcome inconsistently with R, or has weight at most one modulo the stabilizer group"""
    S = [sym(g) for g in group([p[0] for p in plan])]
    m = (1 << N) - 1
    counts = dict(flagged=0, earlier=0, inconsistent=0, weight_le_1=0, failures=0)
    for k, (g, order, flag) in enumerate(plan):
        ops, _ = circuit([(g, order, flag)])
        ops = ops[2 * N:-1]
        meas = [i for i, op in enumerate(ops) if op[0] == "M"]
        sk = cv.Skeleton.__new__(cv.Skeleton)
        sk.ops, sk.nq, sk.tgates, sk.proj_of_t = ops, 11, [], {}
        sk.proj_of_op = {i: j for j, i in enumerate(meas)}
        for i, fs in cv.locations(ops):
            for f in fs:
                fl, res = sk.fault_effect(i, f)
                R = (res.x & m) | ((res.z & m) << N)
                if flag and (fl >> 1) & 1:
                    counts["flagged"] += 1
                elif any(anti(R, sym(h[0])) for h in plan[:k]):
                    counts["earlier"] += 1
                elif (fl & 1) != anti(R, sym(g)):
                    counts["inconsistent"] += 1
                elif min(bin(((R ^ s) & m) | ((R ^ s) >> N)).count("1") for s in S) <= 1:
                    counts["weight_le_1"] += 1
                else:
                    counts["failures"] += 1
    return counts


def verify(plan):
    ops, ideal = circuit(plan)
    A = cv.Analysis(ops, 11, list(range(N)), ideal)
    sk = A.sk
    res = dict(qubits=11, t_gates=sum(op[0] == "T" for op in ops),
               two_qubit_gates=sum(op[0] == "U2" for op in ops),
               preparations=sum(op[0] == "R" for op in ops),
               noisy_measurements=sum(op[0] == "M" for i, op in enumerate(ops) if i not in ideal),
               locations=len(A.locs), elementary_faults=len(A.faults),
               acceptance=float(np.vdot(A.ref, A.ref).real),
               code_dimension_log2=sk.t - len(sk.SC_rref),
               ideal_projections_deterministic=all(sk.deterministic[sk.proj_of_op[i]] for i in ideal))
    psi = cv.simulate(ops, 11)
    psi = cv.out_vec(psi, 11, list(range(N)))
    psi = psi / np.linalg.norm(psi)
    # logical Bloch vector: logical operators of the [[9,1,3]] code from the stabilizer group
    G = group([p[0] for p in plan])
    res["output_is_codeword"] = all(abs(np.vdot(psi, _apply(psi, g)) - 1) < 1e-9
                                    for g in G)
    singles = [(i, f, k) for (li, i, f, k) in A.faults]
    res["harmful_singles_fault_map"] = sum(not A.harmless_key(k) for i, f, k in singles)
    res["harmful_singles_direct"] = sum(not A.harmless_direct({i: f}) for i, f, k in singles)
    cls = A.weight_classes(2)
    res["pair_configurations"] = sum(cls.values())
    res["pair_classes"] = len(cls)
    res["harmful_pairs"] = sum(c for k, c in cls.items() if not A.harmless_key(k))
    res["exact_proof_cases"] = exact_proof(plan)
    kinds = {}
    for i, f, k in singles:
        c = A.classify(k)
        kinds[c] = kinds.get(c, 0) + 1
    res["single_fault_classes"] = kinds
    by_type = {}
    for i, f, k in singles:
        t = ops[i][0] if ops[i][0] != "U2" else "2q"
        t = {"R": "preparation", "T": "T gate", "2q": "two-qubit gate", "M": "measurement"}[t]
        c = A.classify(k)
        by_type.setdefault(t, {}).setdefault(c, 0)
        by_type[t][c] += 1
    res["single_fault_classes_by_location"] = by_type
    return ops, ideal, res


# ---------------------------------------------------------------- uniform depolarizing noise
D1 = [("X", 1 / 3), ("Y", 1 / 3), ("Z", 1 / 3)]
D2 = [("".join(p), 1 / 15) for p in itertools.product("IXYZ", repeat=2) if p != ("I", "I")]


def with_idles(ops, ideal):
    """the circuit in as-soon-as-possible time steps, except that ancilla and flag preparations come just before
    their first gate, with an identity operation on every qubit that is live (prepared and not yet measured) and
    idle in a time step; the ideal decoder stays at the end"""
    step, last = {}, {}
    for i, op in enumerate(ops):
        if i not in ideal:
            qs = cv.support(op)
            step[i] = 1 + max(last.get(q, -1) for q in qs)
            for q in qs:
                last[q] = step[i]
    for i, op in enumerate(ops):
        if op[0] == "R" and op[1] >= N:
            step[i] = min(step[j] for j in step if j > i and op[1] in cv.support(ops[j])) - 1
    out, live = [], set()
    for s in range(max(step.values()) + 1):
        touched = set()
        for i in sorted(i for i in step if step[i] == s):
            op = ops[i]
            out.append(op)
            touched |= set(cv.support(op))
            if op[0] == "R":
                live.add(op[1])
            elif op[0] == "M":
                live -= set(op[2])
        out += [("I", (q,)) for q in sorted(live - touched)]
    new_ideal = set(range(len(out), len(out) + len(ideal)))
    return out + [ops[i] for i in sorted(ideal)], new_ideal


def depolarizing(plan, idle):
    """coefficient c2 of p^2 in the logical error per accepted run under the uniform depolarizing model of the
    cultivation paper (X or Z flip after a preparation, DEPOLARIZE1 after T and on idle qubits, DEPOLARIZE2 after
    two-qubit gates, result flips of measurements), without or with idle noise"""
    ops, ideal = circuit(plan)
    if idle:
        ops, ideal = with_idles(ops, ideal)
    A = cv.Analysis(ops, 11, list(range(N)), ideal)
    weights = {}
    for i, fs in A.locs:
        op = ops[i]
        if op[0] == "R":
            weights[i] = {("Z" if op[2] == "X" else "X"): 1.0}
        elif op[0] == "M":
            weights[i] = {"flip": 1.0}
        else:
            weights[i] = dict(D2 if op[0] == "U2" else D1)
    by = {}
    for li, i, f, k in A.faults:
        if weights[i].get(f, 0) > 0:
            by.setdefault(li, []).append((i, weights[i][f], k))
    cache, c2, nharm, kinds = {}, 0.0, 0, {}
    nref = np.vdot(A.ref, A.ref).real
    for a, b in itertools.combinations(sorted(by), 2):
        for i1, w1, k1 in by[a]:
            for i2, w2, k2 in by[b]:
                D, U, E = (x ^ y for x, y in zip(k1, k2))
                if D or A.harmless_UE(U, E):
                    continue
                if (U, E) not in cache:
                    o = A.output_of(U, E)
                    no = np.vdot(o, o).real
                    cache[(U, E)] = no / nref * (1 - abs(np.vdot(A.ref, o)) ** 2 / (nref * no))
                c2 += w1 * w2 * cache[(U, E)]
                nharm += 1
                kk = "-".join(sorted((ops[i1][0], ops[i2][0])))
                kinds[kk] = kinds.get(kk, 0) + 1
    return dict(locations=len(by), idle_locations=sum(op[0] == "I" for op in ops), harmful_pairs=nharm,
                harmful_pairs_by_operations=kinds, c2=c2)


def _apply(psi, g):
    ps = g[1:]
    v = psi.reshape([2] * N)
    for q, c in enumerate(ps):
        if c != "I":
            v = np.moveaxis(np.tensordot(cv.P1[c], v, axes=([1], [q])), 0, q)
    return (-1 if g[0] == "-" else 1) * v.reshape(-1)


if __name__ == "__main__":
    plan, info = PLAN, None
    out = os.path.join(os.path.dirname(__file__), "..", "results", "cultivation", "t9_circuit.json")
    if os.path.exists(out):
        info = json.load(open(out)).get("mip")
    if "--mip" in sys.argv:
        t0 = time.time()
        plan, info = solve_mip()
        print("integer program:", info)
    for g, order, flag in plan:
        print("measure %s  qubit order %s%s" % (g, order, "  (flagged)" if flag else ""))
    ops, ideal, res = verify(plan)
    for k, v in res.items():
        print(k, v)
    ok = (res["harmful_singles_fault_map"] == 0 and res["harmful_singles_direct"] == 0 and res["t_gates"] == 9
          and res["ideal_projections_deterministic"] and res["output_is_codeword"]
          and res["exact_proof_cases"]["failures"] == 0)
    print("FAULT DISTANCE 2 WITH NINE T GATES: OK" if ok else "CHECK FAILED")
    dep = dict(without_idle=depolarizing(plan, False), with_idle=depolarizing(plan, True))
    for k, v in dep.items():
        print("uniform depolarizing,", k.replace("_", " ") + ":", v)
    json.dump(dict(plan=[[g, list(o), f] for g, o, f in plan], ops=[list(op) for op in ops],
                   ideal=sorted(ideal), mip=info, verification=res, uniform_depolarizing=dep),
              open(out, "w"), indent=1)
