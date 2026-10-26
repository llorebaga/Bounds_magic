"""The distance-three cultivation circuit of Gidney, Shutty and Jones with T gates (paper/cultivation.tex, Sec. S5).

Composes the injection chunk and the T-check chunk of data/gsj_cultivation/ by qubit coordinates, replaces the
S and S_DAG gates that stand for T and T^dagger by T and T^dagger (or keeps S, for comparison), keeps the final
stabilizer measurements as an ideal decoder, and computes with the fault map of Theorem 2:
  - harmful single faults and pairs, for the T and for the S version, with one noisy location per operation;
  - the same with the uniform depolarizing model of the original code (DEPOLARIZE2 after CX, DEPOLARIZE1 after
    T, X/Z flips after R/RX, result flips of M/MX, DEPOLARIZE1 on every live idle qubit in every TICK of the
    chunk text), and the coefficient c2 of p^2 in the logical error per accepted run;
  - a set-cover integer program for flags that would detect every harmful pair;
  - a fit of (eps_T - eps_S)/p^2 = a + b p to the state-vector simulations published with the circuit.
Usage: python cultivation_gsj.py       Writes results/cultivation/gsj_d3.json.
"""
import csv
import itertools
import json
import os
import re
import sys

import numpy as np

import cultiv as cv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "gsj_cultivation")
D1 = [("X", 1 / 3), ("Y", 1 / 3), ("Z", 1 / 3)]
D2 = [("".join(p), 1 / 15) for p in itertools.product("IXYZ", repeat=2) if p != ("I", "I")]


def parse_moments(path):
    coords, moments, cur = {}, [], []
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("QUBIT_COORDS"):
            m = re.match(r"QUBIT_COORDS\(([^)]*)\)\s+(\d+)", line)
            coords[int(m.group(2))] = tuple(float(v) for v in m.group(1).split(","))
            continue
        name = line.split()[0]
        if name == "TICK":
            if cur:
                moments.append(cur)
                cur = []
            continue
        if name.startswith("DETECTOR") or name.startswith("OBSERVABLE"):
            continue
        parts = line.split()
        if name == "MPP":
            cur.append(("MPP", [[(x[0], int(x[1:])) for x in prod.split("*")] for prod in parts[1:]]))
        else:
            cur.append((name, [int(v) for v in parts[1:]]))
    if cur:
        moments.append(cur)
    return coords, moments


def build(gate="T", idle=False):
    """ops, fault weights per noisy op (coefficients of p), ideal op indices, data qubits, number of qubits"""
    qid = {}
    body, final = [], []
    for ci, name in enumerate(("d3_init_unitary.stim", "d3_double_cat_check.stim")):
        coords, moments = parse_moments(os.path.join(DATA, name))
        g = lambda q, coords=coords: qid.setdefault(coords[q], len(qid))
        ms = [[(n, [g(q) for q in qs]) if n != "MPP" else (n, [[(p, g(q)) for p, q in prod] for prod in qs])
               for n, qs in m] for m in moments]
        boundary = [all(n == "MPP" for n, _ in m) for m in ms]
        a = 0 if ci == 0 else min(i for i, b in enumerate(boundary) if not b)
        b = max(i for i, x in enumerate(boundary) if not x)
        body += ms[a:b + 1]
        if ci == 1:
            final = [prod for m in ms[b + 1:] for n, prods in m for prod in prods]
    nq = len(qid)
    ops, weights, live = [], {}, set()
    for m in body:
        touched = set()
        for name, qs in m:
            if name in ("R", "RX"):
                for q in qs:
                    if q in live:
                        ops.append(("RST", q, "Z"))       # discard of a qubit in a known state, basis resolved below
                    weights[len(ops)] = [("X" if name == "R" else "Z", 1.0)]
                    ops.append(("R", q, "Z" if name == "R" else "X"))
                    live.add(q)
                    touched.add(q)
            elif name == "CX":
                for c, t in zip(qs[::2], qs[1::2]):
                    weights[len(ops)] = D2
                    ops.append(("U2", "CX", c, t))
                    touched |= {c, t}
            elif name in ("S", "S_DAG"):
                for q in qs:
                    k = 1 if name == "S" else -1
                    weights[len(ops)] = D1
                    ops.append(("T", q, k) if gate == "T" else ("U1", q, "S" if k == 1 else "Sdg"))
                    touched.add(q)
            elif name in ("M", "MX"):
                for q in qs:
                    weights[len(ops)] = [("flip", 1.0)]
                    ops.append(("M", "Z" if name == "M" else "X", (q,), 1, True))
                    live.discard(q)
                    touched.add(q)
            else:
                raise ValueError(name)
        if idle:
            for q in sorted(live - touched):
                weights[len(ops)] = D1
                ops.append(("I", (q,)))
    data = sorted({q for prod in final for p, q in prod})
    ideal = set()
    for prod in final:
        if set(p for p, q in prod) == {"Y"}:
            continue                                         # the logical observable, not a projection
        ideal.add(len(ops))
        ops.append(("M", "".join(p for p, q in prod), tuple(q for p, q in prod), 1, False))
    ops, random = cv.resolve(ops, nq)
    assert not random, "a measurement is not deterministic in the fault-free run"
    # the weights refer to op indices before resolve; resolve keeps indices
    return ops, weights, ideal, data, nq


def analysis(ops, weights, ideal, data, nq):
    A = cv.Analysis.__new__(cv.Analysis)
    A.ops, A.nq, A.out, A.ideal, A.cache = ops, nq, data, ideal, {}
    A.sk = cv.Skeleton(ops, nq, data)
    A.ref = cv.out_vec(cv.simulate(ops, nq), nq, data)
    faults = []
    for loc, i in enumerate(sorted(weights)):
        for f, w in weights[i]:
            faults.append((loc, i, f, w, A.sk.key(*A.sk.fault_effect(i, f))))
    return A, faults


def harmful_pairs(A, faults):
    by = {}
    for x in faults:
        by.setdefault(x[0], []).append(x)
    out = []
    for a, b in itertools.combinations(sorted(by), 2):
        for (_, i1, f1, w1, k1) in by[a]:
            for (_, i2, f2, w2, k2) in by[b]:
                k = tuple(x ^ y for x, y in zip(k1, k2))
                if not A.harmless_key(k):
                    out.append(((i1, f1, w1), (i2, f2, w2), k))
    return out, sum(len(by[a]) * len(by[b]) for a, b in itertools.combinations(sorted(by), 2))


def damage(A, k):
    o = A.output_of(k[1], k[2])
    nref, no = np.vdot(A.ref, A.ref).real, np.vdot(o, o).real
    return no / nref, 1 - abs(np.vdot(A.ref, o)) ** 2 / (nref * no)


def all_faults(ops, ideal):
    """every Pauli (and flip) at every noisy operation, weight 1 (for fault distances)"""
    return {i: [(f, 1.0) for f in fs] for i, fs in cv.locations(ops, ideal)}


def flag_cover(ops, ideal, data, harm):
    """set cover: standard flags (fresh |0>, CX(q->f) before ops t1 and t2, measure Z) on any qubit q whose Z value
    is unchanged in [t1, t2); a pair is caught if exactly one of its faults puts an X on q inside the window"""
    import gurobipy as gp
    from gurobipy import GRB
    first = min(i for i, op in enumerate(ops) if op[0] == "T" and op[2] == -1 and i > 60)
    last = min(ideal)

    def changes_z(j, q):
        o = ops[j]
        return ((o[0] == "U2" and o[3] == q) or (o[0] in ("R", "RST") and o[1] == q) or (o[0] == "M" and q in o[2])
                or (o[0] == "U1" and o[1] == q and o[2] == "H"))
    qubits = sorted({q for op in ops[first:last] if op[0] == "U2" for q in op[2:]} | set(data))
    cands = []
    for q in qubits:
        for t1 in range(first, last):
            for t2 in range(t1 + 1, last + 1):
                if any(changes_z(j, q) for j in range(t1, t2)):
                    break
                cands.append((q, t1, t2))

    def xtrace(i, f):
        op = ops[i]
        P = cv.pauli_from_string(f, cv.support(op)) if f != "flip" else cv.Pauli()
        start = i if op[0] == "M" else i + 1
        tr = np.zeros((len(ops) + 1, max(qubits) + 1), dtype=np.uint8)
        for j in range(start, len(ops)):
            tr[j] = [(P.x >> q) & 1 for q in range(max(qubits) + 1)]
            o = ops[j]
            if o[0] == "R" and o[2] == "X":
                P.conj1("H", o[1])
            elif o[0] == "U1":
                P.conj1(o[2], o[1])
            elif o[0] == "U2":
                P.conj2(cv.G2[o[1]], o[2], o[3])
            elif o[0] == "T" and o[2] == -1:
                P.conj1("Sdg", o[1])
            elif o[0] == "M" and o[4]:
                P.set(o[2][0], 0, 0)
            elif o[0] == "RST":
                P.set(o[1], 0, 0)
        return tr
    tr = {}
    rows = []
    for (i1, f1, _), (i2, f2, _), _ in harm:
        for key in ((i1, f1), (i2, f2)):
            if key not in tr:
                tr[key] = xtrace(*key)
        rows.append([c for c, (q, t1, t2) in enumerate(cands)
                     if (tr[(i1, f1)][t1, q] ^ tr[(i1, f1)][t2, q]) ^ (tr[(i2, f2)][t1, q] ^ tr[(i2, f2)][t2, q])])
    uncoverable = [(harm[r][0][:2], harm[r][1][:2]) for r, cov in enumerate(rows) if not cov]
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVars(len(cands), vtype=GRB.BINARY)
    for cov in rows:
        if cov:
            m.addConstr(gp.quicksum(x[c] for c in cov) >= 1)
    m.setObjective(x.sum(), GRB.MINIMIZE)
    m.optimize()
    return dict(candidates=len(cands), uncoverable_pairs=[[list(a), list(b)] for a, b in uncoverable],
                flags_for_the_other_pairs=int(round(m.ObjVal)))


def fit_published():
    rows = {}
    with open(os.path.join(DATA, "stats_d3_unitary.csv")) as f:
        for r in csv.reader(f):
            if r[0].strip() == "shots":
                continue
            meta = json.loads(r[6])
            shots, errs, disc = int(r[0]), int(r[1]), int(r[2])
            rows.setdefault(meta["p"], {})[r[4].strip()] = (errs, shots - disc, shots)
    table, P, Dv, S = [], [], [], []
    for p in sorted(rows):
        r = rows[p]
        if "vec_intercept_t" not in r:
            continue
        (et, kt, nt), (es, ks, ns) = r["vec_intercept_t"], r["vec_intercept_s"]
        eT, eS, sT, sS = et / kt, es / ks, np.sqrt(et) / kt, np.sqrt(es) / ks
        table.append(dict(p=p, eps_T=eT, sd_T=sT, eps_S=eS, sd_S=sS, accept_T=kt / nt,
                          excess_over_p2=(eT - eS) / p ** 2, sd_excess=np.hypot(sT, sS) / p ** 2))
        P.append(p)
        Dv.append((eT - eS) / p ** 2)
        S.append(np.hypot(sT, sS) / p ** 2)
    P, Dv, S = map(np.array, (P, Dv, S))
    fits = {}
    for pmax in (0.003, 0.005, 0.01):
        m = P <= pmax + 1e-12
        M = np.vstack([np.ones(m.sum()), P[m]]).T / S[m, None]
        coef, *_ = np.linalg.lstsq(M, Dv[m] / S[m], rcond=None)
        cov = np.linalg.inv(M.T @ M)
        fits[str(pmax)] = dict(a=coef[0], sd_a=np.sqrt(cov[0, 0]), b=coef[1])
    return table, fits


if __name__ == "__main__":
    res = {}
    for gate in ("S", "T"):
        ops, w, ideal, data, nq = build(gate, idle=False)
        A, faults = analysis(ops, all_faults(ops, ideal), ideal, data, nq)
        h1 = [x for x in faults if not A.harmless_key(x[4])]
        harm, npairs = harmful_pairs(A, faults)
        r = dict(qubits=nq, operations=len(ops), t_gates=sum(op[0] == "T" for op in ops),
                 cx=sum(op[0] == "U2" for op in ops), noisy_locations=len(cv.locations(ops, ideal)),
                 elementary_faults=len(faults), harmful_singles=len(h1), pair_configurations=npairs,
                 harmful_pairs=len(harm))
        if gate == "T":
            sk = A.sk
            r["skeleton_code_dimension_log2"] = sk.t - len(sk.SC_rref)
            r["final_projections_deterministic_in_skeleton"] = [bool(sk.deterministic[sk.proj_of_op[i]])
                                                                for i in sorted(ideal)]
            r["direct_simulation_agrees"] = all(not A.harmless_direct({a[0]: a[1], b[0]: b[1]})
                                                for a, b, k in harm)
            r["harmful_pairs_list"] = [[list(a[:2]), list(b[:2]), list(damage(A, k))] for a, b, k in harm]
            r["flag_cover"] = flag_cover(ops, ideal, data, harm)
            T_ops = (ops, ideal, data)
        res[gate + "_one_location_per_operation"] = r
        print(gate, {k: v for k, v in r.items() if k != "harmful_pairs_list"})
    for gate in ("S", "T"):
        ops, w, ideal, data, nq = build(gate, idle=True)
        A, faults = analysis(ops, w, ideal, data, nq)
        harm, npairs = harmful_pairs(A, faults)
        c2, kinds = 0.0, {}
        for a, b, k in harm:
            acc, inf = damage(A, k)
            c2 += a[2] * b[2] * acc * inf
            kk = "-".join(sorted((ops[a[0]][0], ops[b[0]][0])))
            kinds[kk] = kinds.get(kk, 0) + 1
        r = dict(noisy_locations=len(w), idle_locations=sum(op[0] == "I" for op in ops),
                 harmful_singles=sum(not A.harmless_key(x[4]) for x in faults), harmful_pairs=len(harm),
                 harmful_pairs_by_operations=kinds, c2=c2)
        res[gate + "_uniform_depolarizing"] = r
        print(gate, "uniform depolarizing:", r)
    table, fits = fit_published()
    res["published_vector_simulations"] = table
    res["fit_excess_over_p2"] = fits
    print("published (eps_T - eps_S)/p^2:", [(t["p"], round(t["excess_over_p2"], 3), round(t["sd_excess"], 3))
                                             for t in table])
    print("fits a + b p:", {k: (round(v["a"], 3), round(v["sd_a"], 3)) for k, v in fits.items()})
    out = os.path.join(HERE, "..", "results", "cultivation", "gsj_d3.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
