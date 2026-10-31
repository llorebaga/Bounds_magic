"""Flag design v2: counterexample-guided ILP with ACCUMULATED, flag-fault-aware constraints.

Candidates are fixed once: (q, ga, gb, type) on base ticks, kept if non-disturbing in the base skeleton. Each
iteration builds the circuit with the currently chosen flags, certifies it with the TDEM and lists the malignant
configurations of weight < d_target. For every physical representative it adds the constraint

    sum_{c detects rep} y_c  >=  1 - sum_{f in flags(rep)} (1 - y_f) - sum_{g in layers(rep)} (1 - z_g),

so a configuration that involves faults of a flag f (or of the idle noise of an inserted layer g) only has to
be covered when f is chosen (g exists). The ILP is re-solved over all constraints each time, so it may drop or
replace earlier flags.
"""
import collections
import itertools
import time
import numpy as np
import stim

from tdem import TDEM
from faults import FaultTable
from fd_fast import fault_distance_fast
from fd_list import harmful_k4
from flags import split_ticks, tick_kinds, deterministic_flags


def build_flagged(c, flags, p, timing="aligned"):
    """flags: (q, ga, gb, f, typ). timing 'aligned': f is reset in the last reset tick <= ga and measured in
    the first measurement tick > gb; 'tight': reset in base tick ga, measured in base tick gb + 1.
    Returns (circuit, ticks) with ticks[i] = ('base', g) or ('ins', g) for every tick of the new circuit."""
    T = split_ticks(c)
    reset, meas = tick_kinds(T)
    nq = c.num_qubits
    gaps = collections.defaultdict(list)
    plan = []
    for (q, ga, gb, f, typ) in flags:
        if timing == "aligned":
            r = max(t for t in range(ga + 1) if reset[t])
            mm = [t for t in range(gb + 1, len(T)) if meas[t]]
            mt = mm[0] if mm else len(T) - 1
        else:
            r, mt = ga, gb + 1
        plan.append((q, ga, gb, f, r, mt, typ))
        prs = [(q[0], f), (q[1], f)] if typ == "ZZ" else [(q, f) if typ == "Z" else (f, q)]
        for pair in prs:
            gaps[ga].append((pair, "start"))
            gaps[gb].append((pair, "end"))
    allq = list(range(nq)) + [fl[3] for fl in flags]
    out = stim.Circuit()
    ticks = []
    old = new = 0
    remap = {}
    for t, tk in enumerate(T):
        for inst in tk:
            nm = inst.name
            if nm in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                out.append(nm, [stim.target_rec(remap[old + x.value] - new) for x in inst.targets_copy()],
                           inst.gate_args_copy())
                continue
            k = len(inst.targets_copy()) if nm in ("M", "MX", "MY") else (len(inst.target_groups()) if nm == "MPP" else 0)
            for j in range(k):
                remap[old + j] = new + j
            old += k
            new += k
            out.append(inst)
        for (q, ga, gb, f, r, mt, typ) in plan:
            zt = typ in ("Z", "ZZ")
            if t == r:
                out.append("R" if zt else "RX", [f])
                out.append("X_ERROR" if zt else "Z_ERROR", [f], [p])
            elif r < t < mt:
                out.append("DEPOLARIZE1", [f], [p])
            elif t == mt:
                out.append("M" if zt else "MX", [f], [p])
                out.append("DETECTOR", [stim.target_rec(-1)])
                new += 1
        ticks.append(("base", t))
        if t < len(T) - 1:
            out.append("TICK")
        if t in gaps:
            items = gaps[t]
            layers = []
            end_layer = {}
            for role in ("end", "start"):
                for it in items:
                    if it[1] != role:
                        continue
                    lo = 0 if role == "end" else max([end_layer.get(x, -1) + 1 for x in it[0]] + [0])
                    j = lo
                    while j < len(layers) and set(it[0]) & {x for e in layers[j] for x in e[0]}:
                        j += 1
                    while j >= len(layers):
                        layers.append([])
                    layers[j].append(it)
                    if role == "end":
                        for x in it[0]:
                            end_layer[x] = max(end_layer.get(x, -1), j)
            for lay in layers:
                pairs = [it[0] for it in lay]
                busy = {x for pr in pairs for x in pr}
                out.append("CX", [x for pr in pairs for x in pr])
                out.append("DEPOLARIZE2", [x for pr in pairs for x in pr], [p])
                live = [x for x in allq if x not in busy and
                        not any(x == f and not (r <= t < mt) for (q, ga, gb, f, r, mt, typ) in plan)]
                out.append("DEPOLARIZE1", live, [p])
                out.append("TICK")
                ticks.append(("ins", t))
    return out, ticks


def design(c, p, d_target, region, magic="T", tdem_kw=None, types=("Z", "X"), layer_weight=10.0,
           max_iter=12, max_reps=2048, timing="aligned", qubits=None, verbose=True, threads=8, checkpoint=None):
    import gurobipy as gp
    from gurobipy import GRB
    tdem_kw = tdem_kw or {}
    n0 = c.num_qubits
    t0 = time.time()
    base = TDEM(c, magic=magic, **tdem_kw)
    lo, hi = region
    qs = qubits if qubits is not None else list(range(n0))
    cands = [(q, ga, gb, typ) for q in qs for ga, gb in itertools.combinations(range(lo, hi + 1), 2)
             for typ in types if typ != "ZZ"]
    if "ZZ" in types:
        co = {q: tuple(v[:2]) for q, v in c.get_final_qubit_coordinates().items()}
        nbp = [(a, b) for a in qs for b in qs if a < b and abs(co[a][0] - co[b][0]) + abs(co[a][1] - co[b][1]) == 1]
        cands += [(pr, ga, gb, "ZZ") for pr in nbp for ga, gb in itertools.combinations(range(lo, hi + 1), 2)]
    cands = deterministic_flags(base, cands)
    cid = {cd: i for i, cd in enumerate(cands)}
    gaps = sorted({g for (q, ga, gb, typ) in cands for g in (ga, gb)})
    if verbose:
        print(f"{len(cands)} non-disturbing candidates on base ticks {lo}..{hi} ({time.time()-t0:.0f}s)", flush=True)
    constraints = []
    chosen = []
    log = []
    for it in range(max_iter):
        flags = [(q, ga, gb, n0 + i, typ) for i, (q, ga, gb, typ) in enumerate(chosen)]
        cur, ticks = build_flagged(c, flags, p, timing)
        m = TDEM(cur, magic=magic, **tdem_kw)
        ft = FaultTable(m, frames=True)
        cl = ft.classes()
        mech = [(pp, D, o) for pp, D, o, fs in cl]
        k, found = fault_distance_fast(m, min(3, d_target - 1), mech, verbose=False)
        if k is None and d_target - 1 >= 4:
            found = harmful_k4(m, mech, verbose=verbose)
            k = 4 if found else None
        nlay = sum(1 for t in ticks if t[0] == "ins")
        log.append(dict(iter=it, flags=list(chosen), layers=nlay, k=k, malignant=len(found)))
        if verbose:
            print(f"iter {it}: {len(chosen)} flags, {nlay} added layers -> first malignant weight {k} "
                  f"({len(found)} configurations) [{time.time()-t0:.0f}s]  flags={chosen}", flush=True)
        if checkpoint:
            open(checkpoint + f"_iter{it}.stim", "w").write(str(cur))
        if k is None:
            return dict(flags=flags, chosen=chosen, circuit=cur, tdem=m, log=log, candidates=len(cands))
        # tick of every fault, attribution to flags / inserted layers
        ops = list(m.noisy.flattened())
        sk_tick = []
        t = -1
        for op in ops:
            sk_tick.append(t)
            if op.name == "TICK":
                t += 1
        cur_index = {g: i for i, (kind, g) in enumerate(ticks) if kind == "base"}
        X = np.stack(ft.xframes)
        Zf = np.stack(ft.zframes)
        cq = np.array([cd[0][0] if cd[3] == "ZZ" else cd[0] for cd in cands])
        cq2 = np.array([cd[0][1] if cd[3] == "ZZ" else cd[0] for cd in cands])
        two = np.array([cd[3] == "ZZ" for cd in cands])
        ca = np.array([cur_index[cd[1]] for cd in cands])
        cb = np.array([cur_index[cd[2]] for cd in cands])
        isZ = np.array([cd[3] in ("Z", "ZZ") for cd in cands])
        flag_of_qubit = {n0 + i: cid[cd] for i, cd in enumerate(chosen)}

        def attrib(f):
            fl, gp_ = set(), set()
            for item in ft.desc[f]:
                if item[0] != "flip" and item[0] in flag_of_qubit:
                    fl.add(flag_of_qubit[item[0]])
            op = ops[ft.where[f]]
            if op.name in ("M", "MX") and any(x.value in flag_of_qubit for x in op.targets_copy()):
                fl.update(flag_of_qubit[x.value] for x in op.targets_copy() if x.value in flag_of_qubit)
            ct = sk_tick[ft.where[f]]
            if 0 <= ct < len(ticks) and ticks[ct][0] == "ins":
                gp_.add(ticks[ct][1])
            return fl, gp_

        added = undet = 0
        sig_cache = {}

        def signatures(ci):
            """distinct (detection vector over candidates, flag/layer attribution) over all placements of class ci:
            every placement with the same signature gives the same constraint, so covering all signature
            combinations is exact"""
            if ci in sig_cache:
                return sig_cache[ci]
            out = {}
            for fp in cl[ci][3]:
                px = X[ca, fp, cq] ^ X[cb, fp, cq] ^ np.where(two, X[ca, fp, cq2] ^ X[cb, fp, cq2], np.uint8(0)).astype(np.uint8)
                pz = Zf[ca, fp, cq] ^ Zf[cb, fp, cq]
                v = np.where(isZ, px, pz).astype(np.uint8)
                a_, b_ = attrib(fp)
                key = (v.tobytes(), frozenset(a_), frozenset(b_))
                out.setdefault(key, (v, a_, b_))
            sig_cache[ci] = list(out.values())
            return sig_cache[ci]

        for idx, o, acc, bad, prob in found:
            sigs = [signatures(i) for i in idx]
            for combo in itertools.islice(itertools.product(*sigs), max_reps):
                par = np.zeros(len(cands), dtype=np.uint8)
                cF, cG = set(), set()
                for v, a_, b_ in combo:
                    par ^= v
                    cF |= a_
                    cG |= b_
                det = frozenset(int(i) for i in np.nonzero(par)[0])
                if not det and not cF and not cG:
                    undet += 1
                    continue
                constraints.append((det, frozenset(cF), frozenset(cG)))
                added += 1
        constraints = list(set(constraints))
        # ILP
        mdl = gp.Model()
        mdl.Params.OutputFlag = 0
        mdl.Params.Threads = threads
        mdl.Params.TimeLimit = 600
        y = mdl.addVars(len(cands), vtype=GRB.BINARY)
        z = mdl.addVars(gaps, vtype=GRB.BINARY)
        z2 = mdl.addVars(gaps, vtype=GRB.BINARY)
        for i, (q, ga, gb, typ) in enumerate(cands):
            mdl.addConstr(y[i] <= z[ga])
            mdl.addConstr(y[i] <= z[gb])
        per = collections.defaultdict(list)
        z3 = mdl.addVars(gaps, vtype=GRB.BINARY)
        for i, (q, ga, gb, typ) in enumerate(cands):
            for x in (q if typ == "ZZ" else (q,)):
                per[(x, ga, "start")].append(i)
                per[(x, gb, "end")].append(i)
            if typ == "ZZ":
                mdl.addConstr(z3[ga] >= y[i])
                mdl.addConstr(z3[gb] >= y[i])
        for key, ii in per.items():
            mdl.addConstr(gp.quicksum(y[i] for i in ii) <= 1)
            if key[2] == "start" and (key[0], key[1], "end") in per:
                mdl.addConstr(z2[key[1]] >= gp.quicksum(y[i] for i in ii)
                              + gp.quicksum(y[i] for i in per[(key[0], key[1], "end")]) - 1)
        for det, cF, cG in constraints:
            mdl.addConstr(gp.quicksum(y[i] for i in det) >= 1 - gp.quicksum(1 - y[i] for i in cF)
                          - gp.quicksum(1 - z[g] for g in cG))
        mdl.setObjective(layer_weight * (z.sum() + z2.sum() + z3.sum()) + y.sum(), GRB.MINIMIZE)
        mdl.optimize()
        if mdl.SolCount == 0:
            print(f"   ILP infeasible ({len(constraints)} constraints, {undet} undetectable)", flush=True)
            return dict(flags=flags, chosen=chosen, circuit=cur, tdem=m, log=log, failed=True)
        new = [cands[i] for i in range(len(cands)) if y[i].X > 0.5]
        if verbose:
            print(f"   +{added} constraints (total {len(constraints)}, {undet} undetectable reps); ILP -> "
                  f"{len(new)} flags, objective {mdl.ObjVal:.0f}, gap {mdl.MIPGap:.2g}", flush=True)
        if sorted(map(str, new)) == sorted(map(str, chosen)):
            print("   ILP returned the same flags: undetectable configurations remain", flush=True)
            return dict(flags=flags, chosen=chosen, circuit=cur, tdem=m, log=log, failed=True)
        chosen = new
    return dict(flags=flags, chosen=chosen, circuit=cur, tdem=m, log=log, failed=True)
