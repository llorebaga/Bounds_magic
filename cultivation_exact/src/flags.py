"""Nearest-neighbour Z-flag design for cultivation circuits, certified with real magic gates by the TDEM.

A flag (q, ga, gb): a flag qubit f on a grid site next to q, CX(q -> f) in new layers inserted right after base
ticks ga and gb; f is reset in the last reset tick <= ga and measured (Z, noisy) in the first measurement tick
>= gb + 1, idling in between.  It flips iff the X part of the Pauli frame on q differs at the two boundaries.

Design loop (counterexample-guided, as in REPORT section 9 of the first study, now on the full protocol):
  1. TDEM + fault table of the current circuit; malignant fault-class configurations of weight < target;
  2. candidates = (q, ga, gb) whose parity Z_q(ga) Z_q(gb) is deterministic in the skeleton (non-disturbing);
  3. ILP (Gurobi): every physical representative of every malignant configuration must flip a chosen flag;
     minimise  layer_weight * (#inserted layers) + #flags,  at most one flag per (qubit, layer);
  4. insert the flags with noise, re-certify exactly, repeat.
"""
import collections
import itertools
import numpy as np
import stim

from tdem import TDEM
from faults import FaultTable
from fd import fault_distance


def split_ticks(c):
    ticks = [[]]
    for inst in c.flattened():
        if inst.name == "TICK":
            ticks.append([])
        else:
            ticks[-1].append(inst)
    return ticks


def tick_kinds(ticks):
    reset = [any(i.name in ("R", "RX") for i in tk) for tk in ticks]
    meas = [any(i.name in ("M", "MX") and i.gate_args_copy() for i in tk) for tk in ticks]
    return reset, meas


def build_flagged(c, flags, p):
    """c: base circuit (uniform noise p); flags: list of (q, ga, gb, f) with f a new qubit index.
    Returns (circuit, n_added_layers)."""
    ticks = split_ticks(c)
    reset, meas = tick_kinds(ticks)
    nq = c.num_qubits
    gaps = collections.defaultdict(list)
    plan = []
    for fl in flags:
        q, ga, gb, f = fl[:4]
        typ = fl[4] if len(fl) > 4 else "Z"
        r = max(t for t in range(ga + 1) if reset[t])
        mm = [t for t in range(gb + 1, len(ticks)) if meas[t]]
        mt = mm[0] if mm else len(ticks) - 1
        plan.append((q, ga, gb, f, r, mt, typ))
        pair = (q, f) if typ == "Z" else (f, q)
        gaps[ga].append((pair, "start", q))
        gaps[gb].append((pair, "end", q))
    allq = list(range(nq)) + [fl[3] for fl in flags]
    out = stim.Circuit()
    n_layers = 0
    old = new = 0
    remap = {}
    for t, tk in enumerate(ticks):
        for inst in tk:
            nm = inst.name
            if nm in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                tg = [stim.target_rec(remap[old + x.value] - new) for x in inst.targets_copy()]
                out.append(nm, tg, inst.gate_args_copy())
                continue
            k = len(inst.targets_copy()) if nm in ("M", "MX", "MY") else (len(inst.target_groups()) if nm == "MPP" else 0)
            for j in range(k):
                remap[old + j] = new + j
            old += k
            new += k
            out.append(inst)
        for (q, ga, gb, f, r, mt, typ) in plan:
            if t == r:
                out.append("R" if typ == "Z" else "RX", [f])
                out.append("X_ERROR" if typ == "Z" else "Z_ERROR", [f], [p])
            elif r < t < mt:
                out.append("DEPOLARIZE1", [f], [p])
            elif t == mt:
                out.append("M" if typ == "Z" else "MX", [f], [p]); out.append("DETECTOR", [stim.target_rec(-1)])
                new += 1
        if t < len(ticks) - 1:
            out.append("TICK")
        if t in gaps:
            items = gaps[t]
            ends = [it for it in items if it[1] == "end"]
            starts = [it for it in items if it[1] == "start"]
            if {it[2] for it in ends} & {it[2] for it in starts}:
                layers = [ends, starts]           # a flag ending on q acts before one starting on q
            else:
                layers = [items]
            for lay in layers:
                pairs = [it[0] for it in lay]
                busy = {x for pr in pairs for x in pr}
                out.append("CX", [x for pr in pairs for x in pr])
                out.append("DEPOLARIZE2", [x for pr in pairs for x in pr], [p])
                live = [x for x in allq if x not in busy and
                        not any(x == f and not (r <= t < mt) for (q, ga, gb, f, r, mt, typ) in plan)]
                out.append("DEPOLARIZE1", live, [p])
                out.append("TICK")
                n_layers += 1
    return out, n_layers


def deterministic_flags(m, cands):
    """keep candidates (q, ga, gb) whose flag outcome is deterministic in the noiseless skeleton of TDEM m"""
    ops = [op for op in m.ops if op.name not in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR")]
    # skeleton position right after original TICK number g
    tick_pos = []
    for i, inst in enumerate(m.insts):
        if inst.name == "TICK":
            tick_pos.append(m.pos_after[i])
    # pos_after counts noise ops too: map to positions in the noise-free list
    keep_idx = [i for i, op in enumerate(m.ops) if op.name not in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR")]
    import bisect
    clean_pos = lambda p: bisect.bisect_left(keep_idx, p)
    f = m.nq
    good = []
    for (q, ga, gb, typ) in cands:
        pa, pb = clean_pos(tick_pos[ga]), clean_pos(tick_pos[gb])
        sim = stim.TableauSimulator()
        sim.set_num_qubits(f + 1)
        if typ == "X":
            sim.h(f)
        for i, op in enumerate(ops):
            if i == pa or i == pb:
                if typ == "Z":
                    sim.cnot(q, f)
                elif typ == "ZZ":
                    sim.cnot(q[0], f)
                    sim.cnot(q[1], f)
                else:
                    sim.cnot(f, q)
                if i == pb:
                    break
            nm = op.name
            if nm in ("M", "MX", "MPP"):
                sim.do(stim.CircuitInstruction(nm, op.targets_copy()))
            else:
                sim.do(op)
        if (sim.peek_z(f) if typ in ("Z", "ZZ") else sim.peek_x(f)) != 0:
            good.append((q, ga, gb, typ))
    return good


def grid_neighbours(coords, q):
    x, y = coords[q][:2]
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def design(c, p, d_target, magic="T", region=None, layer_weight=10.0, max_iter=6, max_reps=4096, verbose=True,
           qubits=None, types=("Z", "X")):
    import gurobipy as gp
    from gurobipy import GRB
    coords = {q: tuple(v) for q, v in c.get_final_qubit_coordinates().items()}
    n0 = c.num_qubits
    flags = []
    log = []
    for it in range(max_iter):
        cur, nlay = build_flagged(c, flags, p)
        m = TDEM(cur, magic=magic)
        ft = FaultTable(m, frames=True)
        cl = ft.classes()
        mech = [(pp, D, o) for pp, D, o, fs in cl]
        k, mal, _ = fault_distance(m, d_target - 1, mech, verbose=False)
        log.append(dict(iter=it, flags=len(flags), layers=nlay, k=k, malignant=len(mal)))
        if verbose:
            print(f"iter {it}: {len(flags)} flags, {nlay} added layers -> first malignant weight {k} "
                  f"({len(mal)} configurations)", flush=True)
        if k is None:
            return flags, log, cur, m
        # base tick index of every tick of the current circuit (inserted layers are not candidates)
        T = len(ft.xframes)
        ticks_cur = split_ticks(cur)
        base_of = []
        b = -1
        ins = {g for fl in flags for g in (fl[1], fl[2])}
        # walk: current tick list = base ticks with inserted layers after the base ticks in `ins`
        nsub = collections.Counter()
        for g in ins:
            ends = {fl[0] for fl in flags if fl[2] == g}
            starts = {fl[0] for fl in flags if fl[1] == g}
            nsub[g] = 2 if ends & starts else 1
        for g in range(len(split_ticks(c))):
            base_of.append(("base", g))
            for _ in range(nsub[g]):
                base_of.append(("ins", g))
        cur_index = {bo[1]: i for i, bo in enumerate(base_of) if bo[0] == "base"}
        lo, hi = region if region else (0, len(split_ticks(c)) - 2)
        qs = qubits if qubits is not None else list(range(n0))
        cands = [(q, ga, gb, typ) for q in qs for ga, gb in itertools.combinations(range(lo, hi + 1), 2)
                 for typ in types]
        cands_cur = [(q, cur_index[ga], cur_index[gb], typ) for (q, ga, gb, typ) in cands]
        ok = set(deterministic_flags(m, cands_cur))
        cands = [cb for cb, cc in zip(cands, cands_cur) if cc in ok]
        cands_cur = [cc for cc in cands_cur if cc in ok]
        if verbose:
            print(f"   {len(cands)} non-disturbing candidates", flush=True)
        # detection bits per physical fault
        X = np.stack(ft.xframes)                     # (ticks, N, n0)
        Zf = np.stack(ft.zframes)
        isZ = np.array([cc[3] == "Z" for cc in cands_cur])
        cq = np.array([cc[0] for cc in cands_cur])
        ca = np.array([cc[1] for cc in cands_cur])
        cb_ = np.array([cc[2] for cc in cands_cur])
        model = gp.Model()
        model.Params.OutputFlag = 0
        model.Params.Threads = 8
        y = model.addVars(len(cands), vtype=GRB.BINARY)
        gaps = sorted({g for (q, ga, gb, typ) in cands for g in (ga, gb)})
        z = model.addVars(gaps, vtype=GRB.BINARY)
        for i, (q, ga, gb, typ) in enumerate(cands):
            model.addConstr(y[i] <= z[ga]); model.addConstr(y[i] <= z[gb])
        per_qg = collections.defaultdict(list)
        for i, (q, ga, gb, typ) in enumerate(cands):
            per_qg[(q, ga, "start")].append(i); per_qg[(q, gb, "end")].append(i)
        usedr = {(fl[0], fl[1], "start") for fl in flags} | {(fl[0], fl[2], "end") for fl in flags}
        for key, ii in per_qg.items():
            model.addConstr(gp.quicksum(y[i] for i in ii) <= (0 if key in usedr else 1))
        z2 = model.addVars(gaps, vtype=GRB.BINARY)
        for (q, g, role), ii in per_qg.items():
            if role == "start" and (q, g, "end") in per_qg:
                model.addConstr(z2[g] >= gp.quicksum(y[i] for i in ii) + gp.quicksum(y[i] for i in per_qg[(q, g, "end")]) - 1)
        ncons = undetectable = 0
        for idx, o, acc, bad, prob in mal:
            reps = [cl[i][3] for i in idx]
            for rep in itertools.islice(itertools.product(*reps), max_reps):
                par = np.zeros(len(cands), dtype=np.uint8)
                for fphys in rep:
                    px = X[ca, fphys, cq] ^ X[cb_, fphys, cq]
                    pz = Zf[ca, fphys, cq] ^ Zf[cb_, fphys, cq]
                    par ^= np.where(isZ, px, pz)
                det = np.nonzero(par)[0]
                if len(det) == 0:
                    undetectable += 1
                    continue
                model.addConstr(gp.quicksum(y[int(i)] for i in det) >= 1)
                ncons += 1
        model.setObjective(layer_weight * gp.quicksum(z[g] for g in gaps if not any(g in (fl[1], fl[2]) for fl in flags))
                           + layer_weight * gp.quicksum(z2.values())
                           + gp.quicksum(y.values()), GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            print(f"   ILP infeasible ({ncons} constraints, {undetectable} undetectable representatives)")
            model.computeIIS()
            for cst in model.getConstrs():
                if cst.IISConstr:
                    row = model.getRow(cst)
                    print("     IIS:", [cands[int(row.getVar(j).index)] for j in range(row.size()) if row.getVar(j).index < len(cands)], cst.Sense, cst.RHS)
            return flags, log, cur, m
        new = [cands[i] for i in range(len(cands)) if y[i].X > 0.5]
        if verbose:
            print(f"   ILP: {ncons} constraints ({undetectable} undetectable representatives) -> {len(new)} new flags {new}", flush=True)
        if not new:
            return flags, log, cur, m
        for (q, ga, gb, typ) in new:
            flags.append((q, ga, gb, n0 + len(flags), typ))
    return flags, log, cur, m
