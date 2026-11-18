"""Exact S-proxy fault distance by integer programming on the stim detector error model of the proxy circuit (the
circuit read as written, S and S_DAG gates included): the least number of fault classes whose detector patterns
cancel and that flip a logical observable. Capped at CAP - 1 (default delta - 1): infeasibility certifies that the
proxy fault distance is at least CAP, which with the ideal decoder (an X_L or Z_L of weight delta on the data just
before it) makes it exactly delta.

Usage: python proxy_ilp.py KIND DELTA [inject_dag] [cap=K] [solver=gurobi|highs]   (KIND as in late_ilp_test.build)
       python proxy_ilp.py PATH.stim DELTA [cap=K] [solver=...]
Environment: ILP_SECONDS (time limit per observable), THREADS, GRB_PARAMS ("NodefileStart=200;NodefileDir=/path")."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim


def mechanisms(c):
    mech = set()
    for inst in c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True).flattened():
        if inst.type != "error":
            continue
        D, o = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                D ^= {t.val}
            elif t.is_logical_observable_id():
                o ^= {t.val}
        if D or o:
            mech.add((frozenset(D), frozenset(o)))
    return sorted(mech, key=lambda m: (len(m[0]), sorted(m[0]), sorted(m[1])))


def min_weight(mech, obs, cap, solver="gurobi", time_limit=3600, threads=0, params=None):
    """least number of mechanisms with cancelling detectors that flip observable obs, if below cap; else None"""
    for D, o in mech:
        if not D and obs in o:
            return 1, "optimal"
    by_det = {}
    for i, (D, o) in enumerate(mech):
        for d in D:
            by_det.setdefault(d, []).append(i)
    flip = [i for i, (D, o) in enumerate(mech) if obs in o]
    if solver == "gurobi":
        import gurobipy as gp
        from gurobipy import GRB
        m = gp.Model()
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = time_limit
        if threads:
            m.Params.Threads = threads
        for k, v in (params or {}).items():               # e.g. NodefileStart, NodefileDir
            m.setParam(k, v)
        y = m.addVars(len(mech), vtype=GRB.BINARY)
        for idx in by_det.values():
            m.addConstr(gp.quicksum(y[i] for i in idx) == 2 * m.addVar(vtype=GRB.INTEGER, lb=0))
        m.addConstr(gp.quicksum(y[i] for i in flip) == 2 * m.addVar(vtype=GRB.INTEGER, lb=0) + 1)
        obj = gp.quicksum(y[i] for i in range(len(mech)))
        m.addConstr(obj <= cap - 1)
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            return None, "infeasible"
        if not m.SolCount:
            return "unknown", str(m.Status)
        return round(m.ObjVal), "optimal" if m.Status == GRB.OPTIMAL else f"<= (status {m.Status})"
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    if threads:
        h.setOptionValue("threads", int(threads))
    y = [h.addBinary() for _ in mech]
    S = lambda vs: sum(vs[1:], vs[0])
    for idx in by_det.values():
        h.addConstr(S([y[i] for i in idx]) - 2 * h.addIntegral(lb=0, ub=10 ** 6) == 0)
    h.addConstr(S([y[i] for i in flip]) - 2 * h.addIntegral(lb=0, ub=10 ** 6) == 1)
    obj = S(y)
    h.addConstr(obj <= cap - 1)
    h.minimize(obj)
    st = h.getModelStatus()
    if st == highspy.HighsModelStatus.kInfeasible:
        return None, "infeasible"
    if h.getInfo().primal_solution_status != 2:
        return "unknown", str(st)
    return round(sum(h.val(t) for t in y)), "optimal" if st == highspy.HighsModelStatus.kOptimal else str(st)


if __name__ == "__main__":
    src, delta = sys.argv[1], int(sys.argv[2])
    opt = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
    cap = int(opt.get("cap", delta))
    solver = opt.get("solver", "gurobi")
    t0 = time.time()
    if src.endswith(".stim"):
        c, name = stim.Circuit.from_file(src), os.path.basename(src)
    else:
        from colorcode import color_code_faces
        from late_ilp_test import build
        n = color_code_faces(delta)[1]
        c, name = build(src, delta, list(range(n)), "inject_dag" in sys.argv), f"{src} delta={delta}"
    mech = mechanisms(c)
    nobs = c.num_observables
    res = dict(circuit=name, qubits=c.num_qubits, detectors=c.num_detectors, observables=nobs, classes=len(mech),
               cap=cap, solver=solver)
    best = None
    for j in range(nobs):
        params = {k: (v if k.endswith("Dir") else float(v)) for k, v in
                  (kv.split("=", 1) for kv in os.environ.get("GRB_PARAMS", "").split(";") if kv)}
        w, st = min_weight(mech, j, cap, solver, int(os.environ.get("ILP_SECONDS", 4 * 3600)), int(os.environ.get("THREADS", 0)),
                           params)
        res[f"obs{j}"] = dict(weight=w, status=st)
        if isinstance(w, int):
            best = w if best is None else min(best, w)
        elif w == "unknown":
            best = "unknown"
            break
    res["proxy_fd"] = best if best is not None else f">={cap}"
    res["seconds"] = round(time.time() - t0)
    print(json.dumps(res), flush=True)
