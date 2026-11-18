"""The integer-programming certificate of late_ilp.py, solved with HiGHS instead of Gurobi: an independent check of
the solver behind the certificates. The pieces and the exact test (late_ilp.pieces, late_ilp.exact_harm) are shared,
and the model is the same (late-error branches X, Z and C, objective capped at delta - 1, harmless solutions cut off).

Usage: python late_ilp_highs.py KIND DELTA [inject_dag] [branch=XZC]   (KIND as in late_ilp_test.build)
Environment: ILP_SECONDS (time limit per solve), THREADS."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import highspy
from late_ilp import pieces, exact_harm


def certify(circuit, faces, n, delta, time_limit=3600, threads=0, verbose=False, max_cuts=2000, branches=("X", "Z", "C")):
    P, data = pieces(circuit)
    assert len(data) == n
    dets = sorted({d for p in P for d in p[3]})
    di = {d: i for i, d in enumerate(dets)}
    res = dict(solver="HiGHS " + highspy.Highs().version(), pieces=len(P), detectors=len(dets), cuts=0)
    best, wit = delta, None
    by_det = {}
    for i, p in enumerate(P):
        for d in p[3]:
            by_det.setdefault(di[d], []).append(i)
    for branch in branches:
        h = highspy.Highs()
        h.setOptionValue("output_flag", verbose is True)
        h.setOptionValue("time_limit", float(time_limit))
        if threads:
            h.setOptionValue("threads", int(threads))
        B = lambda: h.addBinary()
        y = [B() for _ in P]
        x = [B() for _ in range(n)]
        z = [B() for _ in range(n)]
        u = [B() for _ in range(n)]
        a = [B() for _ in range(n)]
        ap = [B() for _ in range(n)]
        bz = [B() for _ in range(n)]
        w = [B() for _ in range(n)]
        g = B()
        S = lambda vs: sum(vs[1:], vs[0])

        def par(expr, odd=0):
            k = h.addIntegral(lb=0, ub=10 ** 6)
            h.addConstr(expr - 2 * k == odd)

        for q in range(n):
            h.addConstr(u[q] - x[q] >= 0); h.addConstr(u[q] - z[q] >= 0); h.addConstr(w[q] - a[q] <= 0)
        for d, idx in by_det.items():
            par(S([y[i] for i in idx]))
        for q in range(n):
            bit = 1 << q
            for j, extra, out in ((0, g, a[q]), (1, x[q], ap[q]), (2, z[q], bz[q])):
                par(S([y[i] for i, p in enumerate(P) if p[j] & bit] + [extra]) - out)
        h.addConstr(S(a) <= (n - 1) // 2)
        for f in faces:
            par(S([a[q] for q in f] + [ap[q] for q in f]))
            par(S([bz[q] for q in f] + [w[q] for q in f]))
        sumT = S(a + ap)
        sumZ = S(bz + w)
        if branch == "X":
            par(sumT, 1)
        else:
            par(sumT)
            if branch == "Z":
                par(sumZ, 1)
            else:
                v = [B() for _ in range(n)]
                for q in range(n):
                    h.addConstr(v[q] - a[q] <= 0)
                for f in faces:
                    par(S([v[q] for q in f]))
                par(S(v), 1)
        obj = S(y + u)
        h.addConstr(obj <= best - 1)
        h.minimize(obj)
        found = "unknown"
        for it in range(max_cuts):
            if it:
                h.run()
            st = h.getModelStatus()
            if st == highspy.HighsModelStatus.kInfeasible:
                found = None
                break
            if h.getInfo().primal_solution_status != 2:            # no feasible solution
                res[f"{branch}_status"] = str(st)
                break
            val = lambda var: h.val(var) > 0.5
            A = sum(1 << q for q in range(n) if val(a[q]))
            Ap = sum(1 << q for q in range(n) if val(ap[q]))
            Bm = sum(1 << q for q in range(n) if val(bz[q]))
            objv = round(sum(h.val(t) for t in y) + sum(h.val(t) for t in u))
            if verbose == "progress":
                print(f"  {branch} it {it} obj {objv} |A| {bin(A).count('1')} |A'| {bin(Ap).count('1')} "
                      f"|B| {bin(Bm).count('1')} status {st}", flush=True)
            if exact_harm(A, Ap, Bm, faces, n):
                found = objv if st == highspy.HighsModelStatus.kOptimal else f"<={objv}"
                wit = dict(branch=branch, A=A, Ap=Ap, B=Bm)
                break
            res["cuts"] += 1
            lits = []
            for vec, var in ((A, a), (Ap, ap), (Bm, bz)):
                for q in range(n):
                    lits.append((1 - var[q]) if vec >> q & 1 else var[q])
            h.addConstr(S(lits) >= 1)
        if found == "unknown":
            res[branch] = "unknown"
            best = None
            break
        res[branch] = found if found is not None else f">={best}"
        if isinstance(found, str):
            best = None
            break
        best = found if isinstance(found, int) else best
    res["bound"] = best
    res["witness"] = wit
    return res


if __name__ == "__main__":
    from colorcode import color_code_faces
    from late_ilp_test import build
    kind, delta = sys.argv[1], int(sys.argv[2])
    dag = "inject_dag" in sys.argv
    branches = tuple(next((a[7:] for a in sys.argv[3:] if a.startswith("branch=")), "XZC"))
    faces, n = color_code_faces(delta)
    t0 = time.time()
    c = build(kind, delta, list(range(n)), dag)
    print(json.dumps(dict(kind=kind, delta=delta, inject_dag=dag, branches="".join(branches), qubits=c.num_qubits)), flush=True)
    r = certify(c, faces, n, delta, time_limit=int(os.environ.get("ILP_SECONDS", 4 * 3600)),
                threads=int(os.environ.get("THREADS", 0)), verbose="progress", branches=branches)
    r["seconds"] = round(time.time() - t0)
    print(json.dumps(r, default=str), flush=True)
