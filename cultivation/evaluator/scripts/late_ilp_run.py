"""Run the integer-programming certificate (late_ilp.py) on one grid check.
Usage: python late_ilp_run.py KIND DELTA [inject_dag] [ORDERFILE] [branch=XZC] [check]
KIND: endp, ladder or vcat (late_ilp_test.build); the data in natural order unless ORDERFILE (JSON with "order") is given.
branch= runs only those branches (one job per branch on the cluster); ILP_SECONDS and THREADS set the solver.
check runs the exact one-fault test first; it needs the Gauss-sum certifier, which runs out of memory at DELTA=9,
so there choose the injection from the sign sum of the T layer instead (T for sum 1 mod 8, T^dagger for 7).
GRB_PARAMS="NodefileStart=300,NodefileDir=/path" passes further Gurobi parameters."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from late_ilp import certify
from late_ilp_test import build

kind, delta = sys.argv[1], int(sys.argv[2])
dag = "inject_dag" in sys.argv
ofile = next((a for a in sys.argv[3:] if a.endswith(".json")), None)
branches = tuple(next((a[7:] for a in sys.argv[3:] if a.startswith("branch=")), "XZC"))
faces, n = color_code_faces(delta)
order = json.load(open(ofile))["order"] if ofile else list(range(n))
pos = {q: i for i, q in enumerate(order)}
lf = [[pos[q] for q in f] for f in faces]
t0 = time.time()
c = build(kind, delta, order, dag)
print(json.dumps(dict(kind=kind, delta=delta, inject_dag=dag, branches="".join(branches), qubits=c.num_qubits, ticks=c.num_ticks)), flush=True)
if "check" in sys.argv:                          # one fault must not already be harmful: the check measures the logical
    from nn_cat_fd import certify as exact
    fd1 = exact(c, 1, proxy=False)["T"]["fd"]
    print(json.dumps(dict(one_fault_fd=fd1)), flush=True)
    assert fd1 != 1, "wrong injected state: use the other one (inject_dag)"
r = certify(c, lf, n, delta, time_limit=int(os.environ.get("ILP_SECONDS", 4 * 3600)), threads=int(os.environ.get("THREADS", 0)),
            verbose="progress", max_cuts=5000, branches=branches,
            params={k: (v if k.endswith("Dir") else float(v)) for k, v in (kv.split("=", 1) for kv in os.environ.get("GRB_PARAMS", "").replace(",", ";").split(";") if kv)})
r["seconds"] = round(time.time() - t0)
print(json.dumps(r, default=str), flush=True)
