"""The integer-programming certificate (late_ilp.py) on a circuit file, such as a rescheduled check with its resets
and measurements gathered (results/didle/nn_*_w8d_didle.stim), with the data in natural order.
Usage: python late_ilp_file.py PATH DELTA [branch=XZC]
Environment: ILP_SECONDS, THREADS, GRB_PARAMS ("NodefileStart=200;NodefileDir=/path")."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from colorcode import color_code_faces
from late_ilp import certify

path, delta = sys.argv[1], int(sys.argv[2])
branches = tuple(next((a[7:] for a in sys.argv[3:] if a.startswith("branch=")), "XZC"))
faces, n = color_code_faces(delta)
t0 = time.time()
c = stim.Circuit.from_file(path)
print(json.dumps(dict(circuit=os.path.basename(path), delta=delta, branches="".join(branches), qubits=c.num_qubits)), flush=True)
r = certify(c, faces, n, delta, time_limit=int(os.environ.get("ILP_SECONDS", 4 * 3600)), threads=int(os.environ.get("THREADS", 0)),
            verbose="progress", max_cuts=5000, branches=branches,
            params={k: (v if k.endswith("Dir") else float(v)) for k, v in (kv.split("=", 1) for kv in os.environ.get("GRB_PARAMS", "").split(";") if kv)})
r["seconds"] = round(time.time() - t0)
print(json.dumps(r, default=str), flush=True)
