"""Real-T fault distance (to weight 3) of the distance-five RP^2 circuit in every branch of its eight random records,
one share of the 256 branches per task.  Usage: python rp2_branches_job.py TASK NTASKS
Prints one JSON line per branch: {"branch": {record: value}, "fd": .., "malignant": .., "coef": .., "seconds": ..}."""
import sys, os, time, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from lean_exact import LeanTDEM, LeanExact
from fd_fast import fault_distance_fast

task, ntasks = int(sys.argv[1]), int(sys.argv[2])
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "rp2_circuits", "rp_3_rp_5_T_cult.stim")
c = stim.Circuit.from_file(path)
FREE = [2, 5, 39, 40, 42, 49, 50, 51]                   # free records of the reference run (rp2_fd_lean.py)
branches = list(itertools.product((0, 1), repeat=len(FREE)))
for b in branches[task::ntasks]:
    ov = dict(zip(FREE, b))
    t0 = time.time()
    try:
        m = LeanTDEM(c, magic="T", pattern=False, zl_from_obs=True, overrides=ov)
        k, f = fault_distance_fast(LeanExact(m), 3, m.mechanisms(), verbose=False)
        coef = sum(x[4] * x[3] for x in f if x[4] is not None) / 1e-3 ** k if k else None
        out = dict(branch=ov, fd=k if k else ">3", malignant=len(f), coef=coef, A=len(m.A),
                   bloch=[round(float(x), 4) for x in m.output_bloch()], seconds=round(time.time() - t0))
    except Exception as e:                               # an inconsistent branch is reported, not fatal
        out = dict(branch=ov, error=repr(e)[:200], seconds=round(time.time() - t0))
    print(json.dumps(out), flush=True)
