"""Exact real-T fault distance of the RP^2 cultivation circuits with the lean evaluator (src/lean_exact.py), for
skeletons whose affine dimension is too large for TDEM._affine (the d=5 circuit has k = 27).
Usage: python rp2_fd_lean.py FILE KMAX   (FILE in ../external/rp2_circuits/)"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from lean_exact import LeanTDEM, LeanExact
from fd_fast import fault_distance_fast

name, kmax = sys.argv[1], int(sys.argv[2])
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "rp2_circuits", name)
t0 = time.time()
m = LeanTDEM(stim.Circuit.from_file(path), magic="T", pattern=False, zl_from_obs=True)
mech = m.mechanisms()
le = LeanExact(m)
print(f"{name}: t={m.t} k={m.kaff} nonzero table indices A={list(m.A)} Bloch {[round(x, 4) for x in m.output_bloch()]} "
      f"classes={len(mech)} free records {m.free_records} (setup {time.time() - t0:.0f}s)", flush=True)
k, f = fault_distance_fast(le, kmax, mech, verbose=True)
coef = sum(x[4] * x[3] for x in f if x[4] is not None) / 1e-3 ** k if k else None
print("RESULT", json.dumps(dict(circuit=name, fd=k if k else f">{kmax}", malignant=len(f),
                               coef_at_p1e3=coef, seconds=round(time.time() - t0))), flush=True)
