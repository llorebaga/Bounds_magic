"""Exact real-T fault distance (to weight KMAX <= 4) with the Gauss-sum evaluator (src/branch.py FastExact).

Usage: python fd_gauss.py PATH KMAX [default|complement|nopattern]"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast
from branch import FastExact

path, kmax = sys.argv[1], int(sys.argv[2])
opt = sys.argv[3] if len(sys.argv) > 3 else "default"
kw = {"default": {}, "complement": {"complement": True}, "nopattern": {"pattern": False}}[opt]
t0 = time.time()
m = TDEM(stim.Circuit.from_file(path), magic="T", **kw)
mech = m.mechanisms()
fe = FastExact(m)
t1 = time.time()
print(f"{os.path.basename(path)}: t={m.t} k_affine={m.kaff} classes={len(mech)} accepted Z classes={len(fe.A)} "
      f"(setup {t1-t0:.0f}s)", flush=True)
if len(fe.A) != 2:
    print("  note: more than two accepted Z-pattern classes; FastExact sums over all of them", flush=True)
k, f = fault_distance_fast(fe, kmax, mech, verbose=True)
coef = sum(x[4] * x[3] for x in f if x[4] is not None) / 1e-3 ** k if k and k <= 3 else None
out = dict(circuit=os.path.basename(path), opt=opt, t=m.t, k_affine=m.kaff, classes=len(mech),
           fd=k if k else f">{kmax}", malignant=len(f), coef_at_p1e3=coef, evaluations=len(fe._cache),
           seconds=round(time.time() - t1, 1))
print(f"  RESULT real-T fault distance {out['fd']}, {len(f)} malignant"
      + (f" (c_{k} = {coef:.3g} if the file has p = 1e-3)" if coef else "")
      + f", {out['evaluations']} exact evaluations, {out['seconds']}s", flush=True)
print("JSON", json.dumps(out), flush=True)
