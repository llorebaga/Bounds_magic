"""Real-T fault distance with the branch screen (src/branch.py), compared with the exact evaluation.

Usage: python branch_fd.py PATH KMAX [default|complement|nopattern] [exact]
Prints FD_branch (a lower bound on the real-T fault distance, from the screen alone) and, with `exact`, the
exact fault distance found by the hybrid evaluator, with the number of exact contractions it needed."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast
from branch import BranchScreen, Hybrid

path, kmax = sys.argv[1], int(sys.argv[2])
opt = sys.argv[3] if len(sys.argv) > 3 else "default"
kw = {"default": {}, "complement": {"complement": True}, "nopattern": {"pattern": False}}[opt]
do_exact = len(sys.argv) > 4 and sys.argv[4] == "exact"

t0 = time.time()
m = TDEM(stim.Circuit.from_file(path), magic="T", **kw)
mech = m.mechanisms()
t1 = time.time()
scr = BranchScreen(m)
print(f"{os.path.basename(path)}: t={m.t} k_affine={m.kaff} classes={len(mech)} harmful Z patterns "
      f"{scr.n_harmful_patterns}/{len(scr.bad)} (TDEM {t1-t0:.0f}s, table {time.time()-t1:.1f}s)", flush=True)

t2 = time.time()
kb, fb = fault_distance_fast(scr, kmax, mech, verbose=False)
out = dict(circuit=os.path.basename(path), opt=opt, t=m.t, k_affine=m.kaff, classes=len(mech),
           fd_branch=kb if kb else f">{kmax}", n_branch=len(fb), sec_branch=round(time.time() - t2, 1))
print(f"  FD_branch = {out['fd_branch']} ({len(fb)} flagged configurations, {out['sec_branch']}s)", flush=True)

if do_exact:
    h = Hybrid(m)
    t3 = time.time()
    ke, fe = fault_distance_fast(h, kmax, mech, verbose=False)
    out.update(fd_exact=ke if ke else f">{kmax}", n_exact=len(fe), sec_exact=round(time.time() - t3, 1), **h.stats)
    print(f"  FD_real   = {out['fd_exact']} ({len(fe)} malignant, {out['sec_exact']}s; exact contractions "
          f"{h.stats['exact']}, cleared by screen {h.stats['cleared']}, flagged but cancelled {h.stats['cancelled']})",
          flush=True)
print("JSON", json.dumps(out), flush=True)
