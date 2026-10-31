"""Real-T and S-proxy fault distance of a TDEM-ready circuit (no sublattice pattern), with the leading coefficient.
Usage: python google_fd.py PATH KMAX"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast
from branch import FastExact

path, kmax = sys.argv[1], int(sys.argv[2])
c = stim.Circuit.from_file(path)
out = dict(circuit=os.path.basename(path))
for magic in ("T", "S"):
    t0 = time.time()
    m = TDEM(c, magic=magic, pattern=False)
    mech = m.mechanisms()
    ev = FastExact(m) if magic == "T" else m
    k, f = fault_distance_fast(ev, kmax, mech, verbose=False)
    coef = sum(x[4] * x[3] for x in f if x[4] is not None) / 1e-3 ** k if k else None
    out[magic] = dict(fd=k if k else f">{kmax}", malignant=len(f), coef_at_p1e3=coef, t=m.t, classes=len(mech),
                      seconds=round(time.time() - t0, 1))
    print(magic, out[magic], flush=True)
print("JSON", json.dumps(out), flush=True)
