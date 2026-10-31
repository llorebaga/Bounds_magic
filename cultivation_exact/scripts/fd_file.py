"""Exact real-T fault distance (to weight kmax) of a .stim file.  Usage: python fd_file.py path kmax [pattern]"""
import sys, time
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast
from fd_list import harmful_k4
path, kmax = sys.argv[1], int(sys.argv[2])
pattern = (sys.argv[3] == "pattern") if len(sys.argv) > 3 else False
c = stim.Circuit.from_file(path)
t0 = time.time()
m = TDEM(c, magic="T", pattern=pattern)
mech = m.mechanisms()
print(path, "bloch", [round(x, 3) for x in m.output_bloch()], "t", m.t, "k_affine", m.k, "classes", len(mech), flush=True)
k, f = fault_distance_fast(m, min(kmax, 3), mech, verbose=True)
if k is None and kmax >= 4:
    f4 = harmful_k4(m, mech, verbose=True)
    k, f = (4, f4) if f4 else (None, [])
coef = sum(x[4] * x[3] for x in f) / 1e-3 ** k if k else 0
print(f"RESULT real-T fault distance {k if k else '>' + str(kmax)}, {len(f)} malignant, leading coefficient {coef:.4g} ({time.time()-t0:.0f}s)", flush=True)
