"""Fault distance of Gidney's inject+cultivate circuits with the protocol's Pauli frame (feedback) included."""
import sys, time, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM, find_frames
from fd_fast import fault_distance_fast

G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
NAMES = {"3": "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim",
         "5": "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim"}
d1, magic, kmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = stim.Circuit.from_file(G + NAMES[d1])
t0 = time.time()
fr = find_frames(c, magic, verbose=True)
m = TDEM(c, magic=magic, feedback=fr)
k, found = fault_distance_fast(m, kmax, m.mechanisms())
coef = sum(f[4] * f[3] for f in found if f[4] is not None)
res = dict(d1=d1, magic=magic, frames={str(a): b for a, b in fr.items()}, fault_distance=k, n_malignant=len(found),
           coef=(coef / 1e-3 ** k if k else None), seconds=round(time.time() - t0))
print(json.dumps(res))
