import sys, time, json; sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from faults import FaultTable
from fd_fast import fault_distance_fast
G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
d1, magic, kmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = sys.argv[4] if len(sys.argv) > 4 else G + ("c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim" if d1 == "3"
         else "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
t0 = time.time()
m = TDEM(stim.Circuit.from_file(path), magic=magic, verbose=True)
mech = m.mechanisms()
print(f"{path.split('/')[-1]} magic={magic}: t={m.t}, k_affine={m.k}, {len(mech)} mechanisms, {m.n_det} detectors", flush=True)
k, found = fault_distance_fast(m, kmax, mech)
coef = sum(f[4] * f[3] for f in found if f[4] is not None)
print(f"fault distance {'= ' + str(k) if k else '> ' + str(kmax)}; leading P_bad at p=1e-3: {coef:.4g} "
      f"(coefficient {coef / 1e-3 ** k if k else 0:.4g} p^{k}); {time.time() - t0:.0f}s", flush=True)
