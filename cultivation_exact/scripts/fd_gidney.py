import sys, time; sys.path.insert(0, __file__.rsplit('scripts',1)[0] + 'src')
import stim
from tdem import TDEM
from fd import fault_distance
G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
d1 = sys.argv[1]; magic = sys.argv[2]; kmax = int(sys.argv[3])
f = G + ("c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim" if d1 == "3"
         else "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
t0 = time.time()
m = TDEM(stim.Circuit.from_file(f), magic=magic, verbose=True)
mech = m.mechanisms()
print(f"d1={d1} magic={magic}: t={m.t}, {len(mech)} mechanisms, {m.n_det} skeleton detectors")
k, bad, mech = fault_distance(m, kmax, mech)
coef = sum(b[4] * b[3] for b in bad)
print(f"fault distance = {k}; leading P_bad term = {coef:.4g} at p=1e-3 -> coefficient {coef/1e-3**k if k else 0:.4g} p^{k}; {time.time()-t0:.1f}s")
