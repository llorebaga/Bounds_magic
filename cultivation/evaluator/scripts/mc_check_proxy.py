"""Cross-check the TDEM sampler on the S proxy against direct stim sampling of Gidney's d1=3 circuit."""
import sys, time; sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import numpy as np, stim
from tdem import TDEM
from mc import sample, scale_noise
G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
c = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim")
c = scale_noise(c, 3.0)        # p = 3e-3 for statistics
shots = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000
t0 = time.time()
m = TDEM(c, magic="S")
r = sample(m, shots)
print("TDEM-S :", {k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()}, f"{time.time()-t0:.0f}s")
t0 = time.time()
s = c.compile_detector_sampler()
acc = bad = 0; n = 0
while n < shots:
    det, obs = s.sample(2_000_000, separate_observables=True, bit_packed=True)
    k = ~det.any(axis=1); acc += k.sum(); bad += (obs[k][:, 0] & 1).sum(); n += 2_000_000
print(f"direct : P_acc={acc/n:.4g} P_bad={bad/n:.4g} eps={bad/acc:.4g} +- {np.sqrt(bad)/acc:.2g}  {time.time()-t0:.0f}s")
