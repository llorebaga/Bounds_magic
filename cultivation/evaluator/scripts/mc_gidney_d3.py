"""Real-T vs S-proxy logical error of Gidney's d1=3 inject+cultivate circuit (uniform noise), by TDEM sampling."""
import sys, time, json; sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from mc import sample, scale_noise
G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
c0 = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim")
out = []
for p in [2e-3, 1e-3, 5e-4, 2.5e-4]:
    shots = int(4e8 * (1e-3 / p) ** 2)
    c = scale_noise(c0, p / 1e-3)
    for magic in ["T", "S"]:
        t0 = time.time()
        r = sample(TDEM(c, magic=magic), shots)
        r.update(p=p, magic=magic, seconds=round(time.time() - t0))
        out.append(r)
        print(f"p={p:g} {magic}: P_acc={r['P_acc']:.4f} eps={r['eps']:.3e} +- {r['se_eps']:.1e} ({shots:.1e} shots, {r['seconds']}s)", flush=True)
json.dump(out, open(__file__.rsplit('scripts', 1)[0] + "results/mc_gidney_d3.json", "w"), indent=1)
