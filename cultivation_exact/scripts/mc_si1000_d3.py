"""Real-T vs S-proxy logical error of Gidney's d1=3 inject+cultivate circuit under SI1000 noise (the
superconducting-inspired model used for Google hardware projections), by TDEM sampling (parallel)."""
import sys, time, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from mc import sample_parallel
from noise import si1000

if __name__ == "__main__":
    G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
    c0 = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim")
    import os
    path = __file__.rsplit('scripts', 1)[0] + "results/mc_si1000_d3.json"
    out = json.load(open(path)) if os.path.exists(path) else []
    plist = [float(x) for x in sys.argv[1:]] or [2e-3, 1.5e-3, 1e-3, 5e-4]
    for p in plist:
        c = si1000(c0, p)
        shots = int(8e8 * (1e-3 / p) ** 2)
        for magic in ["T", "S"]:
            t0 = time.time()
            r = sample_parallel(c, magic, shots, workers=8)
            r.pop("raw")
            r.update(p=p, magic=magic, model="SI1000", seconds=round(time.time() - t0))
            out.append(r)
            print(f"SI1000 p={p:g} {magic}: P_acc={r['P_acc']:.4f} eps={r['eps']:.3e} +- {r['se_eps']:.1e} ({shots:.1e} shots, {r['seconds']}s)", flush=True)
            json.dump(out, open(path, "w"), indent=1)
