"""Real-T vs S-proxy logical error of Gidney's d1=5 inject+cultivate circuit, by parallel TDEM sampling.
Real T uses the complementary T/T^dag pattern (same logical check as at d=3) and the default branch, which is
the maximal-weight one (no X frame before the d=5 T layers).  Usage: python mc_d5.py model p shots"""
import sys, time, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from mc import sample_parallel
from noise import si1000, uniform

if __name__ == "__main__":
    G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
    c0 = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
    model, p, shots = sys.argv[1], float(sys.argv[2]), int(float(sys.argv[3]))
    c = si1000(c0, p) if model == "si1000" else uniform(c0, p)
    res = []
    for magic in ["T", "S"]:
        t0 = time.time()
        kw = {"complement": True} if magic == "T" else {}
        r = sample_parallel(c, magic, shots, workers=15, chunks=60, **kw)
        r.pop("raw")
        r.update(p=p, magic=magic, model=model, seconds=round(time.time() - t0))
        res.append(r)
        print(f"{model} p={p:g} {magic}: P_acc={r['P_acc']:.4f} P_bad={r['P_bad']:.3e} eps={r['eps']:.3e} +- {r['se_eps']:.1e} ({shots:.1e} shots, {r['seconds']}s)", flush=True)
        json.dump(res, open(__file__.rsplit('scripts', 1)[0] + f"results/mc_d5_{model}_{p:g}.json", "w"), indent=1)
