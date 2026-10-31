"""d=5 real T (complementary pattern): exact leading-order P_bad (fault distance 3) and acceptance by short MC,
uniform and SI1000 noise; eps_lead = c3 p^3 / P_acc."""
import sys, time, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast
from mc import sample_parallel
from noise import si1000, uniform

if __name__ == "__main__":
    G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
    c0 = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
    out = []
    for model in ["uniform", "si1000"]:
        c1 = (si1000 if model == "si1000" else uniform)(c0, 1e-3)
        m = TDEM(c1, magic="T", complement=True)
        k, found = fault_distance_fast(m, 3, m.mechanisms(), verbose=False)
        c3 = sum(f[4] * f[3] for f in found) / 1e-3 ** k
        for p in [1e-3, 5e-4]:
            c = (si1000 if model == "si1000" else uniform)(c0, p)
            r = sample_parallel(c, "T", int(3e7), workers=15, chunks=30, complement=True)
            rec = dict(model=model, p=p, fault_distance=k, n_malignant=len(found), c3=c3, P_acc=r["P_acc"],
                       eps_leading=c3 * p ** 3 / r["P_acc"])
            out.append(rec)
            print(json.dumps(rec), flush=True)
    json.dump(out, open(__file__.rsplit('scripts', 1)[0] + "results/d5_leading.json", "w"), indent=1)
