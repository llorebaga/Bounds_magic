"""Rare-event estimator (src/mitm_rate.py) against the sampled real-T logical error of the d1=3 circuit of Gidney et
al. under uniform noise (results/mc_gidney_d3.json, from mc_gidney_d3.py). Usage: python mitm_d3_check.py"""
import sys, time, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from mitm_rate import MITM
from mc import scale_noise

if __name__ == "__main__":
    ROOT = __file__.rsplit("scripts", 1)[0]
    c0 = stim.Circuit.from_file(ROOT + "external/gidney_circuits/c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim")
    sampled = {r["p"]: r for r in json.load(open(ROOT + "results/mc_gidney_d3.json")) if r["magic"] == "T"}
    out = []
    for p in [2e-3, 1e-3, 5e-4, 2.5e-4]:
        t0 = time.time()
        m = TDEM(scale_noise(c0, p / 1e-3), magic="T")
        pb, se, rows = MITM(m).P_bad(1.0, 8, 4000)
        s = sampled[p]
        r = dict(p=p, P_bad_mitm=pb, se_mitm=se, P_bad_sampled=s["P_bad"], se_sampled=s["se_P_bad"],
                 strata=[(k, a, sd) for k, a, sd in rows], seconds=round(time.time() - t0))
        out.append(r)
        print(f"p={p:g}: MITM P_bad {pb:.4e} +- {se:.1e}   sampled {s['P_bad']:.4e} +- {s['se_P_bad']:.1e}  ({r['seconds']} s)", flush=True)
    json.dump(out, open(ROOT + "results/mitm_d3_check.json", "w"), indent=1)
