"""Real-T logical error of Gidney's d1=5 circuit (or a modified .stim) by meet-in-the-middle completion.
Acceptance from short direct sampling. Usage: python mitm_d5.py [path] [tag]"""
import sys, time, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from mitm_rate import MITM
from mc import sample_parallel
from noise import si1000, uniform

if __name__ == "__main__":
    G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
    path = sys.argv[1] if len(sys.argv) > 1 else G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim"
    tag = sys.argv[2] if len(sys.argv) > 2 else "unflagged"
    models = sys.argv[3].split(",") if len(sys.argv) > 3 else ["uniform", "si1000"]
    c0 = stim.Circuit.from_file(path)
    SOFT = {2e-3: 3.41e-8, 1e-3: 4.59e-9, 5e-4: 1.57e-10}
    out = []
    for model in models:
        for p in [2e-3, 1e-3, 5e-4, 2.5e-4]:
            t0 = time.time()
            c = (si1000 if model == "si1000" else uniform)(c0, p)
            m = TDEM(c, magic="T", complement=True)
            pb, se, rows = MITM(m).P_bad(1.0, 8, 4000)
            acc = sample_parallel(c, "T", int(2e7), workers=15, chunks=30, complement=True)["P_acc"]
            r = dict(model=model, p=p, P_acc=acc, P_bad=pb, se_P_bad=se, eps=pb / acc, se_eps=se / acc,
                     strata=[(k, a, s) for k, a, s in rows], circuit=tag, seconds=round(time.time() - t0))
            if model == "uniform" and tag == "unflagged":
                r["SOFT"] = SOFT.get(p)
            out.append(r)
            print(f"{tag} {model} p={p:g}: acc {acc:.4f} eps {r['eps']:.3e} +- {r['se_eps']:.1e} SOFT {r.get('SOFT')} ({r['seconds']}s)", flush=True)
            json.dump(out, open(__file__.rsplit('scripts', 1)[0] + f"results/mitm_d5_{tag}_{'_'.join(models)}.json", "w"), indent=1)
