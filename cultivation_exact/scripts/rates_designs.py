"""Exact eps (meet-in-the-middle) and acceptance for a list of circuits at uniform / SI1000 p in {1e-3, 5e-4}.
Usage: python rates_designs.py tag path"""
import sys, json, time
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from mitm_rate import MITM
from mc import sample_parallel
from noise import uniform, si1000

if __name__ == "__main__":
    tag, path = sys.argv[1], sys.argv[2]
    kmax = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    c0 = stim.Circuit.from_file(path)
    tkw = {"pattern": False} if (len(sys.argv) > 6 and sys.argv[6] == "nopattern") else {"complement": True}
    out = []
    models = sys.argv[7].split(",") if len(sys.argv) > 7 else ["uniform", "si1000"]
    for model in models:
        m = TDEM((si1000 if model == "si1000" else uniform)(c0, 1e-3), magic="T", **tkw)
        est = MITM(m)
        for p in [1e-3, 5e-4]:
            t0 = time.time()
            pb, se, rows = est.P_bad(p / 1e-3, kmax, int(sys.argv[4]) if len(sys.argv) > 4 else 3000)
            c = (si1000 if model == "si1000" else uniform)(c0, p)
            acc = sample_parallel(c, "T", int(5e6), workers=4, chunks=8, **tkw)["P_acc"]
            out.append(dict(design=tag, model=model, p=p, P_acc=acc, eps=pb / acc, se_eps=se / acc,
                            strata=[(k, a, s) for k, a, s in rows]))
            print(f"{tag} {model} p={p:g}: acc {acc:.4f} eps {pb/acc:.3e} +- {se/acc:.1e} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(out, open(__file__.rsplit('scripts', 1)[0] + f"results/rates_{tag}{sys.argv[5] if len(sys.argv) > 5 else ''}.json", "w"), indent=1)
