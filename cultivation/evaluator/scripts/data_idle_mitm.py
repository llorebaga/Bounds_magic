"""Logical error per accepted state, to all orders up to eight faults, of circuits in results/didle/.

P_bad from the rare-event estimator of src/mitm_rate.py (exact for two and three fault classes, sampled for four
to eight), P_acc from TDEM sampling, both under the given noise model at the given p.
Usage: python data_idle_mitm.py MODEL P[,P...] SHOTS SAMPLES STEM ...   -> results/didle/mitm.json
(SAMPLES: draws per stratum of four to eight faults)"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from mitm_rate import MITM
from mc import sample
from noise import si1000, uniform

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "didle")

if __name__ == "__main__":
    model, ps, shots, samples = sys.argv[1], [float(x) for x in sys.argv[2].split(",")], int(float(sys.argv[3])), int(float(sys.argv[4]))
    path = os.path.join(OUT, "mitm.json")
    for stem, p in [(stem, p) for stem in sys.argv[5:] for p in ps]:
        t0 = time.time()
        c0 = stim.Circuit.from_file(os.path.join(OUT, stem + ".stim"))
        c = si1000(c0, p) if model == "si1000" else uniform(c0, p)
        m = TDEM(c, magic="T", pattern=False)
        pb, se, rows = MITM(m).P_bad(1.0, 8, samples)
        acc = sample(TDEM(c, magic="T", pattern=False), shots, seed=1)["P_acc"]
        row = dict(circuit=stem, model=model, p=p, samples=samples, P_bad=pb, se_P_bad=se, P_acc=acc, eps=pb / acc,
                   se_eps=se / acc, strata=[(k, a, s) for k, a, s in rows], seconds=round(time.time() - t0))
        res = json.load(open(path)) if os.path.exists(path) else []
        res = [x for x in res if (x["circuit"], x["model"], x["p"]) != (stem, model, p)] + [row]
        json.dump(res, open(path, "w"), indent=1)
        print(json.dumps({k: v for k, v in row.items() if k != "strata"}), flush=True)
