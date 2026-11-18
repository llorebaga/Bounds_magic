"""Acceptance relative to the fault-free run of circuits in results/didle/, by TDEM sampling.

Usage: python data_idle_acc.py SHOTS MODEL P[,P...] STEM ...   (MODEL uniform or si1000)
Appends to results/didle/acceptance.json (one row per circuit, model and p, with the number of accepted samples)."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from mc import sample
from noise import si1000, uniform

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "didle")

if __name__ == "__main__":
    shots, model, ps = int(float(sys.argv[1])), sys.argv[2], [float(x) for x in sys.argv[3].split(",")]
    path = os.path.join(OUT, "acceptance.json")
    for stem in sys.argv[4:]:
        c0 = stim.Circuit.from_file(os.path.join(OUT, stem + ".stim"))
        for p in ps:
            t0 = time.time()
            c = si1000(c0, p) if model == "si1000" else uniform(c0, p)
            r = sample(TDEM(c, magic="T", pattern=False), shots, seed=1)
            row = dict(circuit=stem, model=model, p=p, shots=shots, P_acc=r["P_acc"], P_det0=r.get("P_det0"),
                       seconds=round(time.time() - t0))
            res = json.load(open(path)) if os.path.exists(path) else []
            res = [x for x in res if (x["circuit"], x["model"], x["p"]) != (stem, model, p)] + [row]
            json.dump(res, open(path, "w"), indent=1)
            print(json.dumps(row), flush=True)
