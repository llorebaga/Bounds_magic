"""Stratified real-T / proxy logical error of Gidney's d1=5 circuit (unflagged, or a design2 .stim file).
Usage: python strat_d5.py [circuit.stim] ; compares with SOFT (2512.23037) Table III for the unflagged circuit."""
import sys, time, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from stratified import Stratified
from noise import si1000, uniform

G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
path = sys.argv[1] if len(sys.argv) > 1 else G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim"
tag = sys.argv[2] if len(sys.argv) > 2 else "unflagged"
c0 = stim.Circuit.from_file(path)
SOFT = {2e-3: 3.41e-8, 1e-3: 4.59e-9, 5e-4: 1.57e-10}
out = []
for model in ["uniform", "si1000"]:
    for p in [2e-3, 1e-3, 5e-4]:
        c = (si1000 if model == "si1000" else uniform)(c0, p)
        for magic in ["T", "S"]:
            t0 = time.time()
            m = TDEM(c, magic=magic, **({"complement": True} if magic == "T" else {}))
            st = Stratified(m)
            samples = {k: (20_000_000 if 3 <= k <= 7 else 4_000_000) for k in range(1, 15)}
            r = st.estimate(1.0, 14, samples)
            r.pop("strata")
            r.update(model=model, p=p, magic=magic, circuit=tag, seconds=round(time.time() - t0))
            if model == "uniform" and magic == "T" and tag == "unflagged":
                r["SOFT"] = SOFT[p]
            out.append(r)
            print(json.dumps(r), flush=True)
            json.dump(out, open(__file__.rsplit('scripts', 1)[0] + f"results/strat_d5_{tag}.json", "w"), indent=1)
