"""Fault distance, leading terms and acceptance of the circuits of results/didle/ (data_idle_make.py).

For each circuit: real-T fault distance up to KMAX with the Gauss-sum evaluator, the number of malignant fault
classes and the leading coefficient c_k (k <= 3) under the file's uniform noise (p = 1e-3) and under SI1000 at
p = 1e-3; the SI1000 acceptance relative to the fault-free run (TDEM sampling); optionally the S-proxy fault distance.
Coefficients of order four and five come from leading_k4.py and leading_k5.py run on the same files.
Usage: python data_idle_rates.py SHOTS STEM:KMAX[:proxy] ...   -> appends to results/didle/rates.json"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from mc import sample
from noise import si1000
from fd_fast import fault_distance_fast
from branch import FastExact

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "didle")
P = 1e-3


def fd(c, kmax, magic="T"):
    m = TDEM(c, magic=magic, pattern=False)
    fe = FastExact(m)
    k, f = fault_distance_fast(fe, kmax, m.mechanisms(), verbose=False)
    ck = sum(x[4] * x[3] for x in f if x[4] is not None) / P ** k if k and k <= 3 else None
    return dict(fd=k if k else f">{kmax}", malignant=len(f) if k else 0, c=ck, accepted_Z_classes=len(fe.A))


if __name__ == "__main__":
    shots = int(float(sys.argv[1]))
    path = os.path.join(OUT, "rates.json")
    res = json.load(open(path)) if os.path.exists(path) else []
    for spec in sys.argv[2:]:
        parts = spec.split(":")
        stem, kmax, proxy = parts[0], int(parts[1]), len(parts) > 2 and parts[2] == "proxy"
        t0 = time.time()
        c0 = stim.Circuit.from_file(os.path.join(OUT, stem + ".stim"))
        cs = si1000(c0, P)
        row = dict(circuit=stem, qubits=c0.num_qubits, ticks=c0.num_ticks, uniform=fd(c0, kmax), si1000=fd(cs, kmax))
        if proxy:
            row["proxy_fd_uniform"] = fd(c0, kmax, magic="S")["fd"]
        row["si1000"]["P_acc"] = sample(TDEM(cs, magic="T", pattern=False), shots, seed=1)["P_acc"]
        row["si1000"]["shots"] = shots
        row["si1000"]["volume"] = c0.num_qubits * c0.num_ticks / row["si1000"]["P_acc"] if row["si1000"]["P_acc"] else None
        row["seconds"] = round(time.time() - t0)
        res = [r for r in res if r["circuit"] != stem] + [row]
        json.dump(res, open(path, "w"), indent=1)
        print(json.dumps(row), flush=True)
