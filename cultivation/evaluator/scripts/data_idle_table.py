"""Rows of the cost table (SI1000, p = 1e-3) for the checks from an ideally encoded state, with idle noise on the data.

Reads results/didle/rates.json (qubits, layers, acceptance, fault distance, c_k for k <= 3), the weight-4 and
weight-5 enumerations (results/<stem>_si1000_c5.json, results/<stem>_si1000_c4.json), results/didle/mitm.json
(logical error to all orders up to eight faults) and results/didle/acceptance.json (distance seven).
Where the acceptance is below 5% the rare-event estimator is unreliable; there the logical error is given as the
lower bound Q * (exact weights of three, four and five faults) / P_acc, with Q the probability of no fault.
Usage: python data_idle_table.py   -> results/didle/table_si1000.json and LaTeX rows on stdout"""
import os, sys, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import stim
from tdem import TDEM
from noise import si1000

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
D = os.path.join(R, "didle")
P = 1e-3
ROWS = [  # label, delta, stem
    ("GHZ fold", 3, "fold_n7_c_didle"), ("GHZ fold, no rounds", 3, "fold_n7_c_nostab_didle"),
    ("cat, verified, no rounds", 3, "vcat_d3_r1_nostab_didle"),
    ("NN ring, not un-grown", 3, "nn_cat_d3_r1_v1_didle"), ("NN ring, not un-grown, end parity twice", 3, "nn_cat_d3_r1_v2_didle"), ("NN ring", 3, "nn_cat_d3_r1_v1_unf_didle"),
    ("GHZ fold", 5, "fold_n19_b1_didle"), ("GHZ fold, no rounds", 5, "fold_n19_b1_nostab_didle"),
    ("cat, not verified", 5, "ucat_d5_r1_didle"), ("cat, not verified, no rounds", 5, "ucat_d5_r1_nostab_didle"),
    ("cat, verified", 5, "vcat_d5_r1_didle"), ("cat, verified, no rounds", 5, "vcat_d5_r1_nostab_didle"),
    ("NN ladder, not un-grown", 5, "nn_cat_d5_r1_v1_didle"), ("NN ladder", 5, "nn_cat_d5_r1_v1_unf_didle"),
    ("GHZ fold", 7, "fold_n37_c_didle"), ("cat, not verified", 7, "ucat_d7_r1_injdag_didle"),
    ("cat, verified", 7, "vcat_d7_r1_injdag_didle"), ("NN ladder, not un-grown", 7, "nn_cat_d7_r1_v1_dag_didle"),
    ("NN ladder", 7, "nn_cat_d7_r1_v1_unf_didle"),
]
FD7 = {"fold_n37_c_didle": 4, "ucat_d7_r1_injdag_didle": 4, "vcat_d7_r1_injdag_didle": 7,
       "nn_cat_d7_r1_v1_dag_didle": 6, "nn_cat_d7_r1_v1_unf_didle": 7}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def weight(stem, k):
    for path in (os.path.join(R, f"{stem}_si1000_c{k}.json"), os.path.join(R, f"{stem}_si1000_c4.json") if k == 4 else ""):
        d = load(path) if path else None
        if d:
            return d["weight"]
    return None


def no_fault(stem):
    c = si1000(stim.Circuit.from_file(os.path.join(D, stem + ".stim")), P)
    m = TDEM(c, magic="T", pattern=False)
    p = np.array([x[0] for x in m.mechanisms() if x[1] or x[2]])
    return float(np.prod(1 - p))


if __name__ == "__main__":
    rates = {r["circuit"]: r for r in load(os.path.join(D, "rates.json"))}
    mitm = {(r["circuit"], r["p"]): r for r in load(os.path.join(D, "mitm.json"))}
    acc7 = {(r["circuit"], r["model"], r["p"]): r for r in load(os.path.join(D, "acceptance.json"))}
    out = []
    for label, delta, stem in ROWS:
        c0 = stim.Circuit.from_file(os.path.join(D, stem + ".stim"))
        row = dict(label=label, delta=delta, circuit=stem, qubits=c0.num_qubits, layers=c0.num_ticks)
        if delta < 7:
            rs = rates[stem]["si1000"]
            row["P_acc"] = rs["P_acc"]
            if isinstance(rs["fd"], int):
                row["fd"], row["leading"] = rs["fd"], rs["c"]
            else:
                row["fd"], row["leading"] = 5, load(os.path.join(R, f"{stem}_si1000_c5.json"))["c5"]
            if row["P_acc"] >= 0.05:
                m = mitm[(stem, P)]
                row["eps"], row["se_eps"], row["eps_kind"] = m["eps"], m["se_eps"], "rare-event estimator"
            else:
                Q = no_fault(stem)
                w = [weight(stem, k) for k in (4, 5)]
                w3 = rs["c"] * P ** 3 if row["fd"] == 3 else 0.0
                tot = w3 + sum(x for x in w if x)
                row["eps"], row["eps_kind"], row["Q"] = Q * tot / row["P_acc"], "lower bound (exact weights to five faults)", Q
        else:
            a = acc7[(stem, "si1000", P)]
            row["P_acc"], row["accepted_samples_upper"] = a["P_acc"], a["P_det0"] * a["shots"]
            row["fd"] = FD7[stem]
            c4 = load(os.path.join(R, f"{stem}_si1000_c4.json"))
            row["leading"] = c4["c4"] if c4 else None
        row["volume"] = row["qubits"] * row["layers"] / row["P_acc"] if row["P_acc"] > 1e-12 else None
        out.append(row)
        print(json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(D, "table_si1000.json"), "w"), indent=1)
