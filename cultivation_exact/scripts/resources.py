"""Resource comparison of cultivation checks under SI1000 noise at p = 1e-3 (Table II of the paper).

For each circuit: qubits, depth (ticks of the circuit file), acceptance P_acc relative to the fault-free run (TDEM
sampling), real-T fault distance and leading term c_k p^k of the probability of an accepted wrong output (exact:
k <= 3 here, k = 4 and 5 read from leading_k4.py / leading_k5.py outputs), the logical error per accepted state
eps = c_k p^k (the factor prod(1 - p_c) is common to the failure weight and to the acceptance, so to leading
order the error per accepted state is the leading term itself) and the expected spacetime volume per accepted state V = qubits * ticks / P_acc.
Google's experimental circuit (with its own injection and QEC cycles) is taken from google_mc.py instead.
Usage: python resources.py [SHOTS]   -> results/resources_si1000.json"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from mc import sample
from noise import si1000
from fd_fast import fault_distance_fast
from branch import FastExact

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
P = 1e-3
ROWS = [  # label, d, nearest-neighbour, file, leading-term source (None: compute k <= 3)
    ("GHZ fold", 3, False, "fold_n7_c.stim", None),
    ("NN cat (ring)", 3, True, "nn_cat_d3_r1_v2.stim", None),
    ("NN cat (ring), un-grown", 3, True, "nn_cat_d3_r1_v1_unf.stim", None),
    ("NN protocol, prepared by measurement", 3, True, "nn_meas_d3_XCZXCC_unf.stim", None),
    ("cat, verified", 3, False, "vcat_d3_injdag_merged.stim", None),
    ("GHZ fold", 5, False, "fold_n19_b1.stim", None),
    ("cat, not verified", 5, False, "ucat_d5_r1.stim", None),
    ("cat, verified", 5, False, "vcat_d5_r1.stim", "vcat_d5_r1_si1000_c5.json"),
    ("NN cat (ladder)", 5, True, "nn_cat_d5_r1_v1.stim", "nn_cat_d5_r1_v1_si1000_c5.json"),
    ("NN cat (ladder), un-grown", 5, True, "nn_cat_d5_r1_v1_unf.stim", "nn_cat_d5_r1_v1_unf_si1000_c5.json"),
    ("GHZ fold", 7, False, "fold_n37_c.stim", "fold_n37_c_si1000_c4.json"),
    ("cat, not verified", 7, False, "ucat_d7_r1_injdag.stim", "ucat_d7_r1_injdag_si1000_c4.json"),
    ("cat, verified", 7, False, "vcat_d7_r1_injdag.stim", "none"),
    ("NN cat (ladder)", 7, True, "nn_cat_d7_r1_v1_dag.stim", "none"),
    ("NN cat (ladder), un-grown", 7, True, "nn_cat_d7_r1_v1_unf.stim", "none"),
]


def leading(c, src):
    if src:
        path = os.path.join(R, src)
        if src == "none" or not os.path.exists(path):
            return None, None
        d = json.load(open(path))
        k = 5 if "c5" in d else 4
        return k, d[f"c{k}"]
    m = TDEM(c, magic="T", pattern=False)
    k, f = fault_distance_fast(FastExact(m), 3, m.mechanisms(), verbose=False)
    return k, sum(x[4] * x[3] for x in f) / P ** k


if __name__ == "__main__":
    shots = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10 ** 6
    out = []
    for label, d, nn, fname, src in ROWS:
        c0 = stim.Circuit.from_file(os.path.join(R, fname))
        c = si1000(c0, P)
        acc = sample(TDEM(c, magic="T", pattern=False), shots, seed=1)["P_acc"]
        k, ck = leading(c, src)
        row = dict(label=label, d=d, nn=nn, circuit=fname, qubits=c0.num_qubits, ticks=c0.num_ticks, P_acc=acc,
                   fd=k, c=ck, eps=ck * P ** k if ck else None, volume=c0.num_qubits * c0.num_ticks / acc)
        out.append(row)
        print(json.dumps(row), flush=True)
    g = [r for r in json.load(open(os.path.join(R, "google_mc.json")))
         if r["fig"] == "fig3" and r["p"] == P and r["magic"] == "T"]
    if g:
        gc = stim.Circuit.from_file(os.path.join(R, "google_fig3_p0.001_s-1_cut162.stim"))
        layers, cur = [], set()
        for inst in gc.flattened():
            if inst.name == "TICK":
                layers.append(cur); cur = set()
            elif inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS")                     and inst.name not in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"):
                cur.add(inst.name)
        ticks = sum(1 for l in layers if l and not l <= {"S", "S_DAG"})   # T-only layers are virtual
        out.append(dict(label="Google experiment", d=3, nn=True, circuit="google_fig3 (cut 162)",
                        qubits=gc.num_qubits, ticks=ticks, P_acc=g[0]["P_acc"], fd=2, c=None, eps=g[0]["eps"],
                        volume=gc.num_qubits * ticks / g[0]["P_acc"]))
        print(json.dumps(out[-1]), flush=True)
    json.dump(out, open(os.path.join(R, "resources_si1000.json"), "w"), indent=1)
