"""d=3 verified-cat check with the T/T^dag parity that matches the injected state.

results/vcat_d3_T_merged.stim (protocol "T" of src/cult_circuits.py) injects T|+> but applies T on all 7 Steane
qubits in every check layer. Transversal T is the logical T^dag on the Steane code, so the check measures
T^dag X T, of which the injected state is not an eigenstate: half of the runs are lost and the real-T fault
distance is 2 (the parity issue of REPORT §4a, invisible to the S proxy). Injecting T^dag instead fixes it.
Writes results/vcat_d3_injdag_merged.stim and certifies its real-T fault distance with the Gauss-sum evaluator.
Usage: python vcat_d3_fix.py"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast
from branch import FastExact

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def certify(c, kmax=3):
    m = TDEM(c, magic="T", pattern=False)
    fe = FastExact(m)
    k, f = fault_distance_fast(fe, kmax, m.mechanisms(), verbose=False)
    coef = sum(x[4] * x[3] for x in f if x[4] is not None) / 1e-3 ** k if k else None
    return dict(accepted_Z_classes=len(fe.A), n0=m.n0, fd=k, malignant=len(f), coef_p1e3=coef)


src = stim.Circuit.from_file(os.path.join(R, "vcat_d3_T_merged.stim"))
fixed, done = stim.Circuit(), False
for inst in src.flattened():
    if not done and inst.name in ("S", "S_DAG"):           # the injection gate on the carrier
        assert len(inst.targets_copy()) == 1
        fixed.append("S_DAG" if inst.name == "S" else "S", inst.targets_copy())
        done = True
    else:
        fixed.append(inst)
fixed.to_file(os.path.join(R, "vcat_d3_injdag_merged.stim"))
out = {"vcat_d3_T_merged (inject T)": certify(src), "vcat_d3_injdag_merged (inject T^dag)": certify(fixed)}
for k, v in out.items():
    print(k, v, flush=True)
json.dump(out, open(os.path.join(R, "vcat_d3_fix.json"), "w"), indent=1)
