"""List the malignant real-T fault pairs of a TDEM-ready circuit with their circuit locations (tick, gate, Pauli).
Usage: python google_malignant.py PATH [KMAX]"""
import sys, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast
from branch import FastExact

path = sys.argv[1]; kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 2
m = TDEM(stim.Circuit.from_file(path), magic="T", pattern=False)
dem = m.dem()
errs = [inst for inst in dem.flattened() if inst.type == "error"]
mech = m.mechanisms()
keep = [i for i, x in enumerate(mech) if x[1] or x[2]]
k, f = fault_distance_fast(FastExact(m), kmax, mech, verbose=False)
print("fd", k, "malignant", len(f))
used = sorted({keep[i] for x in f for i in x[0]})
filt = stim.DetectorErrorModel()
for i in used:
    filt.append(errs[i])
expl = m.noisy.explain_detector_error_model_errors(dem_filter=filt, reduce_to_one_representative_error=False)
desc = {}
for i, e in zip(used, expl):
    locs = []
    for loc in e.circuit_error_locations[:4]:
        fl = loc.flipped_pauli_product
        ps = " ".join(f"{t.gate_target.pauli_type}{t.gate_target.value}" for t in fl)
        locs.append(f"t{loc.tick_offset} {loc.instruction_targets.gate}{[t.gate_target.value for t in loc.instruction_targets.targets_in_range]} {ps}")
    desc[i] = " | ".join(locs) + (f" (+{len(e.circuit_error_locations)-4})" if len(e.circuit_error_locations) > 4 else "")
cnt = collections.Counter()
for x in f:
    names = [desc[keep[i]] for i in x[0]]
    print(f"bad={x[3]:.3g} acc={x[2]:.3g} p={x[4]:.2e}")
    for nm in names:
        print("    ", nm)
