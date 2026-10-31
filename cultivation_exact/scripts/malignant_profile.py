"""Weight profile of the minimal malignant configurations of a TDEM-ready circuit: which gates and layers the
faults sit at (representative location of each fault class). Usage: python malignant_profile.py PATH KMAX [si1000]"""
import sys, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast
from branch import FastExact
from noise import si1000

path, kmax = sys.argv[1], int(sys.argv[2])
c = stim.Circuit.from_file(path)
if len(sys.argv) > 3:
    c = si1000(c, 1e-3)
m = TDEM(c, magic="T", pattern=False)
dem = m.dem()
errs = [inst for inst in dem.flattened() if inst.type == "error"]
mech = m.mechanisms()
keep = [i for i, x in enumerate(mech) if x[1] or x[2]]
k, f = fault_distance_fast(FastExact(m), kmax, mech, verbose=False)
used = sorted({keep[i] for x in f for i in x[0]})
filt = stim.DetectorErrorModel()
for i in used:
    filt.append(errs[i])
expl = m.noisy.explain_detector_error_model_errors(dem_filter=filt, reduce_to_one_representative_error=True)
desc = {}
for i, e in zip(used, expl):
    loc = e.circuit_error_locations[0]
    tg = [t.gate_target.value for t in loc.instruction_targets.targets_in_range]
    desc[i] = (loc.tick_offset, loc.instruction_targets.gate, tuple(tg))
w = collections.Counter()
tot = 0
for x in f:
    wt = x[4] * x[3]
    tot += wt
    for i in x[0]:
        t, g, tg = desc[keep[i]]
        w[(t // 10 * 10, g)] += wt / len(x[0])
print("fd", k, "malignant", len(f), "c", tot / 1e-3 ** k)
for (t, g), v in sorted(w.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  ticks {t:4d}-{t+9:4d} {g:14s} {v / tot:6.1%}")
