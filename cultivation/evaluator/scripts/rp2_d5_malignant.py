"""Malignant triples of the distance-five RP^2 circuit (real T) and their circuit locations; then the fault
distance (to weight 2, and weight 3 if asked) in further branches of the random records of the circuit.
Usage: python rp2_d5_malignant.py [NBRANCH] [K_BRANCH]"""
import sys, os, time, json, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from lean_exact import LeanTDEM, LeanExact
from fd_fast import fault_distance_fast

nbranch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
kb = int(sys.argv[2]) if len(sys.argv) > 2 else 2
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "rp2_circuits", "rp_3_rp_5_T_cult.stim")
c = stim.Circuit.from_file(path)
ins = list(c.flattened())
tick_of, tk = [], 0
for op in ins:
    tick_of.append(tk)
    tk += op.name == "TICK"
layers = [tick_of[k] for k, op in enumerate(ins) if op.name in ("S", "S_DAG") and len(op.targets_copy()) > 1]
print("ticks of the magic layers (original circuit):", layers, flush=True)

m = LeanTDEM(c, magic="T", pattern=False, zl_from_obs=True)
mech = m.mechanisms()
le = LeanExact(m)
mm = [x for x in mech if x[1] or x[2]]
k, f = fault_distance_fast(le, 3, mech, verbose=False)


def explain(i):
    p, D, o = mm[i]
    tg = [stim.target_relative_detector_id(b) for b in range(m.n_det) if (D >> b) & 1] + \
         [stim.target_logical_observable_id(b) for b in range(o.bit_length()) if (o >> b) & 1]
    flt = stim.DetectorErrorModel()
    flt.append("error", p, tg)
    out = []
    for e in m.noisy.explain_detector_error_model_errors(dem_filter=flt, reduce_to_one_representative_error=True):
        for loc in e.circuit_error_locations[:1]:
            out.append(f"skeleton tick {loc.tick_offset}: {loc.instruction_targets.gate} "
                       f"{[t.gate_target.value for t in loc.instruction_targets.targets_in_range]} "
                       f"{[str(x.gate_target) for x in loc.flipped_pauli_product]}")
    return out


print(f"reference branch: fault distance {k}, {len(f)} malignant", flush=True)
for idx, o, acc, bad, prob in f:
    print(f"  triple: acceptance {acc:.4g}, bad {bad:.4g}", flush=True)
    for i in idx:
        for s in explain(i):
            print("     ", s, flush=True)

# further branches of the random records: fault distance to weight kb
free = list(m.free_records)
rng = random.Random(1)
res = []
for _ in range(nbranch):
    ov = {r: rng.randint(0, 1) for r in free}
    t0 = time.time()
    mb = LeanTDEM(c, magic="T", pattern=False, zl_from_obs=True, overrides=ov)
    kk, ff = fault_distance_fast(LeanExact(mb), kb, mb.mechanisms(), verbose=False)
    res.append(dict(branch=ov, fd=kk if kk else f">{kb}", malignant=len(ff), seconds=round(time.time() - t0)))
    print("BRANCH", json.dumps(res[-1]), flush=True)
