"""S-proxy fault distance (to weight KMAX <= 4) of a circuit in the S-proxy convention, from its stim detector error
model: a configuration is malignant when its detector pattern cancels and it flips the observable.
Usage: python proxy_fd.py KMAX PATH [PATH ...]"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from fd_fast import fault_distance_fast


class Clifford:
    def evaluate(self, o):
        return (1.0, 1.0 if o else 0.0)


kmax = int(sys.argv[1])
for path in sys.argv[2:]:
    c = stim.Circuit.from_file(path)
    mech = []
    for inst in c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True).flattened():
        if inst.type != "error":
            continue
        D = o = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                D ^= 1 << t.val
            elif t.is_logical_observable_id():
                o ^= 1 << t.val
        mech.append((inst.args_copy()[0], D, o))
    k, f = fault_distance_fast(Clifford(), kmax, mech, verbose=False)
    print("JSON", json.dumps(dict(circuit=os.path.basename(path), fd_proxy=k if k else f">{kmax}", malignant=len(f))),
          flush=True)
