"""Acceptance (relative to the fault-free run) under SI1000 or uniform noise, by TDEM sampling with exact weights.
Usage: python acceptance.py MODEL P SHOTS PATH [PATH ...]   (MODEL = si1000 | uniform)"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from mc import sample
from noise import si1000, uniform

model, p, shots = sys.argv[1], float(sys.argv[2]), int(float(sys.argv[3]))
for path in sys.argv[4:]:
    c0 = stim.Circuit.from_file(path)
    c = si1000(c0, p) if model == "si1000" else uniform(c0, p)
    m = TDEM(c, magic="T", pattern=False)
    r = sample(m, shots, seed=1)
    print("JSON", json.dumps(dict(circuit=os.path.basename(path), model=model, p=p, qubits=c0.num_qubits,
                                  ticks=c0.num_ticks, P_acc=r["P_acc"], shots=shots)), flush=True)
