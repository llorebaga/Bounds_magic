"""Exact search behind the integer-programming certificate of the all-to-all verified cat at delta = 7 without
stabilizer rounds (late_ilp_run.py vcat 7 inject_dag): writes the circuit that late_ilp certifies and certifies it
up to KMAX faults with the Gauss-sum evaluator (nn_cat_fd.certify). Weights five and six then follow from
fd_k6.py on the written file (K = 5 and 6), which do not rely on the dominance theorem or on an integer program.
Usage: python vcat_d7_exact.py [KMAX]   -> results/vcat_d7_r1_injdag_nostab.stim"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from late_ilp_test import build
from nn_cat_fd import certify

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n = color_code_faces(7)[1]
    c = build("vcat", 7, list(range(n)), True)
    path = os.path.join(R, "vcat_d7_r1_injdag_nostab.stim")
    if not os.path.exists(path):                    # other jobs may be reading it
        c.to_file(path)
    t0 = time.time()
    r = certify(c, kmax, proxy=False)["T"]
    print(json.dumps(dict(circuit=os.path.basename(path), qubits=c.num_qubits, kmax=kmax, real_T=r,
                          seconds=round(time.time() - t0))), flush=True)
