"""GHZ-fold checks WITHOUT stabilizer rounds (ideal encoded input, one check round of src/fold.py, ideal decoder),
with the fold trees of hook_bound.py. Such a fold is a single check, so Theorem (late errors dominate) applies and
the integer program of late_ilp.py gives its real-T fault distance (Appendix G of the paper).

Usage: python fold_nostab.py DELTA TREES [run]     e.g.  python fold_nostab.py 9 b1,b2 run
Writes results/fold_n<n>_<tree>_nostab.stim (injection T|+>; the late ILP does not depend on the injection sign,
checked at delta = 7) and, with "run", prints the certificate of late_ilp.certify for each tree.
Results (results/logs/late_ilp_fold_d9.log): delta = 7 trees c, b1, b2: 4; delta = 9 tree b1: 5, tree b2: 4,
equal to the hook bounds of hook_bound.py."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from fold import make_fold
from hook_bound import tree

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

if __name__ == "__main__":
    delta, trees = int(sys.argv[1]), sys.argv[2].split(",")
    run = "run" in sys.argv[3:]
    faces, n = color_code_faces(delta)
    for sd in trees:
        parent, root, reps = tree(n, faces, sd)
        c = make_fold(faces, n, inject_dag=False, parent=parent, root=root, reps=reps, stab_rounds=False)
        path = os.path.join(R, f"fold_n{n}_{sd}_nostab.stim")
        open(path, "w").write(str(c))
        print(json.dumps(dict(delta=delta, tree=sd, circuit=os.path.basename(path), qubits=c.num_qubits)), flush=True)
        if run:
            from late_ilp import certify
            t0 = time.time()
            r = certify(c, faces, n, delta, time_limit=int(os.environ.get("ILP_SECONDS", 3600)), verbose=False)
            r["seconds"] = round(time.time() - t0)
            print(json.dumps(dict(tree=sd, **r), default=str), flush=True)
