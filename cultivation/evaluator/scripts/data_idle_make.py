"""Circuits with idle noise on the data (src/idle.py), and checks without stabiliser rounds (ablation).

The checks that start from an ideally encoded state (folds, cats, nearest-neighbour checks) were built without idle
noise on the data qubits, which the Builder never marked as live. This script writes, for each such circuit of the
paper, a copy with that noise added (results/didle/<stem>_didle.stim). It also builds the GHZ-fold and cat checks
without their flagged stabiliser rounds, from the same ideal input to the same ideal decoder as the grid checks
(results/didle/<name>_nostab_didle.stim); the trees and injected states are those of the stored circuits, and the
rebuilt circuits with stabiliser rounds are first compared with the stored ones.
Usage: python data_idle_make.py"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import random
import stim
from colorcode import color_code_faces
from fold import make_fold, bfs_tree
from vcat import make_vcat
from foldtree import merge_final_mpps
from idle import add_data_idle

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(R, "didle")
BASE = ["fold_n7_c", "fold_n7_1", "fold_n7_2", "nn_cat_d3_r1_v1_unf", "nn_cat_d3_r1_v2",
        "fold_n19_c", "fold_n19_bc", "fold_n19_b1", "ucat_d5_r1", "vcat_d5_r1", "nn_cat_d5_r1_v1", "nn_cat_d5_r1_v1_unf",
        "fold_n37_c", "fold_n37_bc", "fold_n37_b1", "fold_n37_b2", "fold_n37_b3", "fold_n37_1",
        "ucat_d7_r1_injdag", "vcat_d7_r1_injdag", "nn_cat_d7_r1_v1_dag", "nn_cat_d7_r1_v1_unf"]


def tree(bw, sd):
    faces, n = color_code_faces(bw)
    cap = 2 if sd.startswith("b") else None
    tok = sd[1:] if sd.startswith("b") else sd
    if tok in ("c", ""):
        parent, root = bfs_tree(n, faces, max_children=cap)
        reps = None
    else:
        parent, root = bfs_tree(n, faces, seed=int(tok), max_children=cap)
        rng = random.Random(int(tok))
        reps = {q: rng.choice("dp") for q in range(n)}
    return faces, n, parent, root, reps


def same(a, b):
    return str(a.without_noise()) == str(b.without_noise()) and str(a) == str(b)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    log = []
    for stem in BASE:
        c0 = stim.Circuit.from_file(os.path.join(R, stem + ".stim"))
        c1, added = add_data_idle(c0)
        c1.to_file(os.path.join(OUT, stem + "_didle.stim"))
        log.append(dict(circuit=stem, idle_locations_added=added))
        print(json.dumps(log[-1]), flush=True)
    # ablation: GHZ folds without stabiliser rounds (same trees and injections as the stored circuits, which are
    # first rebuilt exactly), and cats from the current builder with and without stabiliser rounds (the stored cats
    # come from an earlier schedule of the builder)
    for stem, bw, sd in [("fold_n7_c", 3, "c"), ("fold_n19_b1", 5, "b1")]:
        stored = stim.Circuit.from_file(os.path.join(R, stem + ".stim"))
        faces, n, parent, root, reps = tree(bw, sd)
        found = None
        for dag in (False, True):
            if same(make_fold(faces, n, inject_dag=dag, parent=parent, root=root, reps=reps), stored):
                found = dag
        assert found is not None, f"could not rebuild {stem}"
        c1, added = add_data_idle(make_fold(faces, n, inject_dag=found, parent=parent, root=root, reps=reps,
                                            stab_rounds=False))
        name = f"{stem}_nostab_didle"
        c1.to_file(os.path.join(OUT, name + ".stim"))
        log.append(dict(circuit=name, rebuilt_from=stem, inject_dag=found, qubits=c1.num_qubits,
                        ticks=c1.num_ticks, idle_locations_added=added))
        print(json.dumps(log[-1]), flush=True)
    for name, bw, verify, dag in [("vcat_d3_r1", 3, True, True), ("ucat_d5_r1", 5, False, False),
                                  ("vcat_d5_r1", 5, True, False)]:
        faces, n = color_code_faces(bw)
        for st, suffix in ((True, "builder_didle"), (False, "nostab_didle")):
            c = merge_final_mpps(make_vcat(faces, n, rounds=1, inject_dag=dag, verify=verify, stab_rounds=st))
            c1, added = add_data_idle(c)
            c1.to_file(os.path.join(OUT, f"{name}_{suffix}.stim"))
            log.append(dict(circuit=f"{name}_{suffix}", inject_dag=dag, stab_rounds=st, qubits=c1.num_qubits,
                            ticks=c1.num_ticks, idle_locations_added=added))
            print(json.dumps(log[-1]), flush=True)
    json.dump(log, open(os.path.join(OUT, "make.json"), "w"), indent=1)
