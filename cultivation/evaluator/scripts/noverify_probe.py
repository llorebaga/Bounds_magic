"""Probe: un-verified path cat (no parity checks, no un-growing) on the ladder layout, random data placements.
Usage: python noverify_probe.py DELTA N SEED [KMAX]"""
import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import nn_cat
from colorcode import color_code_faces
from fold import merge_final_mpps, regenerate_detectors
from nn_cat_fd import certify


def build(delta, order, inject_dag=False, unfold=False, root=None):
    """root=None: the ladder's turn (two arms); root=0: grown from one end along the whole path"""
    faces, n = color_code_faces(delta)
    L = nn_cat.ladder({5: 8, 7: 17}[delta], diagonals=False)
    L["ver_pairs"], L["ver_sites"] = [], []
    if root is not None:
        L["root"] = root
    pos = {q: i for i, q in enumerate(order)}
    lf = [[pos[q] for q in f] for f in faces]
    c, G = nn_cat.make_nn_cat(lf, n, unfold=unfold, rounds=1, inject_dag=inject_dag, **L)
    return regenerate_detectors(merge_final_mpps(c)), L["root"]


if __name__ == "__main__":
    delta, N, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    kmax = int(sys.argv[4]) if len(sys.argv) > 4 else delta - 1
    faces, n = color_code_faces(delta)
    rng = random.Random(seed)
    for k in range(N):
        order = list(range(n))
        if k:
            rng.shuffle(order)
        t0 = time.time()
        for dag in (False, True):
            c, root = build(delta, order, dag)
            r = certify(c, kmax, proxy=False)["T"]
            if r["fd"] != 1:
                break
        print(json.dumps(dict(k=k, fd=r["fd"], malignant=r["malignant"], dag=dag, s=round(time.time()-t0, 1),
                              order=order)), flush=True)
