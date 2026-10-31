"""Real-T (and S-proxy) fault distance of GHZ-fold double checks (src/fold.py) on the triangular colour codes.

Usage: python fold_fd.py BASE_WIDTH KMAX SEEDS [proxy]
  BASE_WIDTH 3, 5, 7 -> [[7,1,3]], [[19,1,5]], [[37,1,7]];  SEEDS: comma list, 'c' = central BFS tree with data
  representatives, an integer s = BFS tree with random neighbour order and random representatives (seed s);
  a prefix 'b' ('bc', 'b1', ...) caps the tree at two children per node (deeper, smaller hooks).
For each tree, both injections are built and the one with deterministic checks (two accepted Z classes) is kept.
Writes results/fold_d<d>.json and the circuits results/fold_d<d>_<seed>.stim."""
import os, sys, json, time, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from fold import make_fold, bfs_tree
from tdem import TDEM
from fd_fast import fault_distance_fast
from branch import FastExact

bw, kmax = int(sys.argv[1]), int(sys.argv[2])
seeds = sys.argv[3].split(",")
proxy = len(sys.argv) > 4 and sys.argv[4] == "proxy"
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
faces, n = color_code_faces(bw)
out = []
for sd in seeds:
    cap = 2 if sd.startswith("b") else None
    tok = sd[1:] if sd.startswith("b") else sd
    if tok in ("c", ""):
        parent, root = bfs_tree(n, faces, max_children=cap)
        reps = None
    else:
        parent, root = bfs_tree(n, faces, seed=int(tok), max_children=cap)
        rng = random.Random(int(tok))
        reps = {q: rng.choice("dp") for q in range(n)}
    depth = {}
    for q in parent:
        k, v = 0, q
        while parent[v] is not None:
            v, k = parent[v], k + 1
        depth[q] = k
    for dag in (False, True):
        t0 = time.time()
        c = make_fold(faces, n, inject_dag=dag, parent=parent, root=root, reps=reps)
        m = TDEM(c, magic="T", pattern=False)
        fe = FastExact(m)
        if len(fe.A) != 2:
            continue
        mech = m.mechanisms()
        k, f = fault_distance_fast(fe, kmax, mech, verbose=False)
        coef = sum(x[4] * x[3] for x in f if x[4] is not None) / 1e-3 ** k if k and k <= 3 else None
        row = dict(n=n, seed=sd, max_children=cap, inject_dag=dag, tree_depth=max(depth.values()),
                   partner_reps=sum(1 for v in (reps or {}).values() if v == "p"), t=m.t, k_affine=m.kaff,
                   classes=len(mech), fd_real=k if k else f">{kmax}", malignant=len(f), coef_p1e3=coef,
                   seconds=round(time.time() - t0))
        if proxy:
            mp = TDEM(c, magic="S", pattern=False)
            kp, fp = fault_distance_fast(mp, kmax, mp.mechanisms(), verbose=False)
            row["fd_proxy"] = kp if kp else f">{kmax}"
        c.to_file(os.path.join(R, f"fold_n{n}_{sd}.stim"))
        out.append(row)
        print(json.dumps(row), flush=True)
        json.dump(out, open(os.path.join(R, f"fold_n{n}.json"), "w"), indent=1)
