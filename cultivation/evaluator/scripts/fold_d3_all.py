"""Every GHZ-fold check on the [[7,1,3]] code has real-T fault distance at most two (Proposition 23): a test on
random trees.

The proof takes a deepest internal node v of the fold tree and its last-unfolded child c, a leaf: an X fault on v
just before that CX of the unfold leaves X on the data of v and c only, which no measurement of the check sees, and
every pair of data qubits of the [[7,1,3]] code lies in a weight-3 logical X support, so one data fault after the
U layer completes it (Lemma 4). Here we draw uniformly random labelled trees on the seven pairs (Pruefer
sequences, not restricted to qubits that share a face), random roots and random representatives, build the check
of src/fold.py, and certify its fault distance with the exact evaluator.
Usage: python fold_d3_all.py TREES [SEED]   -> results/fold_d3_all.json"""
import os, sys, json, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from fold import make_fold
from tdem import TDEM
from fd_fast import fault_distance_fast
from branch import FastExact

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def pruefer_tree(n, rng):
    seq = [rng.randrange(n) for _ in range(n - 2)]
    deg = [1] * n
    for x in seq:
        deg[x] += 1
    edges = []
    for x in seq:
        leaf = min(i for i in range(n) if deg[i] == 1)
        edges.append((leaf, x))
        deg[leaf] -= 1
        deg[x] -= 1
    u, w = [i for i in range(n) if deg[i] == 1]
    edges.append((u, w))
    return edges


def rooted(n, edges, root):
    adj = {i: [] for i in range(n)}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    parent, stack = {root: None}, [root]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in parent:
                parent[w] = v
                stack.append(w)
    return parent


if __name__ == "__main__":
    trees = int(sys.argv[1])
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    faces, n = color_code_faces(3)
    out, hist = [], {}
    for i in range(trees):
        edges = pruefer_tree(n, rng)
        root = rng.randrange(n)
        parent = rooted(n, edges, root)
        reps = {q: rng.choice("dp") for q in range(n)}
        for dag in (False, True):
            c = make_fold(faces, n, inject_dag=dag, parent=parent, root=root, reps=reps, stab_rounds=False)
            m = TDEM(c, magic="T", pattern=False)
            fe = FastExact(m)
            if len(fe.A) != 2:
                continue
            k, f = fault_distance_fast(fe, 2, m.mechanisms(), verbose=False)
            fd = k if k else ">2"
            hist[str(fd)] = hist.get(str(fd), 0) + 1
            out.append(dict(edges=edges, root=root, reps="".join(reps[q] for q in range(n)), inject_dag=dag,
                            fd_real=fd, malignant=len(f)))
            print(json.dumps(out[-1]), flush=True)
    json.dump(dict(trees=out, histogram=hist), open(os.path.join(R, "fold_d3_all.json"), "w"), indent=1)
    print("fault distances:", hist)
