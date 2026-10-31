"""GHZ-fold double check in the style of Gidney, Shutty and Jones, on any triangular colour code.

Same surroundings as src/vcat.py (ideal encoded input, flagged stabiliser rounds, noiseless MPP readout), so a fold
check and a cat check at the same distance differ only in the check round. A check round, as in the published d=5
circuit (ticks 74-93):
  RX partners -> U^dag layer -> pair CX(partner_q -> q) -> fold: CX(parent -> child) over a tree whose nodes are one
  representative (data or partner) per pair, leaves first -> MX root, RX root -> unfold (fold CXs reversed)
  -> unpair CX(partner_q -> q) -> U layer -> MX partners.
The root measurement and the partner measurements read the check twice. A single X fault on a tree node during the
fold spreads, after the unfold, to its whole subtree: these hooks are what real T gates turn into coherent errors.
Detectors are not written by hand: every record parity that is deterministic in the noiseless S proxy is added
(foldtree.regenerate_detectors). Gates are written as S / S_DAG (the TDEM reads them as T / T^dag)."""
import sys
import random
import collections
import stim

sys.path.insert(0, __file__.rsplit("fold.py", 1)[0])
from cult_circuits import Builder          # noqa: E402
from tdem import sublattice_signs           # noqa: E402
from vcat import encoder_plan               # noqa: E402
from foldtree import regenerate_detectors, merge_final_mpps   # noqa: E402


def coface_graph(n, faces):
    adj = collections.defaultdict(set)
    for f in faces:
        for a in f:
            for b in f:
                if a != b:
                    adj[a].add(b)
    return adj


def bfs_tree(n, faces, root=None, seed=None, max_children=None):
    """spanning tree of the qubits that share a face, grown breadth-first from the root (random order of
    neighbours for seed != None, at most max_children children per node while possible); returns parent[q]
    (parent[root] = None) and the root"""
    adj = coface_graph(n, faces)
    rng = random.Random(seed)
    if root is None:
        # a central qubit: smallest eccentricity in the co-face graph
        def ecc(r):
            dist = {r: 0}
            dq = collections.deque([r])
            while dq:
                v = dq.popleft()
                for w in adj[v]:
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        dq.append(w)
            return max(dist.values())
        root = min(range(n), key=lambda r: (ecc(r), r))
    parent = {root: None}
    dq = collections.deque([root])
    while dq:
        v = dq.popleft()
        nb = sorted(adj[v])
        if seed is not None:
            rng.shuffle(nb)
        kids = 0
        for w in nb:
            if w not in parent and (max_children is None or kids < max_children):
                parent[w] = v
                dq.append(w)
                kids += 1
    while len(parent) < n:                      # qubits left out by the cap: attach to any tree neighbour
        for w in range(n):
            if w not in parent:
                hook = [v for v in sorted(adj[w]) if v in parent]
                if hook:
                    parent[w] = hook[0]
    return parent, root


def make_fold(faces, n, p=1e-3, inject_dag=False, parent=None, root=None, reps=None, idle=True, flagged=True,
              seed=None, model="uniform"):
    """reps[q] in {'d', 'p'}: the tree node of pair q is the data qubit or its partner (default: data)"""
    if parent is None:
        parent, root = bfs_tree(n, faces, seed=seed)
    reps = reps or {q: "d" for q in range(n)}
    B = Builder(p, idle, model)
    d = B.alloc(n)
    mcount = [0]

    def key(tag):
        mcount[0] += 1
        return (tag, mcount[0])

    signs = sublattice_signs(n, faces)
    gens, piv, Lrow, carrier = encoder_plan(n, faces, list(range(n)))

    def U_layer(dag):
        for q in range(n):
            B.g("S" if (signs[q] > 0) != dag else "S_DAG", d[q])

    def stabiliser_round():
        for P in "XZ":
            for f in faces:
                a = B.alloc(1)[0]
                fl = B.alloc(1)[0] if flagged else None
                if P == "X":
                    B.g("RX", a)
                    if flagged:
                        B.g("R", fl)
                    for i, q in enumerate(f):
                        B.g("CX", a, d[q])
                        if flagged and (i == 0 or i == len(f) - 2):
                            B.g("CX", a, fl)
                    B.g("MX", a, key=key("sx"))
                    if flagged:
                        B.g("M", fl, key=key("fx"))
                else:
                    B.g("R", a)
                    if flagged:
                        B.g("RX", fl)
                    for i, q in enumerate(f):
                        B.g("CX", d[q], a)
                        if flagged and (i == 0 or i == len(f) - 2):
                            B.g("CX", fl, a)
                    B.g("M", a, key=key("sz"))
                    if flagged:
                        B.g("MX", fl, key=key("fz"))

    def check_round():
        part = B.alloc(n)
        for q in range(n):
            B.g("RX", part[q])
        U_layer(True)
        for q in range(n):
            B.g("CX", part[q], d[q])
        node = {q: (d[q] if reps[q] == "d" else part[q]) for q in range(n)}
        depth = {}
        for q in parent:
            k, v = 0, q
            while parent[v] is not None:
                v = parent[v]
                k += 1
            depth[q] = k
        fold = [(node[parent[q]], node[q]) for q in sorted((q for q in parent if parent[q] is not None),
                                                           key=lambda q: -depth[q])]
        for c, t in fold:
            B.g("CX", c, t)
        B.g("MX", node[root], key=key("root"))
        B.g("RX", node[root])
        for c, t in reversed(fold):
            B.g("CX", c, t)
        for q in range(n):
            B.g("CX", part[q], d[q])
        U_layer(False)
        for q in range(n):
            B.g("MX", part[q], key=key("part"))

    stabiliser_round()
    check_round()
    stabiliser_round()
    for f in faces:
        for P in "XZ":
            B.g("MPP", [stim.target_combined_paulis([stim.target_x(d[q]) if P == "X" else stim.target_z(d[q])
                                                     for q in f])], key=key("final"))
    k = key("obs")
    B.g("MPP", [stim.target_combined_paulis([stim.target_y(q) for q in d])], key=k)
    B.observables.append([k])
    body = B.build()

    pre = stim.Circuit()
    pre.append("R", d)
    pre.append("H", [d[carrier]] + [d[q] for q in piv])
    pre.append("S_DAG" if inject_dag else "S", [d[carrier]])
    for q in Lrow:
        if q != carrier:
            pre.append("CX", [d[carrier], d[q]])
    for g, pv in zip(gens, piv):
        for q in g:
            if q != pv:
                pre.append("CX", [d[pv], d[q]])
    pre.append("TICK")
    out = pre.copy()
    for inst in body.flattened():
        out.append(inst)
    return regenerate_detectors(merge_final_mpps(out))
