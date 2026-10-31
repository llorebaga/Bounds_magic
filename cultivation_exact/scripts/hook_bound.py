"""Hook upper bound on the real-T fault distance of a GHZ-fold check, from Lemma 4 and Corollary 18 alone.

The check round of src/fold.py is pair CX(partner -> data), fold CX(parent -> child) deepest first, MX and RX of
the root, the unfold (fold CXs reversed), unpair CX(partner -> data) and the U layer. We propagate every single X
fault (on any qubit before any of these operations) and every X(x)X fault on a CX to the X pattern it leaves on the
data at the U layer; the check's own measurements cannot see X errors. If s such faults leave X exactly on a set u
inside a minimum-weight logical X support l, with u != l and u nonempty, then u holds no stabilizer and, by Lemma 4,
these s faults plus delta - |u| data faults after the U layer are harmful:
    FD <= min over such (s, u) of s + delta - |u|.
The bound needs no simulation, so it also applies at delta = 9 and 11. Trees as in fold_fd.py.
Usage: python hook_bound.py BASE_WIDTH SEEDS [SMAX]   (e.g. 7 c,bc,b1,b2,b3,1)"""
import os, sys, random, json, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from fold import bfs_tree
from foldtree import min_weight_logicals


def tree(n, faces, sd):
    cap = 2 if sd.startswith("b") else None
    tok = sd[1:] if sd.startswith("b") else sd
    if tok in ("c", ""):
        parent, root = bfs_tree(n, faces, max_children=cap)
        return parent, root, {q: "d" for q in range(n)}
    parent, root = bfs_tree(n, faces, seed=int(tok), max_children=cap)
    rng = random.Random(int(tok))
    return parent, root, {q: rng.choice("dp") for q in range(n)}


def check_ops(n, parent, root, reps):
    """operations of one check round between the two U layers, on qubits ('d', q) and ('p', q)"""
    node = {q: ("d", q) if reps[q] == "d" else ("p", q) for q in range(n)}
    depth = {}
    for q in parent:
        k, v = 0, q
        while parent[v] is not None:
            v, k = parent[v], k + 1
        depth[q] = k
    fold = [(node[parent[q]], node[q]) for q in sorted((q for q in parent if parent[q] is not None),
                                                       key=lambda q: -depth[q])]
    ops = [("CX", ("p", q), ("d", q)) for q in range(n)]
    ops += [("CX", c, t) for c, t in fold]
    ops += [("MX", node[root]), ("RX", node[root])]
    ops += [("CX", c, t) for c, t in reversed(fold)]
    ops += [("CX", ("p", q), ("d", q)) for q in range(n)]
    return ops


def propagate(ops, start, xs):
    x = set(xs)
    for op in ops[start:]:
        if op[0] == "CX":
            if op[1] in x:
                x ^= {op[2]}
        elif op[0] == "RX":
            x.discard(op[1])
    return frozenset(q for (k, q) in x if k == "d")


def supports(n, ops):
    qubits = [("d", q) for q in range(n)] + [("p", q) for q in range(n)]
    out = set()
    for k in range(len(ops) + 1):
        for qb in qubits:
            out.add(propagate(ops, k, [qb]))
        if k < len(ops) and ops[k][0] == "CX":
            out.add(propagate(ops, k + 1, [ops[k][1], ops[k][2]]))
    return [s for s in out if s]


def bound(n, delta, sup, logs, smax):
    """min over s <= smax faults of s + delta - |u|, u = XOR of the supports inside a min-weight logical, u != l"""
    vec = lambda s: sum(1 << q for q in s)
    S = sorted({vec(s) for s in sup})
    L = [vec(l) for l in logs]
    best = (delta, 0, None)                      # no hooks: delta data faults (Proposition 9)

    def consider(u, s):
        nonlocal best
        if u == 0:
            return
        w = bin(u).count("1")
        if w >= delta or s + delta - w >= best[0]:
            return
        for l in L:
            if u & ~l == 0:
                best = (s + delta - w, s, u)
                return
    levels = {0: {0}}
    for s in range(1, smax + 1):
        if s <= 2:
            levels[s] = {a ^ b for a in levels[s - 1] for b in S}
            for u in levels[s]:
                consider(u, s)
        else:                                    # meet in the middle: level s-2 against pairs, only near logicals
            near = set()
            for l in L:
                pass
            for a in levels[s - 2]:
                for b in levels[2]:
                    consider(a ^ b, s)
    return best


def local_min_logicals(n, faces, delta, restarts=200, seed=1):
    """minimum-weight X logical supports reached by greedy descent from random logical representatives and closed
    under weight-preserving face moves (a subset of all of them; enough for an upper bound)"""
    rng = random.Random(seed)
    F = [sum(1 << q for q in f) for f in faces]
    w = lambda v: bin(v).count("1")
    found = set()
    for _ in range(restarts):
        v = (1 << n) - 1
        for f in F:
            if rng.random() < 0.5:
                v ^= f
        improved = True
        while improved:
            improved = False
            for f in rng.sample(F, len(F)):
                if w(v ^ f) < w(v):
                    v ^= f
                    improved = True
        if w(v) == delta:
            found.add(v)
    frontier = list(found)
    while frontier:
        nxt = []
        for v in frontier:
            for f in F:
                u = v ^ f
                if w(u) == delta and u not in found:
                    found.add(u)
                    nxt.append(u)
        frontier = nxt
    return [frozenset(q for q in range(n) if (v >> q) & 1) for v in found]


def packing_bound(delta, sup, logs):
    """delta - max sum(|S|-1) over pairwise disjoint single-fault supports inside one min-weight logical, |u| < delta"""
    best = (delta, None)
    sup = sorted(set(sup), key=len, reverse=True)
    for l in logs:
        inside = [s for s in sup if len(s) >= 2 and s <= l]
        top = [0, None]

        def rec(i, used, size, gain, chosen):
            if gain > top[0]:
                top[0], top[1] = gain, list(chosen)
            for j in range(i, len(inside)):
                s = inside[j]
                if not (s & used) and size + len(s) < delta:
                    chosen.append(sorted(s))
                    rec(j + 1, used | s, size + len(s), gain + len(s) - 1, chosen)
                    chosen.pop()
        rec(0, frozenset(), 0, 0, [])
        if delta - top[0] < best[0]:
            best = (delta - top[0], top[1])
    return best


if __name__ == "__main__":
    bw = int(sys.argv[1])
    seeds = sys.argv[2].split(",")
    faces, n = color_code_faces(bw)
    delta = bw
    mode = sys.argv[3] if len(sys.argv) > 3 else ("xor" if delta <= 7 else "pack")
    if delta <= 7:
        w, L = min_weight_logicals(list(range(n)), faces, list(range(n)))
        assert w == delta
    else:
        L = local_min_logicals(n, faces, delta)
    for sd in seeds:
        parent, root, reps = tree(n, faces, sd)
        sup = supports(n, check_ops(n, parent, root, reps))
        if mode == "xor":
            b, s, u = bound(n, delta, sup, L, (delta - 1) // 2)
            hooks = sorted(q for q in range(n) if u and (u >> q) & 1)
        else:
            b, hooks = packing_bound(delta, sup, L)
        print(json.dumps(dict(delta=delta, tree=sd, mode=mode, min_weight_logicals=len(L),
                              single_fault_supports=len(set(sup)), fd_upper_bound=b, witness=hooks)), flush=True)
