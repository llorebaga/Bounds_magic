"""Data placement along the nearest-neighbor ladder (src/nn_cat.py) chosen by the coherent-family criterion.

With rungs but no diagonals, two X faults at mirror positions of the two arms leave X on the data of columns
0..a-1 of both arms and flip no rung. Every Pauli component of such an error has this X part, and with T gates
its Z part is free on it, so it can be completed to a logical operator by as many single-qubit faults as its
distance to the logical X coset. The check keeps the full distance against these patterns if every one of them
is at distance at least delta - 2 from that coset. This script searches placements (simulated annealing) that
maximize the smallest such distance, then certifies the check exactly.

Usage: python ladder_order.py DELTA SEED [KMAX] [diag]"""
import os, sys, json, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import nn_cat
from colorcode import color_code_faces

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def coset_tools(faces, n):
    span = [0]
    for f in faces:
        m = sum(1 << q for q in f)
        span += [s ^ m for s in span]
    allq = (1 << n) - 1
    return [allq ^ s for s in span], span          # logical X coset, X stabilizer group


def mirror_masks(order, L, n):
    """X patterns (as masks over code qubits) of mirror tails: path index x on the top arm and its rung partner
    on the bottom arm, for the columns 0..a-1"""
    out = []
    m = 0
    for x in range(L):
        top, bot = x, n - 1 - x
        m |= (1 << order[top]) | (1 << order[bot])
        out.append(m)
    return out


def dist(m, coset):
    return min(bin(m ^ c).count("1") for c in coset)


def score(order, L, n, LOG):
    return [dist(m, LOG) for m in mirror_masks(order, L, n)]


def search(delta, seed, restarts=30, iters=3000):
    faces, n = color_code_faces(delta)
    L = {5: 8, 7: 17}[delta]
    LOG, _ = coset_tools(faces, n)
    rng = random.Random(seed)
    key = lambda ds: (min(ds), -sum(d == min(ds) for d in ds), sum(ds))
    best = None
    for _ in range(restarts):
        cur = list(range(n)); rng.shuffle(cur)
        cs = key(score(cur, L, n, LOG))
        T = 2.0
        for _ in range(iters):
            i, j = rng.sample(range(n), 2)
            nxt = cur[:]; nxt[i], nxt[j] = nxt[j], nxt[i]
            ns = key(score(nxt, L, n, LOG))
            if ns >= cs or rng.random() < 2.718 ** ((sum(ns) - sum(cs)) / T):
                cur, cs = nxt, ns
            T *= 0.999
        if best is None or cs > best[0]:
            best = (cs, cur[:])
    return faces, n, L, best[1], score(best[1], L, n, LOG), score(list(range(n)), L, n, LOG)


def build(delta, order, diagonals=False, unfold=True):
    """ladder check whose path position k holds code qubit order[k]"""
    faces, n = color_code_faces(delta)
    pos = {q: k for k, q in enumerate(order)}
    lf = [[pos[q] for q in f] for f in faces]
    from fold import merge_final_mpps, regenerate_detectors
    c, G = nn_cat.make_nn_cat(lf, n, unfold=unfold, rounds=1,
                              **nn_cat.ladder({5: 8, 7: 17}[delta], diagonals=diagonals))
    return regenerate_detectors(merge_final_mpps(c))


if __name__ == "__main__":
    from nn_cat_fd import certify
    delta, seed = int(sys.argv[1]), int(sys.argv[2])
    kmax = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    diag = "diag" in sys.argv
    faces, n, L, order, ds, ds0 = search(delta, seed)
    print(json.dumps(dict(delta=delta, order=order, mirror_dists=ds, natural_dists=ds0)), flush=True)
    for name, o in (("optimized", order), ("natural", list(range(n)))):
        c = build(delta, o, diagonals=diag)
        r = certify(c, kmax, proxy=False)["T"]
        c.to_file(os.path.join(R, f"ladder_d{delta}_{'diag' if diag else 'rungs'}_{name}.stim"))
        print(json.dumps(dict(placement=name, qubits=c.num_qubits, ticks=c.num_ticks, **r)), flush=True)
