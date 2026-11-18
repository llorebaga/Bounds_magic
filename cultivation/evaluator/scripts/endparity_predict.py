"""Blind test of the coherent-family criterion on the minimal grid check: an un-grown path cat on the ladder
layout whose only verification is the parity of its two ends (2n+1 qubits).

A growth fault at path index i spreads X to the part of its arm beyond i, towards the end. Un-growing maps the
spread back onto i, whose Z measurement then flips, so every spread costs its growth fault and one flip of that
measurement, and a single tail also flips the end parity. The accepted spread patterns of least cost are
  one tail on one arm                     cost 3
  two tails, one on each arm              cost 4
  an interval on one arm (two growth faults on it)  cost 4
Each pattern A then completes to a harmful configuration with r more faults, the cheaper of the X branch (distance
of A to the logical X coset) and the Z branch (distance to the X stabilizer group plus the Z faults that complete a
logical Z outside the region where Z is free). Predicted fault distance: min(delta, cost + r).
Usage: python endparity_predict.py DELTA N SEED [KMAX]"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import nn_cat
from colorcode import color_code_faces
from ladder_order import coset_tools
from fold import merge_final_mpps, regenerate_detectors

pc = lambda m: bin(m).count("1")


def patterns(order, n, root):
    """(cost, X mask over code qubits) for the cheapest accepted spread patterns"""
    left = list(range(root - 1, -1, -1))      # path indices from the root towards end 0
    right = list(range(root + 1, n))          # from the root towards end n-1
    mask = lambda idx: sum(1 << order[i] for i in idx)
    tailsL = [mask(left[j:]) for j in range(len(left))]
    tailsR = [mask(right[j:]) for j in range(len(right))]
    out = [(3, t) for t in tailsL + tailsR]
    out += [(4, a | b) for a in tailsL for b in tailsR]
    for arm in (left, right):
        for i in range(len(arm)):
            for j in range(i + 1, len(arm)):
                out.append((4, mask(arm[i:j])))
    return out


def completion(A, LOG, STAB):
    x_branch = min(pc(A ^ l) for l in LOG)
    z_branch = min(pc(A ^ g) + min(pc(l & ~(A | g)) for l in LOG) for g in STAB)
    return min(x_branch, z_branch)


def predict(order, faces, n, root, delta):
    LOG, STAB = coset_tools(faces, n)
    return min([delta] + [c + completion(A, LOG, STAB) for c, A in patterns(order, n, root)])


def build(delta, order, inject_dag=False):
    faces, n = color_code_faces(delta)
    L = nn_cat.ladder({5: 8, 7: 17}[delta], diagonals=False)
    L["ver_pairs"], L["ver_sites"] = L["ver_pairs"][:1], L["ver_sites"][:1]
    pos = {q: i for i, q in enumerate(order)}
    lf = [[pos[q] for q in f] for f in faces]
    c, G = nn_cat.make_nn_cat(lf, n, unfold=True, rounds=1, inject_dag=inject_dag, **L)
    return regenerate_detectors(merge_final_mpps(c)), L["root"]


if __name__ == "__main__":
    from nn_cat_fd import certify
    delta, N, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    kmax = int(sys.argv[4]) if len(sys.argv) > 4 else delta - 1
    faces, n = color_code_faces(delta)
    rng = random.Random(seed)
    for k in range(N):
        order = list(range(n))
        if k:
            rng.shuffle(order)
        for dag in (False, True):
            c, root = build(delta, order, dag)
            r = certify(c, kmax, proxy=False)["T"]
            if r["fd"] != 1:
                break
        pred = predict(order, faces, n, root, delta)
        fd = r["fd"] if isinstance(r["fd"], int) else delta
        print(json.dumps(dict(k=k, predicted=pred, certified=fd, match=pred == fd, malignant=r["malignant"],
                              order=order)), flush=True)
