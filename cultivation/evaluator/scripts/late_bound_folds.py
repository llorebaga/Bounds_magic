"""Late-error bound for the GHZ-fold checks of Proposition (fold) and the unverified tree cat, from the fold tree
alone (pieces = single-fault X patterns at the U layer, hook_bound.supports), with the exact completion cost of the
late-error theorem. Usage: python late_bound_folds.py DELTA SEEDS SMAX"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
import hook_bound
from late_bound import CodeTools

delta, seeds, smax = int(sys.argv[1]), sys.argv[2].split(","), int(sys.argv[3])
faces, n = color_code_faces(delta)
ct = CodeTools(faces, n)
allq = (1 << n) - 1
for sd in seeds:
    t0 = time.time()
    if sd == "cat":
        sup = hook_bound.supports(n, hook_bound.cat_ops(n))
    else:
        parent, root, reps = hook_bound.tree(n, faces, sd)
        sup = hook_bound.supports(n, hook_bound.check_ops(n, parent, root, reps))
    vec = sorted({min(m, m ^ allq) for m in (sum(1 << q for q in s) for s in sup)})
    best, wit = delta, None
    level = {0: ()}
    seen = {0}
    for s in range(1, smax + 1):
        nxt = {}
        for A, w in level.items():
            for i, a in enumerate(vec):
                if w and i <= w[-1]:
                    continue
                B = A ^ a
                B = min(B, B ^ allq)
                if B not in seen and B not in nxt:
                    nxt[B] = w + (i,)
        seen |= set(nxt)
        if s >= best:
            break
        for A, w in nxt.items():
            if A:
                c = min(ct.comp(A, best - s), ct.comp(A ^ allq, best - s))
                if s + c < best:
                    best, wit = s + c, (s, [q for q in range(n) if A >> q & 1])
        level = nxt
    print(json.dumps(dict(delta=delta, tree=sd, pieces=len(vec), late_bound=best, witness=wit,
                          s=round(time.time() - t0, 1))), flush=True)
