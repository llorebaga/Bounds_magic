"""Exact certification of the best fold-tree designs: rebuild, regenerate detectors, TDEM fault distance (real T and
S proxy), malignant-class count, then exact eps by meet-in-the-middle at uniform / SI1000 p, against Gidney.
Usage: python certify_trees.py n_designs"""
import sys, json, glob, time
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'scripts')
import stim
from foldtree import D5Check, regenerate_detectors, set_pairing
from search_trees import configure, G
from tdem import TDEM
from fd_fast import fault_distance_fast
from mitm_rate import MITM
from mc import sample_parallel
from noise import uniform, si1000


def rates(c0, tag):
    out = []
    for model in ["uniform", "si1000"]:
        c1 = (si1000 if model == "si1000" else uniform)(c0, 1e-3)
        m = TDEM(c1, magic="T", complement=True)
        est = MITM(m)
        for p in [1e-3, 5e-4]:
            pb, se, _ = est.P_bad(p / 1e-3, 8, 2000)
            c = (si1000 if model == "si1000" else uniform)(c0, p)
            acc = sample_parallel(c, "T", int(5e6), workers=3, chunks=6, complement=True)["P_acc"]
            out.append(dict(design=tag, model=model, p=p, P_acc=acc, eps=pb / acc, se_eps=se / acc))
            print(f"  {tag} {model} p={p:g}: acc {acc:.4f} eps {pb/acc:.3e} +- {se/acc:.1e}", flush=True)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    base = __file__.rsplit('scripts', 1)[0] + "results/"
    c = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
    cands = []
    for f in glob.glob(base + "screen_search_*.json"):
        for row in json.load(open(f)):
            s, reps, root, parent = row[:4]
            pairs = row[4] if len(row) > 4 else None
            cands.append((tuple(s), reps, root, parent, pairs))
    cands.sort(key=lambda x: x[0], reverse=True)
    seen, best = set(), []
    for cd in cands:
        key = (tuple(cd[1]), cd[2], tuple(map(tuple, cd[3])), tuple(map(tuple, cd[4] or [])))
        if key not in seen:
            seen.add(key)
            best.append(cd)
    which = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    results = []
    if which < 0:
        results += rates(regenerate_detectors(c), "Gidney")
    else:
        s, reps, root, parent, pairs = best[which]
        k = D5Check(c)
        if pairs:
            set_pairing(k, pairs)
        configure(k, reps, root)
        cc, layers = k.build(dict(parent))
        cc = regenerate_detectors(cc)
        tag = f"tree{which}"
        open(base + f"design_{tag}.stim", "w").write(str(cc))
        t0 = time.time()
        fds = {}
        for magic, kw in [("T", {"complement": True}), ("S", {})]:
            m = TDEM(cc, magic=magic, **kw)
            kf, found = fault_distance_fast(m, 3, m.mechanisms(), verbose=False)
            fds[magic] = (kf, len(found), sum(f[4] * f[3] for f in found) / 1e-3 ** kf if kf else 0)
        print(f"{tag}: screen {s}, {len(layers)} layers, root {root}, pairs {'Gidney' if not pairs else 'new'}: "
              f"real T {fds['T']}, proxy {fds['S']} ({time.time()-t0:.0f}s)", flush=True)
        results.append(dict(design=tag, layers=len(layers), real_T=fds["T"], proxy=fds["S"]))
        if fds["S"][0] is None:
            results += rates(cc, tag)
    json.dump(results, open(base + f"certify_{'gidney' if which < 0 else which}.json", "w"), indent=1)
