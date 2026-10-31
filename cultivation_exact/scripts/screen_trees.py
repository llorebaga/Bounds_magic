"""Phase 2 of the fold-tree search: exact X-type screen (XScreen, up to 4 faults) on every stored tree with hook
distance >= 4, then exact TDEM certification of the best.  Usage: python screen_trees.py [n_certify]"""
import sys, json, glob, time
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'scripts')
import stim
from foldtree import D5Check, XScreen, screen_fd, regenerate_detectors
from search_trees import configure, G, FACES
from tdem import TDEM
from fd_fast import fault_distance_fast

if __name__ == "__main__":
    n_cert = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    base = __file__.rsplit('scripts', 1)[0] + "results/"
    c = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
    probe = TDEM(c, magic="T", complement=True, light=True)
    xs = XScreen(probe.data, FACES, {q: (-1 if q in probe.flip_set else 1) for q in probe.data}, target_dag=True)
    trees = {}
    for f in glob.glob(base + "tree_search_*.json"):
        for hd, nl, reps, root, parent in json.load(open(f)):
            if hd >= 4:
                trees[(tuple(reps), root, tuple(map(tuple, parent)))] = -nl
    print(len(trees), "distinct trees with hook distance >= 4", flush=True)
    k = D5Check(c)
    scored = []
    t0 = time.time()
    for (reps, root, parent), nl in trees.items():
        configure(k, list(reps), root)
        par = dict(parent)
        kk, cnt = screen_fd(k, par, k.schedule(par), xs, kmax=4)
        scored.append(((kk or 5), -cnt, -nl, reps, root, parent))
    scored.sort(key=lambda s: (-s[0], s[2], -s[1]))
    print("screen fault distance histogram:", {d: sum(1 for s in scored if s[0] == d) for d in (2, 3, 4, 5)},
          f"({time.time()-t0:.0f}s)", flush=True)
    json.dump([[s[0], s[1], s[2], list(s[3]), s[4], [list(e) for e in s[5]]] for s in scored[:200]],
              open(base + "tree_screened.json", "w"))
    for s in scored[:n_cert]:
        sd, ncnt, nl, reps, root, parent = s
        configure(k, list(reps), root)
        cc, layers = k.build(dict(parent))
        cc = regenerate_detectors(cc)
        name = base + f"tree_screen{sd}_L{nl}_r{root}.stim"
        open(name, "w").write(str(cc))
        t1 = time.time()
        res = {}
        for magic, kw in [("T", {"complement": True}), ("S", {})]:
            m = TDEM(cc, magic=magic, **kw)
            kfd, found = fault_distance_fast(m, 3, m.mechanisms(), verbose=False)
            res[magic] = (kfd, len(found))
        print(f"screen FD {sd} (count {-ncnt}), {nl} layers, root {root}: TDEM real T {res['T']}, proxy {res['S']} "
              f"({time.time()-t1:.0f}s) -> {name}", flush=True)
