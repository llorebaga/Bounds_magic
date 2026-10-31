"""Fold-tree search with the exact X-type screen as objective: random grown trees, then local search (re-attach
one node / switch one pair's representative).  Objective: screen fault distance (higher better), then fewer
harmful combinations at that weight, then fewer layers.  Usage: python search_screen.py n_random n_local max_layers seed"""
import sys, json, random, time, collections
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'scripts')
import stim
from foldtree import D5Check, XScreen, screen_fd, set_pairing, random_pairing
from search_trees import configure, grow, G, FACES
from tdem import TDEM


def evaluate(k, reps, root, parent, xs, max_layers):
    configure(k, reps, root)
    lay = k.schedule(parent)
    if len(lay) > max_layers:
        return None
    kk, cnt = screen_fd(k, parent, lay, xs, kmax=4)
    return ((kk or 5), -cnt, -len(lay))


def neighbours_of(k, reps, root, parent, rng):
    """re-attach a random non-root node to another adjacent tree node (if it keeps a tree)"""
    reps = list(reps)
    par = dict(parent)
    v = rng.choice([x for x in reps if x != root])
    nb = [u for u in reps if u != v and abs(k.coords[u][0] - k.coords[v][0]) + abs(k.coords[u][1] - k.coords[v][1]) == 1]
    # descendants of v cannot be its new parent
    ch = collections.defaultdict(list)
    for a, b in par.items():
        ch[b].append(a)
    desc, st = set(), [v]
    while st:
        x = st.pop()
        for y in ch[x]:
            desc.add(y)
            st.append(y)
    opts = [u for u in nb if u not in desc and u != par[v]]
    if not opts:
        return None
    par[v] = rng.choice(opts)
    return reps, root, par


if __name__ == "__main__":
    n_rand, n_loc, max_layers, seed = map(int, sys.argv[1:5])
    rng = random.Random(seed)
    c = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
    probe = TDEM(c, magic="T", complement=True, light=True)
    xs = XScreen(probe.data, FACES, {q: (-1 if q in probe.flip_set else 1) for q in probe.data}, target_dag=True)
    k = D5Check(c)
    pool = []
    t0 = time.time()
    # Gidney's tree as a seed
    g0 = (list(k.reps), k.root, dict(k.parent), list(k.pairs))
    s0 = evaluate(k, *g0[:3], xs, 99)
    pool.append((s0, g0))
    print("Gidney:", s0, flush=True)
    vary_pairs = len(sys.argv) > 5 and sys.argv[5] == "pairs"
    for it in range(n_rand):
        if vary_pairs:
            set_pairing(k, random_pairing(k, rng))
        root = rng.choice([q for pr in k.pairs for q in pr])
        reps, parent = grow(k, rng, root, bias=rng.choice([0.0, 2.0, 5.0]))
        if reps is None:
            continue
        s = evaluate(k, reps, root, parent, xs, max_layers)
        if s:
            pool.append((s, (reps, root, parent, list(k.pairs))))
    pool.sort(key=lambda x: x[0], reverse=True)
    print(f"random phase: best {pool[0][0]} ({time.time()-t0:.0f}s)", flush=True)
    # local search from the top 10
    for start in pool[:10]:
        cur_s, cur = start
        set_pairing(k, cur[3])
        for _ in range(n_loc):
            nb = neighbours_of(k, *cur[:3], rng)
            if nb is None:
                continue
            s = evaluate(k, *nb, xs, max_layers)
            if s and s >= cur_s:
                cur_s, cur = s, (*nb, cur[3])
        pool.append((cur_s, cur))
        print(f"  local search -> {cur_s} ({time.time()-t0:.0f}s)", flush=True)
    pool.sort(key=lambda x: x[0], reverse=True)
    out = [[list(s), r[0], r[1], sorted(r[2].items()), r[3]] for s, r in pool[:30]]
    tag = "_pairs" if vary_pairs else ""
    json.dump(out, open(__file__.rsplit('scripts', 1)[0] + f"results/screen_search_L{max_layers}_s{seed}{tag}.json", "w"))
    print("best:", pool[0][0], flush=True)
