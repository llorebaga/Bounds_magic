"""Random + local search over fold trees of Gidney's d=5 double check: representative (data or partner) of each
pair, root and nearest-neighbour spanning tree.  Objective: hook distance (target 5), then fewer fold layers,
then fewer minimum-cardinality hook combinations.  Usage: python search_trees.py n_random max_layers seed"""
import sys, json, random, collections, time
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from foldtree import D5Check, fault_supports, min_weight_logicals, hook_distance

G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
FACES = [[0, 3, 7, 11], [9, 13, 15, 19, 21, 27], [17, 25, 32, 36], [3, 7, 9, 13], [15, 21, 23, 29],
         [19, 25, 27, 32, 34, 38], [7, 11, 13, 17, 19, 25], [21, 27, 29, 34], [32, 36, 38, 40]]


def configure(k, reps, root):
    k.reps = sorted(reps)
    k.root = root
    k.adj = collections.defaultdict(set)
    for a in k.reps:
        for b in k.reps:
            if a < b and abs(k.coords[a][0] - k.coords[b][0]) + abs(k.coords[a][1] - k.coords[b][1]) == 1:
                k.adj[a].add(b)
                k.adj[b].add(a)
    seen = {root}
    st = [root]
    while st:
        v = st.pop()
        for n in k.adj[v]:
            if n not in seen:
                seen.add(n)
                st.append(n)
    return len(seen) == len(k.reps)


def score(k, parent, logs, kmax=5):
    lay = k.schedule(parent)
    sup = fault_supports(k, parent, lay)
    hd = hook_distance(sup, logs, kmax=kmax)
    return (hd if hd is not None else kmax + 1), len(lay)


def grow(k, rng, root, bias=0.0):
    """random tree grown from the root: repeatedly attach a member of an uncovered pair that is a grid neighbour
    of a chosen representative (bias > 0 prefers attaching to shallow nodes -> fewer layers)"""
    pair_of = {}
    for a, b in k.pairs:
        pair_of[a] = (a, b)
        pair_of[b] = (a, b)
    nb = lambda x, y: abs(k.coords[x][0] - k.coords[y][0]) + abs(k.coords[x][1] - k.coords[y][1]) == 1
    chosen = {root}
    covered = {pair_of[root]}
    parent = {}
    depth = {root: 0}
    allq = [q for pr in k.pairs for q in pr]
    while len(covered) < len(k.pairs):
        opts = [(v, u) for v in chosen for u in allq if pair_of[u] not in covered and nb(v, u)]
        if not opts:
            return None, None
        if bias:
            opts.sort(key=lambda e: depth[e[0]] + rng.random() * bias)
            v, u = opts[0]
        else:
            v, u = rng.choice(opts)
        parent[u] = v
        depth[u] = depth[v] + 1
        chosen.add(u)
        covered.add(pair_of[u])
    return sorted(chosen), parent


if __name__ == "__main__":
    n_rand, max_layers, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rng = random.Random(seed)
    c = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
    k = D5Check(c)
    data = sorted({b for a, b in k.pairs})
    w, logs = min_weight_logicals(data, FACES, data)
    best = []
    t0 = time.time()
    tried = 0
    for it in range(n_rand):
        root = rng.choice([q for pr in k.pairs for q in pr])
        reps, parent = grow(k, rng, root, bias=rng.choice([0.0, 2.0, 5.0]))
        if reps is None:
            continue
        configure(k, reps, root)
        tried += 1
        for _ in range(1):
            hd, nl = score(k, parent, logs)
            if nl <= max_layers and hd >= 4:
                best.append((hd, -nl, sorted(reps), root, sorted(parent.items())))
        if it % 2000 == 0:
            best.sort(reverse=True)
            top = best[0] if best else None
            print(f"{it}: {tried} connected rep sets; best hook distance {top[0] if top else None} with {-top[1] if top else None} layers "
                  f"({time.time()-t0:.0f}s)", flush=True)
    best.sort(reverse=True)
    json.dump(best, open(__file__.rsplit('scripts', 1)[0] + f"results/tree_search_L{max_layers}_s{seed}.json", "w"))
    print("hook-distance histogram of kept:", collections.Counter(b[0] for b in best))
    print("best:", best[0][:4] if best else None)
