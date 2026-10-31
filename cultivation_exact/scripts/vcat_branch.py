import sys, itertools, json, time
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
c = stim.Circuit.from_file(__file__.rsplit('scripts', 1)[0] + "results/vcat_d3_T_merged.stim")
out = {}
for magic in "ST":
    base = TDEM(c, magic=magic, light=True)
    fr = base.free_records
    best = None
    weights = {}
    t0 = time.time()
    for bits in itertools.product([0, 1], repeat=len(fr)):
        try:
            mm = TDEM(c, magic=magic, light=True, overrides=dict(zip(fr, bits)))
        except RuntimeError:
            continue
        w = round(mm.n0, 12)
        weights[w] = weights.get(w, 0) + 1
        if best is None or mm.n0 > best[0] + 1e-12:
            best = (mm.n0, bits)
    print(magic, "weights histogram", weights, "best", best, f"{time.time()-t0:.0f}s", flush=True)
    out[magic] = dict(free=fr, best=list(best[1]) if best else None)
json.dump(out, open(__file__.rsplit('scripts', 1)[0] + "results/vcat_branch.json", "w"))
