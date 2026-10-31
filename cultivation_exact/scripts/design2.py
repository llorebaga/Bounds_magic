"""Flag design v2 on Gidney's circuits.  Usage: python design2.py d1 lo hi [timing] [layer_weight]"""
import sys, time, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from flags2 import design

if __name__ == "__main__":
    G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
    NAMES = {"3": "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim",
             "5": "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim"}
    d1, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    timing = sys.argv[4] if len(sys.argv) > 4 else "aligned"
    lw = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    if d1.endswith(".stim"):
        c = stim.Circuit.from_file(d1)
        kw = {"complement": True}
        d1 = "5"
        label = sys.argv[1].replace("\\", "/").split("/")[-1].replace(".stim", "")
    else:
        c = stim.Circuit.from_file(G + NAMES[d1])
        kw = {"complement": True} if d1 == "5" else {}
        label = ""
    t0 = time.time()
    types = tuple(sys.argv[6].split(",")) if len(sys.argv) > 6 else ("Z", "X")
    res = design(c, 1e-3, int(d1), (lo, hi), tdem_kw=kw, timing=timing, layer_weight=lw, types=types, max_reps=int(sys.argv[7]) if len(sys.argv) > 7 else 2048, max_iter=30,
                 checkpoint=__file__.rsplit('scripts', 1)[0] + 'results/ckpt_' + label)
    tag = "d%s_%s_%d-%d_%s_lw%g_%s" % (d1, label, lo, hi, timing, lw, "-".join(types))
    print(f"RESULT {tag}: flags {res['chosen']}, failed={res.get('failed', False)}, {time.time()-t0:.0f}s", flush=True)
    base = __file__.rsplit('scripts', 1)[0] + "results/"
    open(base + f"design2_{tag}.stim", "w").write(str(res["circuit"]))
    json.dump({"flags": res["chosen"], "log": res["log"], "failed": res.get("failed", False)},
              open(base + f"design2_{tag}.json", "w"), indent=1, default=str)
