import sys, time, json; sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from flags import design, split_ticks
from fd import fault_distance
G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
c = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim")
lo, hi = int(sys.argv[1]), int(sys.argv[2])
t0 = time.time()
flags, log, cur, m = design(c, 1e-3, 3, region=(lo, hi), layer_weight=10.0)
print(f"{len(flags)} flags {flags}; {time.time()-t0:.0f}s")
open(__file__.rsplit('scripts', 1)[0] + f"results/flagged_d3.stim", "w").write(str(cur))
json.dump({"flags": flags, "log": log}, open(__file__.rsplit('scripts', 1)[0] + "results/flags_d3.json", "w"), indent=1)
