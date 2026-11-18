import sys, time, json, collections
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim, numpy as np
from tdem import best_reference2 as best_reference
from fd_fast import fault_distance_fast
from faults import FaultTable
from noise import si1000
G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
c = stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
magic, kmax = sys.argv[1], int(sys.argv[2])
if len(sys.argv) > 3 and sys.argv[3] == "si1000":
    c = si1000(c, 1e-3)
t0 = time.time()
m = best_reference(c, magic, verbose=True)
print("overrides", m.overrides_used, "bloch", [round(x, 3) for x in m.output_bloch()], flush=True)
k, found = fault_distance_fast(m, kmax, m.mechanisms())
coef = sum(f[4] * f[3] for f in found if f[4] is not None)
print(f"magic={magic}: FD {k}, {len(found)} malignant, leading P_bad coefficient {coef / 1e-3 ** k if k else 0:.4g} p^{k}; {time.time()-t0:.0f}s", flush=True)
if k and k <= 3:
    ft = FaultTable(m); cl = {(D, o): fs for p, D, o, fs in ft.classes()}
    mech = [x for x in m.mechanisms() if x[1] or x[2]]
    ops = list(m.noisy.flattened()); tk = []; t = -1
    for op in ops:
        tk.append(t)
        if op.name == "TICK": t += 1
    cnt = collections.Counter()
    for idx, o, acc, bad, prob in found:
        key = tuple(sorted(tk[ft.where[cl[(mech[i][1], mech[i][2])][0]]] for i in idx))
        cnt[(key, round(acc, 3), round(bad, 3))] += 1
    for kk, v in cnt.most_common(15):
        print("   ticks", kk[0], "acc", kk[1], "bad", kk[2], "x", v)
