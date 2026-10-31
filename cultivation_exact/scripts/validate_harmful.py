"""Validate the harmful (malignant) fault pairs of Gidney d1=3 with real T by state vector."""
import sys, time, json, collections, itertools
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src'); sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'scripts')
import numpy as np, stim
from tdem import TDEM
from statevec import run
import validate_sv as V

def symptoms(m, cfg):
    ops_D = None
    # full detector mask version of V.skeleton_symptoms
    c = stim.Circuit()
    inserts, flips = {}, {}
    for f in cfg:
        if f[0] == "pauli": inserts.setdefault(m.pos_after[f[1]], []).extend(f[2])
        else: flips.setdefault(m.pos_after[f[1]] - 1, set()).add(f[2])
    for pos, op in enumerate(m.ops):
        if op.name in V.NOISE: pass
        elif op.name in ("M", "MX", "MPP"):
            tg = op.targets_copy()
            if pos in flips:
                for k, x in enumerate(tg): c.append(op.name, [x], [1.0] if k in flips[pos] else [])
            else: c.append(op.name, tg)
        else: c.append(op)
        for q, P in inserts.get(pos + 1, []): c.append(P + "_ERROR", [q], [1.0])
    for op in list(m.noisy)[len(list(m.body)):]: c.append(op)
    det, obs = c.compile_detector_sampler().sample(1, separate_observables=True)
    D = sum(1 << i for i, b in enumerate(det[0]) if b); o = sum(1 << i for i, b in enumerate(obs[0]) if b)
    return D, o

G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
m = TDEM(stim.Circuit.from_file(G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"), magic="T")
v0, n0 = run(m, {}); u0 = v0 / np.sqrt(n0)
F = V.physical_faults(m)
sym = [symptoms(m, [f]) for f in F]
byD = collections.defaultdict(list)
for i, (D, o) in enumerate(sym): byD[D].append(i)
mal = []
for D, idx in byD.items():
    for i, j in itertools.combinations(idx, 2):
        acc, bad = m.evaluate(sym[i][1] ^ sym[j][1])
        if bad > 1e-9: mal.append((i, j, acc, bad))
print(len(mal), "malignant physical pairs (TDEM)")
worst = 0
for i, j, acc, bad in mal:
    v, nrm = run(m, V.sv_faults(m, [F[i], F[j]]))
    acc_sv = nrm / n0; bad_sv = acc_sv - abs(np.vdot(u0, v)) ** 2 / n0
    worst = max(worst, abs(acc - acc_sv), abs(bad - bad_sv))
print("max deviation from state vector:", worst)
ex = mal[0]; print("example:", F[ex[0]], str(m.insts[F[ex[0]][1]])[:60], "|", F[ex[1]], str(m.insts[F[ex[1]][1]])[:60], "acc", ex[2], "bad", ex[3])
json.dump({"malignant_pairs": len(mal), "max_dev": worst, "acc_values": sorted({round(x[2], 6) for x in mal}), "bad_values": sorted({round(x[3], 6) for x in mal})},
          open(__file__.rsplit('scripts', 1)[0] + "results/validate_harmful_d3.json", "w"), indent=1)
