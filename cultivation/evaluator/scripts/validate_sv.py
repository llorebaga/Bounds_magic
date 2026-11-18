"""Validate the TDEM against brute-force state-vector simulation of the real-T circuit (Gidney d1=3).
For every single physical fault, and for random pairs, compare
  * rejection:  skeleton detector mask D != 0  <=>  state-vector postselected norm = 0;
  * acceptance ratio and bad weight of the accepted output, from the Choi-state contraction vs state vector.
"""
import sys, time, random, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import numpy as np
import stim
from tdem import TDEM, n_results
from statevec import run

NOISE = ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR")
PAULIS2 = [(a, b) for a in "IXYZ" for b in "IXYZ" if a + b != "II"]


def physical_faults(m):
    """all single faults: ('pauli', i, ((q, P), ...)) after instruction i, or ('flip', i, k) of target k"""
    F = []
    for i, inst in enumerate(m.insts):
        nm = inst.name
        tg = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if nm == "DEPOLARIZE1":
            F += [("pauli", i, ((q, P),)) for q in tg for P in "XYZ"]
        elif nm == "DEPOLARIZE2":
            for a, b in zip(tg[::2], tg[1::2]):
                F += [("pauli", i, tuple((q, P) for q, P in ((a, Pa), (b, Pb)) if P != "I")) for Pa, Pb in PAULIS2]
        elif nm in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
            F += [("pauli", i, ((q, nm[0]),)) for q in tg]
        elif nm in ("M", "MX") and inst.gate_args_copy() and inst.gate_args_copy()[0] > 0:
            F += [("flip", i, k) for k in range(len(tg))]
    return F


def skeleton_symptoms(m, faults):
    ops = list(m.ops)
    inserts = {}
    flips = {}
    for f in faults:
        if f[0] == "pauli":
            inserts.setdefault(m.pos_after[f[1]], []).extend(f[2])
        else:
            flips.setdefault(m.pos_after[f[1]] - 1, set()).add(f[2])
    c = stim.Circuit()
    for pos, op in enumerate(ops):
        if op.name in NOISE:
            pass
        elif op.name in ("M", "MX", "MPP"):
            tg = op.targets_copy()
            if pos in flips:
                for k, x in enumerate(tg):
                    c.append(op.name, [x], [1.0] if k in flips[pos] else [])
            else:
                c.append(op.name, tg)
        else:
            c.append(op)
        for q, P in inserts.get(pos + 1, []):
            c.append(P + "_ERROR", [q], [1.0])
    tail = list(m.noisy)[len(list(m.body)):]
    for op in tail:
        c.append(op)
    det, obs = c.compile_detector_sampler().sample(1, separate_observables=True)
    D = int(det[0].any())
    o = 0
    for i, b in enumerate(obs[0]):
        if b:
            o |= 1 << i
    return D, o


def sv_faults(m, faults):
    d = {}
    for f in faults:
        if f[0] == "pauli":
            d.setdefault(f[1], []).extend(f[2])
        else:
            d[("flip", m.rec_start[f[1]] + f[2])] = 1
    return d


def main():
    G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
    f = G + "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"
    npairs = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    m = TDEM(stim.Circuit.from_file(f), magic="T", verbose=True)
    v0, n0 = run(m, {})
    u0 = v0 / np.sqrt(n0)
    F = physical_faults(m)
    print(f"{len(F)} single physical faults")
    random.seed(1)
    configs = [[x] for x in F] + [random.sample(F, 2) for _ in range(npairs)]
    stats = dict(n=0, rejected_agree=0, accepted=0, max_err_acc=0.0, max_err_bad=0.0, mismatch=0, harmful=0)
    t0 = time.time()
    for cfg in configs:
        D, o = skeleton_symptoms(m, cfg)
        v, nrm = run(m, sv_faults(m, cfg))
        acc_sv = nrm / n0
        bad_sv = acc_sv - (abs(np.vdot(u0, v)) ** 2 / n0 if v is not None else 0.0)
        stats["n"] += 1
        if D:
            if acc_sv < 1e-10:
                stats["rejected_agree"] += 1
            else:
                stats["mismatch"] += 1
                print("MISMATCH (D!=0 but accepted)", cfg, acc_sv)
            continue
        acc, bad = m.evaluate(o)
        stats["accepted"] += 1
        stats["harmful"] += int(bad > 1e-9)
        ea, eb = abs(acc - acc_sv), abs(bad - bad_sv)
        stats["max_err_acc"] = max(stats["max_err_acc"], ea)
        stats["max_err_bad"] = max(stats["max_err_bad"], eb)
        if ea > 1e-8 or eb > 1e-8:
            stats["mismatch"] += 1
            print("MISMATCH", cfg, (acc, bad), (acc_sv, bad_sv))
    stats["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(stats, indent=1))
    json.dump(stats, open(__file__.rsplit('scripts', 1)[0] + "results/validate_sv_d3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
