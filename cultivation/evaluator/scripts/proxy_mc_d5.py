"""S-proxy logical error of Gidney's d1=5 circuit by direct stim sampling (the proxy is a Clifford circuit), and its
exact leading term. Every run with a detection event is discarded, and an accepted run fails when it flips the
observable. The leading term sums the malignant configurations of the least weight (fd.fault_distance on the TDEM of
the proxy). Usage: python proxy_mc_d5.py MODEL P SHOTS [WORKERS]
-> results/proxy_mc_d5_<model>_<p>.json"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import multiprocessing as mp
import numpy as np
import stim
from noise import si1000, uniform

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PATH = os.path.join(ROOT, "external", "gidney_circuits",
                    "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
BATCH = 1 << 20


def circuit(model, p):
    c0 = stim.Circuit.from_file(PATH)
    return si1000(c0, p) if model == "si1000" else uniform(c0, p)


def work(args):
    model, p, shots, seed = args
    s = circuit(model, p).compile_detector_sampler(seed=seed)
    acc = bad = 0
    done = 0
    while done < shots:
        n = min(BATCH, shots - done)
        det, obs = s.sample(n, separate_observables=True, bit_packed=True)
        ok = ~det.any(axis=1)
        acc += int(ok.sum())
        bad += int(obs[ok].any(axis=1).sum())
        done += n
    return acc, bad


def leading(c, kmax=4):
    """least weight, number and total weight (probability times bad weight) of the malignant configurations of the
    proxy, from the full enumeration of fd.fault_distance (the fast search does not keep weights at four faults)"""
    from tdem import TDEM
    from fd import fault_distance
    m = TDEM(c, magic="S")
    k, bad, mech = fault_distance(m, kmax, m.mechanisms())
    return k, len(bad), sum(b[4] * b[3] for b in bad)


if __name__ == "__main__":
    model, p, shots = sys.argv[1], float(sys.argv[2]), int(float(sys.argv[3]))
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
    c = circuit(model, p)
    t0 = time.time()
    k, nmal, w = leading(c, 4)
    out = dict(model=model, p=p, proxy_fd=k, malignant=nmal, leading_P_bad=w, seconds_leading=round(time.time() - t0))
    print(json.dumps(out), flush=True)
    t0 = time.time()
    per = shots // workers
    with mp.Pool(workers) as pool:
        res = pool.map(work, [(model, p, per, 1000 + i) for i in range(workers)])
    acc = sum(a for a, b in res)
    bad = sum(b for a, b in res)
    out.update(shots=per * workers, accepted=acc, failures=bad, P_acc=acc / (per * workers),
               eps=bad / acc if acc else None, se_eps=(bad ** 0.5) / acc if acc else None,
               leading_eps=w / (acc / (per * workers)) if acc else None, seconds_sampling=round(time.time() - t0))
    print(json.dumps(out), flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", f"proxy_mc_d5_{model}_{p:g}.json"), "w"), indent=1)
