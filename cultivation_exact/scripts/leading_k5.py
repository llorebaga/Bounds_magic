"""Exact weight-5 failure term of a real-T circuit of fault distance 5 (every malignant 5-set of fault classes,
cluster_search.k5 run locally on all cores). MODEL = file (the circuit's own noise, p = 1e-3) or si1000.
Usage: python leading_k5.py PATH MODEL [WORKERS]   -> results/<stem>_<model>_c5.json"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import multiprocessing as mp
import stim

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def setup(path, model):
    from tdem import TDEM
    from branch import FastExact
    from cluster_search import Classes
    from noise import si1000
    c = stim.Circuit.from_file(path)
    if model == "si1000":
        c = si1000(c, 1e-3)
    m = TDEM(c, magic="T", pattern=False)
    C = Classes(m)
    C.m = FastExact(m)
    return C


def work(args):
    path, model, shard, nshards = args
    from cluster_search import k5
    return k5(setup(path, model), shard, nshards)


if __name__ == "__main__":
    path, model = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    t0 = time.time()
    nshards = 4 * workers
    with mp.Pool(workers) as pool:
        parts = pool.map(work, [(path, model, s, nshards) for s in range(nshards)])
    cnt = sum(x["malignant"] for x in parts)
    w = sum(x["weight"] for x in parts)
    out = dict(circuit=os.path.basename(path), model=model, malignant_5sets=cnt, weight=w, c5=w / 1e-3 ** 5,
               seconds=round(time.time() - t0))
    print(json.dumps(out), flush=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    json.dump(out, open(os.path.join(R, f"{stem}_{model}_c5.json"), "w"), indent=1)
