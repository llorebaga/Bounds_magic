"""S-proxy weight-5 or weight-6 search, with the meet-in-the-middle of fd_k6.py on the stim detector error model of
the proxy circuit (the circuit file read as written, S gates included). A configuration is malignant when its
detector pattern cancels and it flips the logical observable. Hash collisions and triples that share classes only
add candidates, so every reported candidate is re-checked on the exact detector patterns; no exact witness and no
candidate at all certifies that the proxy has no malignant set of K faults (given none of fewer, from proxy_fd.py).
Usage: python proxy_k56.py PATH [WORKERS] [BUCKET_BITS] [K]   -> results/<stem>_proxy_k<K>.json"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
import numpy as np
import stim
import fd_k6
from fd_fast import _hash

R = fd_k6.R


class Clifford:
    def evaluate(self, o):
        return (1.0, 1.0 if o else 0.0)


def mechanisms(path):
    c = stim.Circuit.from_file(path)
    mech = set()
    for inst in c.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True).flattened():
        if inst.type != "error":
            continue
        D = o = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                D ^= 1 << t.val
            elif t.is_logical_observable_id():
                o ^= 1 << t.val
        if D or o:
            mech.add((D, o))
    return sorted(mech)


def setup(path):
    mech = mechanisms(path)
    Dm = [x[0] for x in mech]
    H = _hash(Dm, max(D.bit_length() for D in Dm) + 1).astype(np.uint64)
    O = np.array([[x[1]] for x in mech], dtype=np.uint64)
    return None, Clifford(), H, O, len(mech), 1


fd_k6.setup = setup          # also applied in spawned workers, which re-import this module


if __name__ == "__main__":
    path = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    bits = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    mech = mechanisms(path)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        parts = list(pool.imap_unordered(fd_k6.work, [(path, b, bits, k) for b in range(2 ** bits)]))
    cands = [x for p in parts for x in p["harmful"]]
    witnesses = []
    for x in cands:                                   # exact re-check of each reported candidate
        t1, t2 = x["triples"]
        s1 = [i for i in t1 if i >= 0]
        s2 = [i for i in t2 if i >= 0]
        if set(s1) & set(s2):
            continue
        D = o = 0
        for i in s1 + s2:
            D ^= mech[i][0]
            o ^= mech[i][1]
        if D == 0 and o:
            witnesses.append(sorted(s1 + s2))
    out = dict(circuit=os.path.basename(path), k=k, classes=len(mech), triples=sum(p["triples"] for p in parts),
               candidates_evaluated=sum(p["candidates"] for p in parts), candidates_reported=len(cands),
               exact_witnesses=len(witnesses), example=witnesses[:3], seconds=round(time.time() - t0))
    print(json.dumps(out), flush=True)
    json.dump(out, open(os.path.join(R, os.path.splitext(os.path.basename(path))[0] + f"_proxy_k{k}.json"), "w"),
              indent=1)
