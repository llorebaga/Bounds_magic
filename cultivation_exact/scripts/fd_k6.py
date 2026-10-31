"""Exact weight-6 certificate for a real-T circuit of fault distance >= 6 (no malignant set of at most five faults).

A set of six fault classes with total detector pattern zero splits into two triples with equal detector patterns,
D(t1) = D(t2). We enumerate all triples of classes with their detector hash h (a random GF(2)-linear 64-bit hash)
and observable mask o, partitioned by the top bits of h so that each worker holds one bucket. Inside a bucket,
for every hash value, every XOR of two distinct observable masks that occur with it is a candidate error; it is
evaluated exactly with the Gauss sums. This over-approximates the malignant 6-sets: hash collisions, triples that
share classes (whose union is a smaller set, harmless when the fault distance is >= 6) and orderings only add
candidates, so finding no harmful candidate certifies that there is no malignant set of six faults. Harmful
candidates, if any, are reported with one triple pair each for an exact re-check.
With K = 5 the other side holds pairs instead of triples (a check of the method against leading_k5.py).
Usage: python fd_k6.py PATH [WORKERS] [BUCKET_BITS] [K]   -> results/<stem>_k<K>.json"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import multiprocessing as mp
import numpy as np
import stim

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def setup(path):
    from tdem import TDEM
    from branch import FastExact
    from fd_fast import _hash
    m = TDEM(stim.Circuit.from_file(path), magic="T", pattern=False)
    mech = [x for x in m.mechanisms() if x[1] or x[2]]
    n = len(mech)
    Dm = [x[1] for x in mech]
    H = _hash(Dm, max(D.bit_length() for D in Dm) + 1).astype(np.uint64)
    nobs = len(m.gens)
    W = (nobs + 63) // 64
    O = np.zeros((n, W), dtype=np.uint64)
    for i, x in enumerate(mech):
        for w in range(W):
            O[i, w] = np.uint64((x[2] >> (64 * w)) & ((1 << 64) - 1))
    return m, FastExact(m), H, O, n, W


def to_int(row):
    return sum(int(v) << (64 * w) for w, v in enumerate(row))


def work(args):
    path, bucket, bits, k = args
    m, fe, H, O, n, W = setup(path)
    iu, ju = np.triu_indices(n, 1)
    hp = H[iu] ^ H[ju]
    op = O[iu] ^ O[ju]
    shift = np.uint64(64 - bits)
    hs, os_, src = [], [], []
    for a in range(n - 2):
        sel = iu > a
        h = H[a] ^ hp[sel]
        keep = (h >> shift) == np.uint64(bucket)
        if not keep.any():
            continue
        idx = np.nonzero(sel)[0][keep]
        hs.append(h[keep])
        os_.append(O[a] ^ op[idx])
        src.append(np.column_stack([np.full(len(idx), a), iu[idx], ju[idx]]))
    if k == 5:                                   # pairs on the other side (tag: third index -1)
        keep = (hp >> shift) == np.uint64(bucket)
        hs.append(hp[keep]); os_.append(op[keep])
        src.append(np.column_stack([iu[keep], ju[keep], np.full(int(keep.sum()), -1)]))
    if not hs:
        return dict(bucket=bucket, triples=0, candidates=0, harmful=[])
    h = np.concatenate(hs)
    o = np.concatenate(os_)
    s = np.concatenate(src)
    order = np.lexsort([o[:, w] for w in range(W - 1, -1, -1)] + [h])
    h, o, s = h[order], o[order], s[order]
    starts = np.r_[0, np.nonzero(np.diff(h))[0] + 1]
    ends = np.r_[starts[1:], len(h)]
    cache, harmful, cand = {}, [], 0
    for st, en in zip(starts, ends):
        if en - st < 2:
            continue
        oo = o[st:en]
        if k == 5:
            tri = s[st:en, 2] >= 0
            if tri.all() or not tri.any():
                continue
            ut, ft = np.unique(oo[tri], axis=0, return_index=True)
            up, fp = np.unique(oo[~tri], axis=0, return_index=True)
            ia, ib = np.meshgrid(np.arange(len(ut)), np.arange(len(up)), indexing="ij")
            ia, ib = ia.ravel(), ib.ravel()
            X = ut[ia] ^ up[ib]
            first = np.r_[np.nonzero(tri)[0][ft], np.nonzero(~tri)[0][fp]]
            ib = ib + len(ut)
        else:
            uniq, first = np.unique(oo, axis=0, return_index=True)
            if len(uniq) < 2:
                continue
            ia, ib = np.triu_indices(len(uniq), 1)
            X = uniq[ia] ^ uniq[ib]
        for t in range(len(X)):
            key = to_int(X[t])
            if key in cache:
                continue
            cand += 1
            acc, bad = fe.evaluate(key)
            cache[key] = bad
            if bad > 1e-9 and len(harmful) < 50:
                t1 = s[st + first[ia[t]]].tolist()
                t2 = s[st + first[ib[t]]].tolist()
                harmful.append(dict(o=hex(key), bad=bad, acc=acc, triples=[t1, t2]))
    return dict(bucket=bucket, triples=int(len(h)), candidates=cand, harmful=harmful)


if __name__ == "__main__":
    path = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    bits = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    t0 = time.time()
    with mp.Pool(workers) as pool:
        parts = []
        for r in pool.imap_unordered(work, [(path, b, bits, k) for b in range(2 ** bits)]):
            parts.append(r)
            if r["harmful"]:
                print("bucket", r["bucket"], "harmful", len(r["harmful"]), flush=True)
    harmful = [x for p in parts for x in p["harmful"]]
    out = dict(circuit=os.path.basename(path), k=k, triples=sum(p["triples"] for p in parts),
               candidates_evaluated=sum(p["candidates"] for p in parts), harmful_candidates=len(harmful),
               examples=harmful[:20], seconds=round(time.time() - t0))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}), flush=True)
    json.dump(out, open(os.path.join(R, os.path.splitext(os.path.basename(path))[0] + f"_k{k}.json"), "w"), indent=1)
