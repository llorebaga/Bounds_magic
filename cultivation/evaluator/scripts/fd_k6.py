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

Undetected classes. The hash-zero group pairs all triples with zero detector pattern and can be far too large (286,002
triples for the flagged delta = 7 verified cat). It is not needed in full: if a set S of six classes has D(S) = 0 and
every split of S into two triples has D(t1) = 0, then all twenty triples of S are undetected, so all six classes have
the same pattern x with 3x = x = 0. Every other S has a split with D(t1) = D(t2) != 0 and is found in the group of that
pattern. So in the hash-zero group we keep only triples with a nonzero exact pattern (hash collisions), and the sets of
six undetected classes are decided separately (zero_sets): their errors lie in the span of the undetected classes,
a breadth-first search gives the least number of classes for every element of the span, and every element reached
with at most K classes is evaluated exactly.
Usage: python fd_k6.py PATH [WORKERS] [BUCKET_BITS] [K] [BUCKETS]   -> results/<stem>_k<K>.json
BUCKETS ("a-b" or "a,b,c", default all) restricts the run; every finished bucket is appended to
results/<stem>_k<K>_buckets.jsonl and skipped on a rerun, and the final json is written once all buckets are there."""
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


def exact_patterns(m):
    """exact detector patterns of the classes, in the order of setup() (None for a Clifford proxy)"""
    if m is None:
        return None
    return [x[1] for x in m.mechanisms() if x[1] or x[2]]


def zero_sets(path, k):
    """decide every set of at most k classes with zero detector pattern each (see the module docstring)"""
    m, fe, H, O, n, W = setup(path)
    Dm = exact_patterns(m)
    Zc = [i for i in range(n) if Dm[i] == 0]
    masks = [to_int(O[i]) for i in Zc]
    basis, coords = [], []                         # GF(2) basis of the span, with pivot bits
    for v in masks:
        for b, piv in basis:
            if (v >> piv) & 1:
                v ^= b
        if v:
            piv = v.bit_length() - 1
            for j, (b, p) in enumerate(basis):     # keep the basis fully reduced
                if (b >> piv) & 1:
                    basis[j] = (b ^ v, p)
            basis.append((v, piv))
    r = len(basis)

    def coord(v):
        c = 0
        for j, (b, piv) in enumerate(basis):
            if (v >> piv) & 1:
                v ^= b
                c |= 1 << j
        assert v == 0
        return c

    gen = np.array(sorted({coord(v) for v in masks}), dtype=np.int64)
    dist = np.full(1 << r, -1, dtype=np.int64)
    dist[0] = 0
    frontier = np.array([0], dtype=np.int64)
    for level in range(1, k + 1):                  # a shortest path never uses a class twice
        nxt = np.unique((frontier[:, None] ^ gen[None, :]).ravel())
        nxt = nxt[dist[nxt] < 0]
        dist[nxt] = level
        frontier = nxt
    reached = np.nonzero(dist > 0)[0]
    harmful = []
    for c in reached.tolist():
        v = 0
        for j, (b, piv) in enumerate(basis):
            if (c >> j) & 1:
                v ^= b
        acc, bad = fe.evaluate(v)
        if bad > 1e-9:
            harmful.append(dict(o=hex(v), bad=bad, acc=acc, classes=int(dist[c])))
    return dict(undetected_classes=len(Zc), span_rank=r, reached=int(len(reached)),
                by_weight={int(w): int((dist == w).sum()) for w in range(1, k + 1)}, harmful=harmful[:50],
                harmful_count=len(harmful))


def work(args):
    path, bucket, bits, k = args
    m, fe, H, O, n, W = setup(path)
    Dm = exact_patterns(m) if bucket == 0 else None
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
    cache, harmful, cand, dropped = {}, [], 0, 0
    for st, en in zip(starts, ends):
        if en - st < 2:
            continue
        rows = np.arange(st, en)
        if Dm is not None and h[st] == 0:            # hash-zero group: drop triples that are exactly undetected
            exact = np.array([(Dm[int(s[r, 0])] ^ Dm[int(s[r, 1])] ^ (Dm[int(s[r, 2])] if s[r, 2] >= 0 else 0)) != 0
                              for r in rows], dtype=bool)
            dropped += int((~exact).sum())
            rows = rows[exact]
            if len(rows) < 2:
                continue
        oo = o[rows]
        if k == 5:
            tri = s[rows, 2] >= 0
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
                t1 = s[rows[first[ia[t]]]].tolist()
                t2 = s[rows[first[ib[t]]]].tolist()
                harmful.append(dict(o=hex(key), bad=bad, acc=acc, triples=[t1, t2]))
    return dict(bucket=bucket, triples=int(len(h)), candidates=cand, dropped_undetected=dropped, harmful=harmful)


def parse_buckets(spec, nb):
    if not spec:
        return list(range(nb))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == "__main__":
    path = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    bits = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    todo = parse_buckets(sys.argv[5] if len(sys.argv) > 5 else "", 2 ** bits)
    stem = os.path.splitext(os.path.basename(path))[0]
    log = os.path.join(R, f"{stem}_k{k}_buckets.jsonl")
    done = {}
    if os.path.exists(log):
        for line in open(log):
            r = json.loads(line)
            done[r["bucket"]] = r
    t0 = time.time()
    if 0 in todo and "zero_sets" not in done:
        z = zero_sets(path, k)
        z["bucket"] = "zero_sets"
        print("zero sets:", json.dumps({a: b for a, b in z.items() if a != "harmful"}), flush=True)
        with open(log, "a") as f:
            f.write(json.dumps(z) + "\n")
        done["zero_sets"] = z
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(work, [(path, b, bits, k) for b in todo if b not in done]):
            done[r["bucket"]] = r
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
            print("bucket", r["bucket"], "candidates", r["candidates"], "harmful", len(r["harmful"]),
                  "s", round(time.time() - t0), flush=True)
    parts = [done[b] for b in range(2 ** bits) if b in done]
    if len(parts) < 2 ** bits or "zero_sets" not in done:
        print(f"{len(parts)} of {2 ** bits} buckets done; final json not written", flush=True)
        sys.exit(0)
    z = done["zero_sets"]
    harmful = [x for p in parts for x in p["harmful"]] + z["harmful"]
    out = dict(circuit=os.path.basename(path), k=k, triples=sum(p["triples"] for p in parts),
               candidates_evaluated=sum(p["candidates"] for p in parts),
               undetected_classes=z["undetected_classes"], zero_span_rank=z["span_rank"],
               zero_span_reached=z["reached"], harmful_candidates=len(harmful) + z["harmful_count"] - len(z["harmful"]),
               examples=harmful[:20], seconds=round(time.time() - t0))
    print(json.dumps({a: b for a, b in out.items() if a != "examples"}), flush=True)
    json.dump(out, open(os.path.join(R, f"{stem}_k{k}.json"), "w"), indent=1)
