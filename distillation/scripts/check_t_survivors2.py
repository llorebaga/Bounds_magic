"""Classify collapse survivors of the T-target kernel by the decoded state.
For each signed code: v = Pi_C phi_0 (applied generator by generator), then count Pauli operators P with
|<v|P|v>| = 1.  A code state encoding |T> (nu=1) has 2^(n-1) of them, an encoded stabilizer state 2^n.
Also reports the number of Pauli expectations equal to 1/sqrt2 (2^n for an encoded T state).
Usage: python check_t_survivors2.py logfile n [nproc]"""
import sys, re, json, numpy as np
from multiprocessing import Pool
from common import popc_arr, popcount, herm, phi

fname, n = sys.argv[1], int(sys.argv[2]); NP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
D = 1 << n; idx = np.arange(D)
Hm = (-1.0) ** popc_arr(idx[:, None] & idx[None, :]); XB = idx[:, None] ^ idx[None, :]
PHI0 = phi(0, n)

def apply_pauli(P, v):
    p, x, z = P
    ph = (-1.0) ** popc_arr(idx & z)
    out = np.zeros_like(v); out[idx ^ x] = ph * v
    return (1j ** p) * out

def classify(task):
    AX, BZ, cs, sg = task
    a = len(AX)
    gens = [(1 if not (sg >> i) & 1 else -1, herm(ax, c)) for i, (ax, c) in enumerate(zip(AX, cs))]
    gens += [(1 if not (sg >> (a + k)) & 1 else -1, herm(0, b)) for k, b in enumerate(BZ)]
    v = PHI0.astype(complex)
    for s, P in gens: v = (v + s * apply_pauli(P, v)) / 2
    nv = np.linalg.norm(v)
    if nv < 1e-9: return "zero"
    v = v / nv
    spec = np.abs((np.conj(v[XB]) * v[None, :]) @ Hm).ravel()
    c1 = int((np.abs(spec - 1) < 1e-6).sum()); cr = int((np.abs(spec - 2 ** -0.5) < 1e-6).sum())
    if c1 == D: return "stabilizer"
    if c1 == D // 2 and cr == D: return "T"
    return f"other:{c1}:{cr}"

def tasks():
    seen = set()
    for line in open(fname):
        m = re.match(r"SURVIVOR a=(\d+) AX=\[([^\]]*)\] BZ=\[([^\]]*)\] c=(\d+) signs=\[([^\]]*)\]", line.strip())
        if not m: continue
        a = int(m[1]); AX = tuple(int(x) for x in m[2].split(","))
        BZ = tuple(int(x) for x in m[3].split(",")) if m[3].strip() else ()
        cv = int(m[4]); cs = tuple((cv >> (i * n)) & (D - 1) for i in range(a))
        for sg in [int(x) for x in m[5].split(",") if x.strip()]:
            key = (AX, BZ, cs, sg)
            if key not in seen: seen.add(key); yield key

if __name__ == "__main__":
    counts = {}; examples = []
    with Pool(NP) as pool:
        for t, r in zip(tasks(), pool.imap(classify, tasks(), chunksize=256)):
            counts[r] = counts.get(r, 0) + 1
            if r == "T" and len(examples) < 50: examples.append({"AX": t[0], "BZ": t[1], "c": t[2], "signs": t[3]})
    print(json.dumps(counts))
    json.dump(examples, open(fname + ".T_equiv.json", "w"))
