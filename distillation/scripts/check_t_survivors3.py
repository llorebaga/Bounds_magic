"""Fast classification of collapse survivors for a 1-qubit target (r=1) by the logical Bloch vector.
For each signed [[n,1]] code: find logical Paulis L1, L2 (commute with all stabilizers, independent mod S),
compute v = Pi_C phi_0 and the expectations of L1, L2, L1L2.  Encoded stabilizer state <=> some |<L>| = 1;
encoded state Clifford-equivalent to |T> <=> {|<L>|} = {1/sqrt2, 1/sqrt2, 0}.
Usage: python check_t_survivors3.py logfile n"""
import sys, re, json, numpy as np
from common import popc_arr, popcount, herm, phi, nullspace, rref, in_span_rref, pmul

fname, n = sys.argv[1], int(sys.argv[2]); D = 1 << n; idx = np.arange(D); PHI0 = phi(0, n).astype(complex)

def apply_pauli(P, v):
    p, x, z = P
    out = np.zeros_like(v); out[idx ^ x] = ((-1.0) ** popc_arr(idx & z)) * v
    return (1j ** p) * out

def logicals(gens_xz):
    cons = [(gz) | (gx << n) for gx, gz in gens_xz]   # symplectic product with (x,z) packed as x | z<<n
    ns = nullspace(cons, 2 * n)
    Srows = rref([gx | (gz << n) for gx, gz in gens_xz])
    L = []; span = list(Srows)
    for w in ns:
        if not in_span_rref(w, rref(span)):
            L.append(w); span.append(w)
        if len(L) == 2: break
    return [(w & (D - 1), w >> n) for w in L]

def classify(AX, BZ, cs, signs, cache={}):
    a = len(AX)
    key = (AX, BZ, cs)
    gxz = [(ax, c) for ax, c in zip(AX, cs)] + [(0, b) for b in BZ]
    if key not in cache: cache.clear(); cache[key] = logicals(gxz)
    (x1, z1), (x2, z2) = cache[key]
    L1 = herm(x1, z1); L2 = herm(x2, z2); L3 = pmul(L1, L2); L3 = herm(L3[1], L3[2])
    out = []
    for sg in signs:
        v = PHI0.copy()
        for i, (gx, gz) in enumerate(gxz):
            s = -1 if (sg >> i) & 1 else 1
            v = (v + s * apply_pauli(herm(gx, gz), v)) / 2
        nv = np.vdot(v, v).real
        if nv < 1e-12: out.append("zero"); continue
        e = sorted(abs(np.vdot(v, apply_pauli(L, v)).real) / nv for L in (L1, L2, L3))
        if abs(e[2] - 1) < 1e-7: out.append("stabilizer")
        elif abs(e[0]) < 1e-7 and abs(e[1] - 2 ** -0.5) < 1e-7 and abs(e[2] - 2 ** -0.5) < 1e-7: out.append("T")
        else: out.append("other:" + ",".join(f"{x:.4f}" for x in e))
    return out

if __name__ == "__main__":
    counts = {}; examples = []; texamples = []
    # finditer over the whole log: threads occasionally print two records on one line
    for m in re.finditer(r"SURVIVOR a=(\d+) AX=\[([^\]]*)\] BZ=\[([^\]]*)\] c=(\d+) signs=\[([^\]]*)\]", open(fname).read()):
        a = int(m[1]); AX = tuple(int(x) for x in m[2].split(","))
        BZ = tuple(int(x) for x in m[3].split(",")) if m[3].strip() else ()
        cv = int(m[4]); cs = tuple((cv >> (i * n)) & (D - 1) for i in range(a))
        signs = [int(x) for x in m[5].split(",") if x.strip()]
        for sg, r in zip(signs, classify(AX, BZ, cs, signs)):
            counts[r] = counts.get(r, 0) + 1
            if r == "T": texamples.append({"AX": AX, "BZ": BZ, "c": cs, "signs": sg})
            if r != "stabilizer" and len(examples) < 50: examples.append({"AX": AX, "BZ": BZ, "c": cs, "signs": sg, "class": r})
    print(json.dumps(counts))
    json.dump(examples, open(fname + ".nonstab.json", "w"), indent=1)
    if texamples: json.dump(texamples, open(fname + ".T.json", "w"), indent=1)
