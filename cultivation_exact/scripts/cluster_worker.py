"""Cluster worker: one shard of an exact malignant-configuration search.

Usage: python cluster_worker.py MODE CIRCUIT SHARD NSHARDS OUTDIR [complement]
  MODE in {k3, k4, k5}; CIRCUIT a .stim file whose magic gates are written explicitly (pattern=False).
Writes OUTDIR/<circuit stem>_<MODE>_<SHARD>of<NSHARDS>.json (skipped if it already exists, so re-submitting an
array resumes)."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
import cluster_search as cs

mode, path, shard, nshards, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
stem = os.path.splitext(os.path.basename(path))[0]
out = os.path.join(outdir, f"{stem}_{mode}_{shard}of{nshards}.json")
if os.path.exists(out):
    print("exists:", out)
    sys.exit(0)
os.makedirs(outdir, exist_ok=True)
t0 = time.time()
kw = {"complement": True} if (len(sys.argv) > 6 and sys.argv[6] == "complement") else {"pattern": False}
m = TDEM(stim.Circuit.from_file(path), magic="T", **kw)
C = cs.Classes(m)
res = getattr(cs, mode)(C, shard, nshards)
res.update(circuit=stem, shard=shard, nshards=nshards, classes=C.n, k_affine=m.kaff, t=m.t,
           seconds=round(time.time() - t0))
tmp = out + ".tmp"
json.dump(res, open(tmp, "w"))
os.replace(tmp, out)
print(json.dumps({k: v for k, v in res.items() if k != "examples"}))
