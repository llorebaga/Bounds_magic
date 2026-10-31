"""Weight-3 malignant search of a .stim file, one shard of the outer loop.  Usage: fd3_shard.py path shard nshards"""
import sys, time, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast
path, shard, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = TDEM(stim.Circuit.from_file(path), magic="T", pattern=False)
t0 = time.time()
k, f = fault_distance_fast(m, 3, m.mechanisms(), verbose=False, shard=shard, nshards=ns, kmin=3)
print(json.dumps(dict(shard=shard, malignant=len(f), coef=sum(x[4] * x[3] for x in f) / 1e-9, seconds=round(time.time() - t0))), flush=True)
