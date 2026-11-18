"""Check FastExact (src/branch.py) against the exact TDEM contraction on random undetected-or-not classes.

Usage: python fastexact_check.py PATH [default|complement|nopattern] [N]"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import stim
from tdem import TDEM
from branch import FastExact

path = sys.argv[1]
opt = sys.argv[2] if len(sys.argv) > 2 else "default"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
kw = {"default": {}, "complement": {"complement": True}, "nopattern": {"pattern": False}}[opt]
m = TDEM(stim.Circuit.from_file(path), magic="T", **kw)
fe = FastExact(m)
print(f"{os.path.basename(path)}: k={m.kaff} t={m.t} accepted Z-pattern classes {len(fe.A)}", flush=True)
mech = [x for x in m.mechanisms() if x[2]]
rng = np.random.default_rng(1)
os_ = set()
for _ in range(N):
    w = rng.integers(1, 5)
    o = 0
    for i in rng.choice(len(mech), size=w, replace=False):
        o ^= mech[i][2]
    if o:
        os_.add(o)
worst = 0.0
t_ex = t_fe = 0.0
nharm = 0
for o in os_:
    t0 = time.time(); e = m.evaluate(o); t1 = time.time(); f = fe.evaluate(o); t2 = time.time()
    t_ex += t1 - t0; t_fe += t2 - t1
    worst = max(worst, abs(e[0] - f[0]), abs(e[1] - f[1]))
    nharm += e[1] > 1e-9
print(f"  {len(os_)} classes ({nharm} harmful): max |difference| in (acc, bad) = {worst:.2e}; "
      f"exact {t_ex:.2f}s, fast {t_fe:.2f}s, largest enumeration {fe.max_terms}", flush=True)
