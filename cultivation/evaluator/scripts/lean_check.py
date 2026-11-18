"""Check LeanTDEM + LeanExact against TDEM + FastExact class by class: every single mechanism and every pair of
mechanisms with cancelling detectors (up to a cap), on circuits small enough for the original evaluator.
Usage: python lean_check.py   (prints ALL OK)"""
import sys, os, itertools, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import stim
from tdem import TDEM
from branch import FastExact
from lean_exact import LeanTDEM, LeanExact

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, "..", "external", "gidney_circuits")
cases = [
    ("RP2 d=3", os.path.join(HERE, "..", "external", "rp2_circuits", "rp_3_T_cult.stim"), dict(pattern=False, zl_from_obs=True)),
    ("Gidney d=3", os.path.join(G, "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim"), {}),
    ("verified cat d=3", os.path.join(HERE, "..", "results", "vcat_d3_injdag_merged.stim"), {}),
    ("ring d=3", os.path.join(HERE, "..", "results", "nn_cat_d3_r1_v1_unf.stim"), dict(pattern=False)),
    ("Gidney d=5", os.path.join(G, "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim"), dict(complement=True)),
]
ok_all = True
for name, path, kw in cases:
    c = stim.Circuit.from_file(path)
    t0 = time.time()
    m = TDEM(c, magic="T", **kw)
    fe = FastExact(m)
    ml = LeanTDEM(c, magic="T", **kw)
    le = LeanExact(ml)
    mech = [x for x in m.mechanisms() if x[1] or x[2]]
    os_ = {x[2] for x in mech if x[1] == 0}
    byD = {}
    for x in mech:
        byD.setdefault(x[1], []).append(x[2])
    for D, lst in byD.items():
        for a, b in itertools.combinations(lst, 2):
            os_.add(a ^ b)
            if len(os_) > 20000:
                break
    os_.discard(0)
    worst = 0.0
    nh_full = nh_lean = 0
    for o in os_:
        a1, b1 = fe.evaluate(o)
        a2, b2 = le.evaluate(o)
        worst = max(worst, abs(a1 - a2), abs(b1 - b2))
        nh_full += b1 > 1e-9
        nh_lean += b2 > 1e-9
    same_out = np.allclose(m.u0, ml.u0) or np.allclose(abs(np.vdot(m.u0, ml.u0)), 1)
    ok = worst < 1e-9 and nh_full == nh_lean and same_out and m.kaff == ml.kaff
    ok_all &= ok
    print(f"{name}: k={ml.kaff}, A={list(ml.A)}, {len(os_)} classes, harmful {nh_full} / {nh_lean}, "
          f"max deviation {worst:.1e}, {'OK' if ok else 'MISMATCH'} ({time.time() - t0:.0f}s)", flush=True)
print("ALL OK" if ok_all else "FAILED")
