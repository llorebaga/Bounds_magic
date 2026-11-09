"""Smallest n not excluded by the level-k relaxation, for one T state and several d."""
import sys, time
from t_lp import solve
ks = [int(x) for x in sys.argv[1].split(",")]
ds = [int(x) for x in sys.argv[2].split(",")]
for d in ds:
    for k in ks:
        t0 = time.time()
        n = d
        while True:
            feas, lam = solve(n, d, k, n_random=400)
            if feas:
                break
            n += 1
        print(f"T d={d} level k={k}: first feasible n = {n}  (LB n >= {n}; analytic n >= {d})  [{time.time()-t0:.0f}s]", flush=True)
