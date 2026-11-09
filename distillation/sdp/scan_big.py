import sys, time
import numpy as np
from t_cut import solve
for d in [6, 7, 8]:
    t0 = time.time()
    n = 2 * d - 2
    while True:
        feas, lam, added = solve(n, d, (1, 2, 3), n_random=4000, rng=np.random.default_rng(1))
        print(f"  d={d} n={n}: {'feasible' if feas else 'INFEASIBLE'} ({added} cuts, {time.time()-t0:.0f}s)", flush=True)
        if feas:
            break
        n += 1
    print(f"T d={d} levels=(1,2,3): first n not excluded = {n}", flush=True)
