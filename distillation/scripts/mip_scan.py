"""Scan (n, s) for a target with tri_mip; print first feasible n and infeasibility certificates below it."""
import sys, json
from tri_mip import solve, overlaps_ok, distance
target, d = sys.argv[1], int(sys.argv[2]); ns = range(int(sys.argv[3]), int(sys.argv[4]) + 1)
svals = [int(x) for x in sys.argv[5].split(",")]; tl = float(sys.argv[6]) if len(sys.argv) > 6 else 600
for n in ns:
    for s in svals:
        st, M, sec, k = solve(target, n, s, d, tl)
        rec = {"target": target, "d": d, "n": n, "s": s, "status": st, "sec": round(sec, 1)}
        if M is not None:
            rec["dZ"] = distance(M, k, d); rec["ok"] = overlaps_ok(M, k, target)
            rec["matrix"] = ["".join(map(str, r)) for r in M]
        print(json.dumps(rec), flush=True)
