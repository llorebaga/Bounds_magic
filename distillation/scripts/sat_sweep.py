"""Run block-model instances (target D s nT nJ nC), each in its own subprocess with a hard timeout;
certify SAT designs exactly with verify_block.certify.
Usage: python sat_sweep.py out.jsonl timeout_s 'target D s nT nJ nC' ...   (or @file with one instance per line)"""
import sys, json, subprocess, time, os
from verify_block import certify

out, timeout = sys.argv[1], float(sys.argv[2])
lines = []
for a in sys.argv[3:]:
    if a.startswith("@"): lines += [l.strip() for l in open(a[1:]) if l.strip()]
    else: lines.append(a)
here = os.path.dirname(os.path.abspath(__file__))
with open(out, "a") as fo:
    for l in lines:
        p = l.split(); t0 = time.time()
        try:
            r = subprocess.run([sys.executable, os.path.join(here, "sat_block.py"), *p, "cadical195", str(timeout * 10)],
                               capture_output=True, text=True, timeout=timeout)
            rec = json.loads(r.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            rec = {"target": p[0], "D": int(p[1]), "s": int(p[2]), "nT": int(p[3]), "nJ": int(p[4]), "nC": int(p[5]),
                   "Tcost": int(p[3]) + 4 * int(p[4]) + 8 * int(p[5]), "status": "TIMEOUT", "sec": round(time.time() - t0, 1)}
        if rec.get("status") == "SAT":
            try: rec["certificate"] = certify(rec)
            except Exception as ex: rec["certificate"] = {"error": str(ex)}
        fo.write(json.dumps(rec) + "\n"); fo.flush()
        print(json.dumps({k: rec.get(k) for k in ("target", "D", "s", "nT", "nJ", "nC", "Tcost", "status", "sec")}
                         | ({"cert_ok": rec["certificate"].get("ok"), "harm_at_D": rec["certificate"].get("harmful_combos_at_cost_D")} if "certificate" in rec else {})), flush=True)
