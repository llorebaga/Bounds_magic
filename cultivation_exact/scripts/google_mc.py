"""Logical error and retention of Google's experimental cultivation circuits with real T gates and with the S proxy,
under SI1000 noise, by TDEM sampling (each sampled fault class weighted exactly).
Usage: python google_mc.py fig2|fig3 CUT|- P_LIST SHOTS_AT_2.3e-3   -> results/google_mc.json (appends)"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mc import sample_parallel
import google_tdem as g

if __name__ == "__main__":
    fig = sys.argv[1]
    cut = None if sys.argv[2] == "-" else int(sys.argv[2])
    plist = [float(x) for x in sys.argv[3].split(",")]
    base = float(sys.argv[4])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "google_mc.json")
    out = json.load(open(path)) if os.path.exists(path) else []
    for p in plist:
        c, info = g.make(fig, p, -1, cut)
        shots = int(min(base * (2.3e-3 / p) ** 2, 3e10))
        for magic in ("T", "S"):
            t0 = time.time()
            r = sample_parallel(c, magic, shots, workers=14, chunks=56, pattern=False)
            r.pop("raw", None)
            r.update(fig=fig, cut=info["cut"], p=p, magic=magic, model="SI1000", seconds=round(time.time() - t0))
            out.append(r)
            print(f"{fig} cut={info['cut']} p={p:g} {magic}: P_acc={r['P_acc']:.4f} eps={r['eps']:.3e} "
                  f"+- {r['se_eps']:.1e} ({shots:.1e} shots, {r['seconds']}s)", flush=True)
            json.dump(out, open(path, "w"), indent=1)
