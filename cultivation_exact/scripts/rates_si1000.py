"""Logical error per accepted state and acceptance under SI1000 noise, real T and S proxy, by TDEM sampling,
for any TDEM-ready circuit (our builders, no sublattice pattern). Shots scale as BASE * (1e-3 / p)^2.
Usage: python rates_si1000.py TAG PATH P_LIST BASE [T|S|TS]   -> results/rates_si1000.json (appends)"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from mc import sample_parallel
from noise import si1000

if __name__ == "__main__":
    tag, path = sys.argv[1], sys.argv[2]
    plist = [float(x) for x in sys.argv[3].split(",")]
    base = float(sys.argv[4])
    magics = sys.argv[5] if len(sys.argv) > 5 else "TS"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "rates_si1000.json")
    c0 = stim.Circuit.from_file(path)
    for p in plist:
        c = si1000(c0, p)
        shots = int(min(base * (1e-3 / p) ** 2, 4e10))
        for magic in magics:
            t0 = time.time()
            r = sample_parallel(c, magic, shots, workers=14, chunks=56, pattern=False)
            r.pop("raw", None)
            r.update(tag=tag, circuit=os.path.basename(path), qubits=c0.num_qubits, ticks=c0.num_ticks, p=p,
                     magic=magic, model="SI1000", seconds=round(time.time() - t0))
            out = json.load(open(out_path)) if os.path.exists(out_path) else []
            out.append(r)
            json.dump(out, open(out_path, "w"), indent=1)
            print(f"{tag} p={p:g} {magic}: P_acc={r['P_acc']:.4g} eps={r['eps']:.3e} +- {r['se_eps']:.1e} "
                  f"({shots:.1e} shots, {r['seconds']}s)", flush=True)
