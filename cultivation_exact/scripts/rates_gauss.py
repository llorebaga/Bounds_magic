"""Logical error per accepted output and acceptance with the Gauss-sum evaluator (src/rates_fast.py).

Usage: python rates_gauss.py TAG PATH OPTION P_LIST KMAX SAMPLES SHOTS [MODEL]
  OPTION default|complement|nopattern; P_LIST comma list of p; MODEL uniform (default) or si1000.
P_bad: meet-in-the-middle completion up to KMAX faults (exact for k <= 3, SAMPLES draws per stratum k >= 4),
on the error model of the circuit with noise model MODEL at p = 1e-3, rescaled to each p.
P_acc: SHOTS samples of the error model at each p. eps = P_bad / P_acc.
Appends to results/rates_gauss_<TAG>.json."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from branch import FastExact
from rates_fast import FastMITM, acceptance
from noise import uniform, si1000

tag, path, opt = sys.argv[1], sys.argv[2], sys.argv[3]
ps = [float(x) for x in sys.argv[4].split(",")]
kmax, samples, shots = int(sys.argv[5]), int(sys.argv[6]), int(float(sys.argv[7]))
model = sys.argv[8] if len(sys.argv) > 8 else "uniform"
noise = {"uniform": uniform, "si1000": si1000}[model]
kw = {"default": {}, "complement": {"complement": True}, "nopattern": {"pattern": False}}[opt]
c0 = stim.Circuit.from_file(path)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"rates_gauss_{tag}.json")
rows = json.load(open(out_path)) if os.path.exists(out_path) else []

t0 = time.time()
m = TDEM(noise(c0, 1e-3), magic="T", **kw)
fe = FastExact(m)
est = FastMITM(fe)
print(f"{tag}: t={m.t} k={m.kaff} classes={est.n} accepted Z classes={len(fe.A)} ({time.time()-t0:.0f}s)", flush=True)
for p in ps:
    t1 = time.time()
    pb, se, strata = est.P_bad(p / 1e-3, kmax, samples, verbose=True)
    mp = TDEM(noise(c0, p), magic="T", **kw)
    fep = FastExact(mp)
    pa, sea = acceptance(fep, mp.dem(), shots)
    row = dict(tag=tag, circuit=os.path.basename(path), model=model, p=p, P_acc=pa, se_P_acc=sea, P_bad=pb,
               se_P_bad=se, eps=pb / pa, se_eps=se / pa, strata=[(k, a, s) for k, a, s in strata], kmax=kmax,
               samples=samples, shots=shots, seconds=round(time.time() - t1))
    rows.append(row)
    json.dump(rows, open(out_path, "w"), indent=1)
    print(f"  {model} p={p:g}: acceptance {pa:.4f} +- {sea:.4f}, eps {pb/pa:.3e} +- {se/pa:.1e} "
          f"({time.time()-t1:.0f}s)", flush=True)
