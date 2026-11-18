"""Exact fault distance of the RP^2 cultivation circuits of Chen et al. (PRX Quantum 7, 010315; circuits from
github.com/Zihan-Chen-PhMA/Cultiv_T_RP2, MIT licence, in ../external/rp2_circuits/) with real T gates or with the
S proxy. Their final MPP holds the RP^2 stabilizers (a redundant list) and the logical Y product of the proxy, whose
Z part is the logical Z (zl_from_obs). Their detectors include fixed parities of one and inverted records.
Usage: python rp2_fd.py FILE T|S kmax   (FILE: rp_3_T_cult.stim or rp_3_rp_5_T_cult.stim)"""
import sys, time
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from tdem import TDEM
from fd import fault_distance

name, magic, kmax = sys.argv[1], sys.argv[2], int(sys.argv[3])
path = __file__.rsplit("scripts", 1)[0] + "external/rp2_circuits/" + name
t0 = time.time()
m = TDEM(stim.Circuit.from_file(path), magic=magic, pattern=False, zl_from_obs=True)
mech = m.mechanisms()
print(f"{name} magic={magic}: t={m.t}, output Bloch {[round(x, 4) for x in m.output_bloch()]}, {len(mech)} mechanisms, "
      f"{m.n_det} skeleton detectors, free records {m.free_records}", flush=True)
k, bad, mech = fault_distance(m, kmax, mech)
coef = sum(b[4] * b[3] for b in bad)
print(f"RESULT {name} magic={magic}: fault distance {k if k else '>' + str(kmax)}; {len(bad)} malignant; leading "
      f"coefficient {coef / 1e-3 ** k if k else 0:.4g} p^{k} (uniform p = 1e-3 file noise); {time.time() - t0:.1f}s",
      flush=True)
