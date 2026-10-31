# cultivation_exact

Exact real-T evaluation of cultivation circuits for `paper/cultivation.tex` ("The limits of magic-state
cultivation"): Theorem 2 (Gauss sums, precise form Theorem 11 and Proposition 12), Lemma 4, Propositions 5 and 6,
Tables I and SIV to SIX, and Figures 2 to 4. The fault map of Theorem 1 is computed as a stim detector error
model of the skeleton (the T-aware detector error model, TDEM). The collapse condition is decided by Z4 Gauss sums,
which cost O(t^3) per fault class. This is independent of the state-vector implementation in `../scripts/cultiv.py`,
used for Propositions 5 (d = 3) and 16. The two agree on the published d = 3 circuit: 96 harmful pairs with idle
noise, leading coefficient 32/75.

All commands are run from `scripts/` (paths such as `../results/` are relative to this directory). Propositions,
tables and figures in the table below refer to the original draft. In `paper/cultivation.tex`, the folds, the
published circuits and Google's experiment are Proposition 5 (with details in Proposition 19), the cats and grid
checks Proposition 6, flags Proposition 17, the old Table I is Table SVIII, the old Table II (cost) is Table SIX,
the old Tables SII, SIII and SIV are Tables SV, SVI and SVII, the fault-distance figure is Fig. 3 and the grid
figure Fig. 4. Here d is the code distance (delta in the paper).

## Layout

```
src/       the TDEM and all library code (scripts add ../src to the path)
scripts/   experiments (run from inside scripts/: paths are relative, e.g. ../results/)
results/   circuits (.stim), logs, JSON outputs and certificates referenced in the paper
external/  the Gidney-Shutty-Jones cultivation circuits used as input
```

Library code in `src/`:
- `tdem.py`: the TDEM. Use `complement=True` for the d=5 circuit of Gidney et al., which gives the correct global
  T/T-dagger parity (Sec. S4).
- `gf2.py`: linear algebra over GF(2).
- `faults.py`: physical single-fault table (FlipSimulator): detectors, observables, record flips, X/Z frames.
- `fd.py`, `fd_fast.py`, `fd_list.py`: fault distance and leading-order terms (meet in the middle, k <= 4).
- `cluster_search.py`: sharded exact searches for malignant sets of weight 3, 4 and 5.
- `mc.py`: Monte Carlo of logical error and acceptance, weighting each sampled fault class exactly.
- `mitm_rate.py`: rare-event estimator (exact sum over the last two faults).
- `noise.py`: uniform and SI1000 noise models.
- `flags.py`, `flags2.py`: flag design (Z, X and ZZ flags; Gurobi cover; exact re-certification).
- `foldtree.py`: GHZ fold-tree redesign and `XScreen`, the closed-form coherent-completion test.
- `vcat.py`, `colorcode.py`, `cult_circuits.py`: verified-cat circuits on the d = 3, 5, 7 color codes.
- `statevec.py`: brute-force state vector with real T, used only for validation.
- `nn_cat.py`: nearest-neighbour verified-cat checks (ring at d=3, U-shaped ladder with rungs and diagonals at d=5, 7).
- `noise.py` SI1000 also applies to our own circuits; `cult_circuits.Builder(model="si1000")` builds them with it.

## Requirements

- Python >= 3.10 with `numpy`, `stim` 1.16
- `gurobipy` (Gurobi 12; an academic license suffices) for the flag designs

## Claims and how to reproduce them

All commands are run from `scripts/`.

| Claim | Command | Output |
|---|---|---|
| Validation against a state vector with real T (2,444 single faults, 400 pairs, 96 malignant pairs; Sec. S3) | `python validate_sv.py`, `python validate_harmful.py` | `results/validate_sv_d3.json` |
| Proxy cross-check against direct stim sampling | `python mc_check_proxy.py` | stdout |
| d=3 circuit of Gidney et al.: real-T fault distance 2, real vs proxy rates (Table SII) | `python fd_gidney.py`, `python mc_gidney_d3.py`, `python mc_si1000_d3.py` | `results/mc_*_d3.json` |
| d=5 full protocol: fault distance 3, leading term 1.52 p^3 | `python fd_d5_best.py`, `python d5_leading.py` | `results/d5_leading.json` |
| d=5 rates vs SOFT (Table SII) | `python mitm_d5.py` | `results/mitm_d5_unflagged*.json` |
| Single-qubit flags cannot exceed fault distance 3; two ZZ flags give 4 (Proposition 12) | `python design2.py`, `python rates_designs.py` | `results/design2_*`, `results/flagged_*.stim`, `results/rates_*_hi.json` |
| Verified cat at d=5: fault distance exactly 5, c5 = 3452; unverified: 3 (Proposition 13, Table I) | `python fd_file.py ../results/vcat_d5_r1.stim 4`, cluster run C6 | `results/fd_vcat_d5_*.log`, `results/cluster/out/C6/` |
| Verified cat at d=3 (T^dag injected, parity matching the check): fault distance 3 (Proposition 13) | `python vcat_d3_fix.py` | `results/vcat_d3_injdag_merged.stim`, `results/vcat_d3_fix.json` |
| Gauss-sum evaluation (Theorem 4, Proposition 5) agrees with the contraction to 5e-16 | `python fastexact_check.py ../results/vcat_d5_r1.stim nopattern` (and the other circuits) | stdout |
| Real-T fault distances with the Gauss-sum evaluator, d=3/5/7 (Table I, Propositions 11 and 13) | `python fd_gauss.py PATH KMAX OPTION` | `results/fd_gauss.log`, `results/fd_gauss_k4.log` |
| Unverified cat d=7: fault distance 4, exact witnesses for all 38 harmful errors | `python witness_k4.py ../results/ucat_d7_r1_injdag.stim nopattern` | `results/ucat_d7_r1_injdag_witness_k4.json` |
| GHZ-fold double checks (`src/fold.py`): real-T fault distance 2, 3, 4 at d=3, 5, 7 for every tree tested, proxy 3, 5, >=5 (Proposition 11) | `python fold_fd.py 3 3 c,1,2 proxy`, `python fold_fd.py 5 4 c,bc,b1 proxy`, `python fold_fd.py 7 4 c,bc,b1,b2,b3,1`, `python witness_k4.py ../results/fold_n37_<tree>.stim nopattern`, `python proxy_fd.py 4 ...` | `results/fold_n*.json`, `results/fold_n37_*_witness.log`, `results/proxy_fd.log` |
| d=7 leading terms: every malignant 4-set, exact (Table I, Sec. S4) | `python leading_k4.py ../results/fold_n37_<tree>.stim nopattern` (also `ucat_d7_r1_injdag.stim`) | `results/*_c4.json` |
| Acceptance at d=5 and 7 (Table SIII) | `python rates_gauss.py TAG PATH nopattern 1e-3,5e-4 9 20000 5e6` | `results/rates_gauss_<TAG>.json`, `results/rates/` |
| Fig. 3 | `python fig_fd.py` | `../paper/fig_fd.pdf` |
| X patterns inside minimum-weight logicals (Table SIV) | `python logical_cover.py` | `results/logical_cover.txt` |
| Nearest-neighbour checks on a square grid (`src/nn_cat.py`): ring at d=3 (17 qubits) real-T FD 3; ladder at d=5 (46 qubits) FD 5 with one and two rounds (FD 4 without diagonals); ladder at d=7 (91 qubits) FD >= 5, >= 6 with the un-grown cat (Proposition 14, Tables I and II) | `python nn_cat_fd.py d3`, `python fd_gauss.py ../results/nn_cat_d5_r1_v1.stim 4 nopattern` (also `nn_cat_d5_r2_v1.stim`, `nn_cat_d7_r1_v1_dag.stim`) | `results/nn_cat_*.stim`, `results/nn_cat_*_k4.log` |
| Exact weight-5 terms: NN ladder 43 p^5 (SI1000) and 121 p^5 (uniform); verified cat under SI1000 | `python leading_k5.py ../results/nn_cat_d5_r1_v1.stim si1000` (and `file`) | `results/*_c5.json` |
| Google's experimental circuits (Rosenfeld et al., arXiv:2512.13908) with real T: FD 2 vs proxy 3 after 1-3 checks, also with ideal injection; malignant pairs (Proposition 15) | `python google_crumble_to_stim.py` (decodes the paper's Crumble links), `python google_tdem.py fig3 1e-3 -1 162`, `python google_fd.py PATH 3`, `python google_malignant.py PATH` | `external/google_circuits/`, `results/google_*.stim` |
| Google's circuit under SI1000, real T vs proxy (Fig. 2c) | `python google_mc.py fig3 162 3e-3,2.3e-3,1.5e-3,1e-3 2e9` | `results/google_mc.json`, `results/google_mc.log` |
| Resources under SI1000 at p = 1e-3 (Table II) | `python resources.py`, `python acceptance.py si1000 1e-3 1e6 PATHS` | `results/resources_si1000.json` |
| Fig. 4 | `python fig_nn.py` | `../paper/fig_nn.pdf` |
| Un-grown NN checks (Prop. 14): ring 3.6 p^3 uniform, 0.056 p^3 + 30 p^4 SI1000; ladder 12 p^5 uniform, 0.012 p^5 SI1000; d=7 ladder FD >= 6 (no malignant 5-set, `leading_k5.py ../results/nn_cat_d7_r1_v1_unf.stim file`, 15 min on 14 cores) | `python fd_gauss.py ../results/nn_cat_d5_r1_v1_unf.stim 4 nopattern`, `python leading_k5.py ../results/nn_cat_d5_r1_v1_unf.stim si1000`, `python leading_k4.py ../results/nn_cat_d3_r1_v1_unf_si1000.stim nopattern` | `results/nn_cat_*_unf*` |
| Complete NN protocol at d=3 (relayed injection, Shor-style stabilizer round, three checks): FD 3, 7.8e3 p^3 SI1000; variants and the measurement-prepared protocol | `python nn_full.py d3 3 1 0 1 3 shor routed` (args: ROUNDS ZROUNDS ZBETWEEN VER KMAX KIND INJECT [unfold]), `python nn_sched.py d3 XCZXCC unfold` | `results/nn_full_*.stim`, `results/nn_meas_*.stim` |
| Where the malignant configurations sit | `python malignant_profile.py PATH 3 [si1000]` | stdout |

The weight-5 enumerations run locally with `scripts/leading_k5.py` (all cores); `scripts/fd_k6.py` is the weight-6 search (triples against triples) with a weight-5 control mode.

## Inputs

The `inject+cultivate` circuits in `external/gidney_circuits/` are from Gidney, Shutty and Jones, *Magic state
cultivation* (Zenodo, doi:10.5281/zenodo.13777072). They are only parsed, never executed.
