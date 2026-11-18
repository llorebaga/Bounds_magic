# The exact evaluator

Exact evaluation of cultivation circuits with real T gates, for "The limits of magic-state cultivation". The fault
map of Theorem 1 is computed as a stim detector error model of the skeleton, the T-aware detector error model (TDEM).
For circuits with deterministic checks (Definition 17) the collapse condition is decided by the Gauss sums of
Theorem 19, at a cost of O(t^3) per fault class (Proposition 20). Appendix C 1 describes the method. The
state-vector implementation in `../checks/cultiv.py` is independent of this one, and the two agree on the published
distance-three circuit.

All commands run from `scripts/`, and paths such as `../results/` are relative to this directory. Here d is the code
distance, written delta in the paper.

## Layout

```
src/        library code (scripts add ../src to the path)
scripts/    the computations behind the paper
results/    circuits (.stim), JSON outputs, logs/, rp2/ and didle/ (the circuits and rates of Table IV)
external/   published circuits used as input
```

Library code in `src/`:
- `tdem.py`: the TDEM. `complement=True` gives the correct global T/T-dagger parity for the d=5 circuit of Gidney et al.
- `branch.py`: the Gauss-sum evaluator (`FastExact`); `lean_exact.py`: an in-place version for large skeletons.
- `gf2.py`: linear algebra over GF(2). `faults.py`: the single-fault table (detectors, observables, X and Z frames).
- `fd.py`, `fd_fast.py`, `fd_list.py`, `cluster_search.py`: fault distance and malignant sets by meet in the middle.
- `mc.py`, `mitm_rate.py`, `stratified.py`: logical error and acceptance (sampling with exact weights, rare-event estimator).
- `noise.py`: uniform and SI1000 noise. `idle.py`: idle noise on the data of checks from an ideal encoder.
- `colorcode.py`, `cult_circuits.py`, `fold.py`, `foldtree.py`, `vcat.py`, `nn_cat.py`: the color codes and the checks
  of the paper (GHZ folds, verified and unverified cats, nearest-neighbor ring and ladder).
- `statevec.py`: a brute-force state vector with real T gates, for validation only.

## Validation (Appendix C 1)

| Claim | Command | Output |
|---|---|---|
| The evaluator against a state vector on the published d=3 circuit: all 2,444 single faults, 400 random pairs, the same 96 malignant pairs | `python validate_sv.py`, `python validate_harmful.py` | `results/validate_sv_d3.json`, `results/validate_harmful_d3.json` |
| Gauss sums against the direct sum (to 5e-16) | `python fastexact_check.py ../results/vcat_d5_r1.stim nopattern` | stdout |
| The in-place evaluator against the standard one, class by class | `python lean_check.py` | stdout |
| Rare-event estimator against sampling on the d=3 circuit: 6.17e-7 against 6.14(23)e-7 at p = 1e-3 | `python mitm_d3_check.py` | `results/mitm_d3_check.json` |
| The TDEM sampler on the S proxy against direct stim sampling | `python mc_check_proxy.py` | stdout |

## Published circuits (Proposition 10, Proposition 36, Table II, Appendix F)

| Claim | Command | Output |
|---|---|---|
| d=3 circuit of Gidney et al.: real-T fault distance 2, proxy 3; real and proxy rates of Table II | `python fd_gidney.py 3 T 2`, `python fd_gidney.py 3 S 3`, `python mc_gidney_d3.py`, `python mc_si1000_d3.py` | `results/mc_gidney_d3.json`, `results/mc_si1000_d3.json` |
| d=5 circuit: real-T fault distance 3 in every growth branch (Proposition 35), leading term 1.5 p^3 | `python fd_frames.py 5 T 3`, `python fd_d5_best.py T 3`, `python d5_leading.py` | `results/d5_leading.json` |
| d=5 circuit, real T in Table II | `python mitm_d5.py`, `python mc_d5.py uniform 2e-3 3e8` | `results/mitm_d5_unflagged*.json`, `results/mc_d5_uniform_0.002.json` |
| d=5 circuit, S proxy: fault distance 4, 5.3 p^4, and the sampled rates of Table II | `python proxy_ilp.py PATH.stim 5 cap=6`, `python proxy_mc_d5.py uniform 1e-3 1.5e12 120` (and the other rows) | `results/proxy_mc_d5_*.json` |
| RP^2 circuits of Chen et al., d=3: real-T fault distance 2 in all four first-round branches (12 malignant pairs, 0.29 p^2), proxy 3 | `python rp2_fd.py rp_3_T_cult.stim T 2`, `python rp2_fd.py rp_3_T_cult.stim S 3`, `python proxy_fd.py 4 ...` | stdout, `results/logs/proxy_fd_rp2.log` |
| RP^2 d=5: real-T fault distance 3 in all 256 branches (4 malignant triples, 0.77 p^3), proxy 4 | `python rp2_fd_lean.py rp_3_rp_5_T_cult.stim 3`, `python rp2_branches_job.py TASK NTASKS`, `python rp2_d5_malignant.py 6 3` | `results/rp2/` |
| The circuit run on hardware (Rosenfeld et al.): real-T fault distance 2 after one, two or three checks, proxy 3; malignant pairs; rates under SI1000 | `python google_crumble_to_stim.py PAPER.html ../external/google_circuits` (from the HTML of arXiv:2512.13908), `python google_tdem.py fig3 1e-3 -1 162`, `python google_fd.py PATH 3`, `python google_malignant.py PATH`, `python google_mc.py fig3 162 3e-3,2.3e-3,1.5e-3,1e-3 2e9` | `results/google_mc.json`, `results/logs/google_mc.log` |

## GHZ folds (Proposition 10, Proposition 34, Appendix F 1)

| Claim | Command | Output |
|---|---|---|
| GHZ-fold checks between flagged stabilizer rounds: real-T fault distance 2, 3, 4 at d=3, 5, 7 for every tree, proxy 3, 5, 7 | `python fold_fd.py 3 3 c,1,2 proxy`, `python fold_fd.py 5 4 c,bc,b1 proxy`, `python fold_fd.py 7 4 c,bc,b1,b2,b3,1`, `python witness_k4.py ../results/fold_n37_<tree>.stim nopattern`, `python proxy_k56.py ../results/<stem>.stim 24 9 5` (and 6) | `results/fold_n*.json`, `results/*_witness_k4.json`, `results/*_proxy_k{5,6}.json` |
| Proposition 34: every fold on [[7,1,3]] has fault distance at most 2; 200 random trees have exactly 2 | `python fold_d3_all.py 200 1` | `results/fold_d3_all.json` |
| Bounds from the fold tree alone at d=9, 11, 13 | `python hook_bound.py 7 c,bc,b1,b2,b3,1`, `python hook_bound.py 9 cat,c,bc,b1,b2,b3,1,2 pack2` (also 11 and 13) | `results/logs/hook_bound_*.log` |
| Folds without stabilizer rounds at d=9: fault distance 5 and 4, equal to their bounds | `python fold_nostab.py 9 b1,b2 run` | `results/fold_n61_*_nostab.stim`, `results/logs/late_ilp_fold_d9.log` |

## Checks on an encoded state (Proposition 12, Corollary 37, Appendices E and G)

| Claim | Command | Output |
|---|---|---|
| Table I: X patterns inside minimum-weight logical supports | `python logical_cover.py` | `results/logical_cover.txt` |
| Real-T fault distances of the checks of Table III with the Gauss-sum evaluator | `python fd_gauss.py PATH KMAX OPTION`, `python fd_file.py ../results/vcat_d5_r1.stim 4` | `results/logs/fd_gauss*.log`, `results/logs/fd_vcat_d5_*.log` |
| Verified cats: d=3 with the T-dagger injection that matches the check; d=5 fault distance 5; d=7 no malignant set of five faults and none of the 2.2e9 candidates of six | `python vcat_d3_fix.py`, `python leading_k5.py ../results/vcat_d7_r1_injdag.stim file`, `python fd_k6.py ../results/vcat_d7_r1_injdag.stim 46 9 6` | `results/vcat_d3_fix.json`, `results/vcat_d7_r1_injdag_file_c5.json`, `results/vcat_d7_r1_injdag_k6.json` |
| Unverified cat at d=7: fault distance 4, witnesses for all 38 harmful errors | `python witness_k4.py ../results/ucat_d7_r1_injdag.stim nopattern` | `results/ucat_d7_r1_injdag_witness_k4.json` |
| Nearest-neighbor checks: ring at d=3 and ladder at d=5 keep the full distance (4 without diagonals); at d=7 the ladder with reversed growth has no harmful set of six faults, with direct readout it has fault distance 6 | `python nn_cat_fd.py d3`, `python fd_gauss.py ../results/nn_cat_d5_r1_v1.stim 4 nopattern`, `python fd_k6.py ../results/nn_cat_d7_r1_v1_unf.stim 14 7 6`, `python late_ilp_file.py ../results/nn_cat_d7_r1_v1_dag.stim 7` | `results/logs/nn_cat_*.log`, `results/nn_cat_*_k{5,6}.json` |
| S proxy of the d=7 checks: no malignant set of five or six faults | `python proxy_k56.py ../results/<stem>.stim 24 9 5` (and 6) | `results/<stem>_proxy_k{5,6}.json` |
| Appendix E 6: integer-program certificates at d=9 (ladder, verified cat without rounds) and d=7, in the three branches X, Z and C | `python late_ilp_run.py ladder 9 branch=X` (and Z, C), `python late_ilp_run.py vcat 9 branch=X` (and Z, C), `python late_ilp_run.py vcat 7 inject_dag` | `results/logs/late_ilp_ladder_d9.log`, `results/logs/late_ilp_vcat_d9.log`, `results/logs/late_ilp_vcat_d7.log` |
| The second solver: HiGHS confirms the d=9 ladder and the d=7 verified cat, and agrees with Gurobi on random placements | `python late_ilp_highs.py ladder 9 branch=X` (and Z, C), `python late_ilp_highs.py vcat 7 inject_dag`, `python late_ilp_highs_test.py 5 8 3` | `results/logs/late_ilp_highs.log` |
| Exact search without the integer program: nothing harmful up to six faults in the d=7 verified cat without rounds | `python vcat_d7_exact.py 4`, `python fd_k6.py ../results/vcat_d7_r1_injdag_nostab.stim 120 7 5` (and 6) | `results/vcat_d7_r1_injdag_nostab_k{5,6}.json` |
| The integer program against exact certificates, and as a bound on the S proxy | `python late_ilp_test.py KIND DELTA N SEED`, `python late_ilp_proxy_test.py 5 8 11`, `python proxy_ilp.py vcat 9` | `results/logs/late_ilp_endp_d5.log`, `results/logs/proxy_ilp_vcat.log` |
| The simpler late-error bound matches the exact fault distance (folds and unverified cats at d=3, 5, the central d=7 tree, end-parity grid checks) | `python late_bound_test.py endp 5 12 7`, `python late_bound_test.py nover 5 12 3`, `python late_bound_folds.py 5 c,bc,b1,cat 4` | `results/logs/late_bound_*.log` |
| Condition (i) of Theorem 32 for a check with stabilizer rounds before it | `python early_check.py PATH DELTA` | stdout |
| Proposition 30: unverified path cats grown from one end lose distance at d=5 | `python noverify_chain.py 5` | `results/logs/noverify_chain_d5.log` |
| Corollary 37: the 3n-qubit line check with the cat verified d - 1 times, at d=3 and 5 | `python line_check.py 3 2 3 proxy`, `python line_check.py 5 4 4 proxy` | `results/line_d*_v*.json` |

## Costs (Table IV, Appendix G)

Checks that start from an ideal encoder carry idle noise on their data (`src/idle.py`). It leaves every fault distance
unchanged, because each idle fault on a data qubit acts like a fault of the preceding gate on that qubit, but it changes
the leading terms and acceptances.

| Claim | Command | Output |
|---|---|---|
| Circuits with idle noise on the data, also without stabilizer rounds | `python data_idle_make.py` | `results/didle/*.stim`, `results/didle/make.json` |
| Fault distance, leading terms and acceptance | `python data_idle_rates.py 1e6 STEM:KMAX[:proxy] ...`, `python leading_k4.py ../results/didle/STEM.stim nopattern`, `python leading_k5.py ../results/didle/STEM.stim si1000` | `results/didle/rates.json`, `results/*_c4.json`, `results/*_c5.json` |
| Logical error per accepted state up to eight faults, and acceptance | `python data_idle_mitm.py si1000 1e-3,5e-4,2e-4,1e-4 1e6 40000 STEM ...`, `python data_idle_acc.py SHOTS MODEL P STEM ...` | `results/didle/mitm.json`, `results/didle/acceptance.json` |
| Resets and measurements gathered into few layers (column eps_g), with the same fault distances | `python nn_layers.py d5 4 1e-3 20000 w8d w8du` (also `ucat_d5`, `vcat_d5`, `vcat_d7`, `fold_n19_b1`, `fold_n7_c`, `vcat_d3`, `d3`, `full_d3`), `python late_ilp_file.py ../results/didle/nn_vcat_d7_w8d_didle.stim 7` | `results/didle/layers.json`, `results/logs/late_ilp_vcat_d7_gathered.log` |
| The rows of Table IV | `python data_idle_table.py` | `results/didle/table_si1000.json` |
| Appendix G 4: the complete nearest-neighbor protocol at d=3 | `python nn_full.py d3 3 1 0 1 3 shor routed` | `results/nn_full_*.stim`, `results/logs/nn_full_*.log` |
| Acceptance of any circuit file | `python acceptance.py MODEL P SHOTS PATH ...` | stdout |

## Figures

| Figure | Command | Output |
|---|---|---|
| Fig. 3 | `python fig_fd.py` | `../../paper/fig_fd.pdf` |
| Fig. 4 | `python fig_nn.py` | `../../paper/fig_nn.pdf` |
