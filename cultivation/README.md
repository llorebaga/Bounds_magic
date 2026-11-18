# The limits of magic-state cultivation

Code, circuits and numerical results for **"The limits of magic-state cultivation"** by Llorenç Balada Gaggioli and
Jakub Mareček.

Theorem, proposition, table, figure and appendix numbers below refer to the paper. All paths are relative to this
directory.

Magic-state cultivation has been designed and benchmarked with every T gate replaced by an S gate. The paper gives an
exact theory of which faults are harmful with real T gates, shows that the checks in use reach a fault distance of only
about half their code distance, and builds checks, also with nearest-neighbor gates, that keep the full distance.

## Layout

```
checks/      state-vector checks of the theorems, the nine-T circuit and the published distance-three circuit
  cultiv.py    their library: circuits, state vectors and the fault map (D, E) of Theorem 1
  results/     JSON outputs of cultivation_t9.py, cultivation_gsj.py and cultivation_codes.py
evaluator/   the exact evaluator for large circuits, on stim and Gauss sums (see evaluator/README.md)
  src/         library: T-aware detector error model, searches, circuit builders, noise models
  scripts/     the computations behind the paper, run from inside scripts/
  results/     circuits (.stim), JSON outputs and logs/
  external/    published circuits used as input
data/        stim chunks and state-vector data of the published distance-three circuit
```

## Requirements

Python 3.10 or later with `numpy`, `scipy` and `stim` 1.16. The integer programs use `gurobipy` (Gurobi 12; an
academic licence suffices) and, for the second solver, `highspy`. The figures use `matplotlib`.

## Where to start

- A fast test of Theorem 1: `cd checks && python cultiv_test.py 300 11`.
- The real-T fault distance of a circuit file: `cd evaluator/scripts && python fd_gauss.py ../results/vcat_d5_r1.stim 4 nopattern`.
- The computations behind Secs. VI to VIII and Appendices C, E, F and G: [`evaluator/README.md`](evaluator/README.md).

## State-vector checks

The scripts in `checks/` verify the analytical statements on small circuits and codes, independently of the evaluator.
They run from any directory, print `ALL OK` when every check passes, and need only `numpy`, `scipy` and `stim`, except
where noted.

| Statement | Command (in `checks/`) | Output |
|---|---|---|
| Theorem 1: the fault map (D, E) decides harmlessness exactly, against direct simulation (42,367 configurations) and dense skeleton maps (9,271) | `python cultiv_test.py 300 11` | stdout |
| Theorem 1 step by step: gadget identity, the Pauli form of every fault, the collapse criterion, E = Z_j for Z faults at T gates | `python cultivation_check.py 10 7` | stdout |
| Proposition 16 (any non-Clifford gates; 3,327 configurations of 30 random circuits with T, Z rotations, CS, CCZ and random single-qubit gates), Lemma 5 for CSS codes and any angle, Theorem 4 on the [[7,1,3]] and [[19,1,5]] codes | `python cultivation_general_check.py` | stdout |
| Theorem 19 against state vectors, and Proposition 20 (variable elimination) against brute force on 2,000 random quadratic forms | `python cultivation_gauss_check.py` | stdout |
| Lemma 15 and Theorem 22: the construction for the nine-input, CCZ and 15-to-1 codes (Appendix D); order 3 and spreading distance 2 of the code of the published distance-three circuit | `python cultivation_codes.py` (about 5 min) | `results/codes.json` |
| Proposition 24: nine T gates, eleven qubits and 34 two-qubit gates give fault distance two; the case split of the proof; leading terms 53 p^2 and 1.9e3 p^2 | `python cultivation_t9.py` | `results/t9_circuit.json` |
| Proposition 24: the integer program over the fault map, which finds that one flag suffices (Gurobi, about 1 min) | `python cultivation_t9.py --mip` | `results/t9_circuit.json`, key `mip` |
| Appendix D 1: with S or Z gates the nine-T circuit has 8 and 74 harmful single faults; with two flagged rounds it meets the condition of Simsek et al. with T gates but not with S or Z gates | `python proxy_understates.py` | stdout |
| Proposition 10, distance three: the published circuit has 48 harmful pairs with T gates and none with S gates, 96 with idle noise, leading term 32 p^2/75; 4 of 20 projections enter the code; the fit of Fig. 2(b) | `python cultivation_gsj.py` | `results/gsj_d3.json` |
| Appendix F 2: the 48 harmful pairs form 9 pairs of fault classes, and 12 classes when grouped as in Chan et al. | `python cultivation_gsj_classes.py` | stdout |
| Lemma 5 and its two neighboring cases on the [[7,1,3]] code, for every T/T-dagger pattern whose check fixes the T state | `python cultivation_completion.py` | stdout |
| Lemma 27, Theorem 28, Lemma 31 and the decomposition behind Theorem 32, on the [[7,1,3]] and [[19,1,5]] codes | `python cultivation_late_check.py` | stdout |
| Theorem 28 with Z stabilizers of either sign (the RP^2 check of Appendix F 3) | `python cultivation_signed_late_check.py` | stdout |
| Lemma 33 and Theorem 8: every late configuration that is harmful with S gates is harmful with T gates | `python cultivation_proxy_bound_check.py` | stdout |
| Corollary 29: every face of the triangular color codes up to distance 11 meets a minimum-weight logical support (integer programs with HiGHS through scipy) | `python cultivation_face_cover_check.py 11` | stdout |
| Proposition 9: a correcting final stage, on the [[7,1,3]] code | `python cultivation_final_stage_check.py` | stdout |
| Figs. 2 and 5 (TikZ sources) | `python make_fig_cultivation.py` | `../paper/fig_cult_gsj.tex`, `../paper/fig_cult_small.tex` |

## Published circuits and their licences

| Source | Where | Licence |
|---|---|---|
| Gidney, Shutty and Jones, *Magic state cultivation*: circuit chunks and state-vector data of the distance-three circuit | `data/gsj_cultivation/` (see its README) | Apache License 2.0 |
| Gidney, Shutty and Jones, sampled `inject[unitary]+cultivate` circuits (Zenodo, doi:10.5281/zenodo.13777072) | `evaluator/external/gidney_circuits/` | CC BY 4.0 |
| Chen, Chen, Lu and Pan, RP^2 cultivation circuits (github.com/Zihan-Chen-PhMA/Cultiv_T_RP2) | `evaluator/external/rp2_circuits/` (with their LICENSE) | MIT License |
| Rosenfeld et al., *Magic state cultivation on a superconducting quantum processor* (arXiv:2512.13908): the circuits linked from the paper | not included; `evaluator/scripts/google_crumble_to_stim.py` decodes them from the paper's Crumble links | |

Only the circuit files of these works are read; their generator code was not run.

## Licence

MIT, see `LICENSE` at the top of the repository.
