# Bounds_magic

Code and data for **"Every magic-state factory is a single code: exact limits and new constructions"**
(`paper/paper.tex`, `paper/refs.bib`, compiled `paper/paper.pdf`).

Every stabilizer protocol for magic-state distillation is equivalent to projecting the noisy T inputs onto one
stabilizer code; it tolerates a set of input faults exactly when they all *collapse* onto one code vector
(Theorem 1). For rational targets such as CCZ, the field automorphism sqrt2 -> -sqrt2 pairs every fault pattern
with its complement (Theorem 2). This repository contains the exhaustive searches that turn these reductions
into lower bounds, the integer/SAT programs that find constructions, and the exact verifiers that certify them.

## Layout

```
paper/     paper.tex, refs.bib, paper.bbl, paper.pdf
scripts/   all code (run from inside scripts/: paths are relative, e.g. ../results/)
results/   logs, JSON outputs, certificates and designs referenced in the paper
```

## Requirements

- Python >= 3.10 with `numpy`, `sympy`
- `gurobipy` (Gurobi 12; an academic license suffices) for the integer programs
- `python-sat` (`pip install python-sat`; CaDiCaL 1.9.5 backend) for the SAT programs
- Julia 1.10 (multi-threaded) for the compiled kernel `fast_enum.jl` and the independent check

Exact arithmetic everywhere except where noted (floating point only in the witness search of App. B and in
dense cross-checks).

## Claims and how to reproduce them

All commands are run from `scripts/`. Environment variables: `R` = number of logical qubits of the target,
`TARGET` = spectral target filter (`ccz`, `ccz2`, `cccz`, `cs`, `t`, `tt`), `GALOIS=0` disables the
complementary patterns (needed for non-rational targets such as T).

### Reductions and Clifford orbit (Sec. II, App. A)
| Claim | Command | Output |
|---|---|---|
| Prop. 3: CS12CS13\|+++> = CX23 H2 CX12 S2 H2 \|CCZ> (exact, sympy) | `python verify_orbit.py` | stdout |
| Clifford orbit of \|CCZ> has 69,120 states; its diagonal members | `python clifford_orbit_ccz.py` | stdout |

### Analytic bounds (Theorem 4, App. B)
| Claim | Command | Output |
|---|---|---|
| Witness identities, n >= 2d+1 (exact in Q(zeta_8)) | `python 41_check_free_witnesses.py` | `results/free_witness_checks.json` |
| Low-nullity witness, n >= d + nu + 1 (exact) | `python 52_check_low_nullity_witness.py 10` | `results/low_nullity_witness_checks.json` |
| Limits of witnesses (floating point) | `python 53_search_magic_witnesses.py 6 2 4` and `... 7 2 4` | stdout |

### Exhaustive searches (Theorems 5, 6, App. C, Table II)
| Claim | Command | Output |
|---|---|---|
| Any 3-qubit nu=3 target, d=2, n=7: no code (weight <= 1 only, 17,161,440 signed codes) | `GALOIS=0 R=3 python enum_codes_v2.py 7 2` | stdout |
| CCZ d=2, n=7 with complements (7,699,696) | `python enum_codes.py 7 2` | `results/enum_n7_d2_a0-4.json`, `log_n7_d2.txt` |
| CCZ d=3, n=7 / n=8 | `python enum_codes.py 7 3`; `python enum_codes.py 8 3 0 4` + shards `python enum_codes.py 8 3 5 5 s 12` | `results/enum_n8_d3_*.json` |
| CCZ d=3, n=9 (a<=5 Python shards; a=6 Julia kernel) | `python enum_codes.py 9 3 0 4`; `python enum_codes.py 9 3 5 5 s 10`; `R=3 python export_structs.py 9 3 6`; `julia -t 16 fast_enum.jl ../results/structs_r3_n9_d3_a6.txt 9 3 6` | `results/enum_n9_d3_*.json`, `log_fast_n9_d3_a6.txt` |
| CCZ d=4, n=9 (consistency check) | `python enum_codes.py 9 4 0 5`; kernel on `structs_r3_n9_d4_a6.txt` | `log_n9_d4_a0-5.txt`, `log_fast_n9_d4_a6.txt` |
| Two CCZ: exact cost 8; d=2 needs >= 10 | `R=6 TARGET=ccz2 python enum_codes.py 7 1`; `R=6 TARGET=ccz2 python enum_codes.py 9 2` | `results/enum_r6_ccz2_*.json` |
| CCCZ exact cost 6 (n=5 excluded) | `R=4 TARGET=cccz python enum_codes.py 5 1` | `results/enum_r4_cccz_n5_d1_a0-1.json` |
| CS: exact cost 3; d=2 needs >= 8; d=3 needs >= 9 | `R=2 TARGET=cs python enum_codes_v2.py {3,7} {1,2}`; `R=2 TARGET=cs python enum_codes_v2.py 8 3` | `results/enum_r2_cs_*.json`, `log_cs_n8_d3.txt` |
| One T state, d=2, n<=7 excluded | `GALOIS=0 R=1 TARGET=t python enum_codes_v2.py 7 2` | `results/enum_nogal_r1_t_n7_d2_a0-6.json` |
| One T state, d=2, n=8 (largest sectors, Julia) | `GALOIS=0 R=1 TARGET=t python export_structs.py 8 2 {6,7}`; `GALOIS=0 julia -t 16 fast_enum.jl ../results/structs_nogal_r1_n8_d2_a{6,7}.txt 8 2 7` | `log_fast_t_n8_d2_a7.txt`, `log_fast_t_n8_d2_a6.txt.gz` |
| Classification of the 890,940 n=8 collapse survivors (all encode stabilizer states) | `gunzip -k ../results/log_fast_t_n8_d2_a6.txt.gz; python check_t_survivors3.py ../results/log_fast_t_n8_d2_a6.txt 8` | `results/t_n8_survivor_classes.json` |

Status note: for one T state at n=8, sectors a<=4 (Python, `log_t_n8_d2_a0-5.txt`) and a=6,7 (Julia) are
complete without a T-state protocol; the a=5 sector (`GALOIS=0 R=1 TARGET=t python enum_codes_v2.py 8 2 5 5`)
was still running when this repository was created, so the paper states the bound n >= 8.

Validation (App. C): `python validate_enum.py 5 2` (brute force), `python validate_canon.py 7 2 300`
(canonicalization), `python validate_arith.py 7 2 30` (exact vs dense), `python control_cube.py` (8-T cube
factory), `julia -t 16 independent_check.jl 7 2` (independent implementation), `python diag.py` (which condition
fails first). The Julia kernel is cross-checked on `structs_nogal_r1_n7_d2_a6.txt` (402,653,184 signed codes).

### Integer programs and the 36-input cubic CCZ factory (Sec. IV, App. D)
| Claim | Command | Output |
|---|---|---|
| Controls: cube 8T (d=2), 7T infeasible, 15-to-1 | `python tri_mip.py ccz 8 1 2`; `python tri_mip.py ccz 7 2 2`; `python tri_mip.py t 15 4 3` | stdout |
| 36-column CS12CS13 matrix, distance 3 | `python tri_mip.py cs12cs13 36 6 3 3600` | `results/mip/cs12cs13_36.json` |
| Exact certificate: Clifford correction, order 3, 120 harmful weight-3 patterns | `python verify_tri2.py ../results/mip/cs12cs13_36.json 3` | `results/mip/cs12cs13_36_cert.json` |
| In-class infeasibility scans (not used in Table I) | `python mip_scan.py TARGET d n_min n_max s_list timelimit` | `results/mip/*.txt` |

### Two-level factories (Sec. IV, App. D)
| Claim | Command | Output |
|---|---|---|
| SAT model controls | `python sat_block.py ccz 2 1 8 0 0`; `python sat_block.py ccz 2 2 7 0 0`; `python sat_block.py ccz 1 0 0 1 0` | stdout |
| Quartic CCZ from six cube blocks (48 T), certified, 54 harmful block pairs -> 864 p^4 | `python sat_sweep.py out.jsonl 600 "ccz 4 5 0 0 6"` | `results/sat/quartic_ccz_6cube_s{4,5,6,7}.json`, `results/sat/A2.jsonl` |
| Exact certificate of a design | `python verify_block.py ../results/sat/quartic_ccz_6cube_s5.json` | stdout |
| Negative / undecided instances listed in App. D | see `results/sat/*.log`, `*.jsonl` | |

`block_mip.py` is the Gurobi version of the same two-level model (much slower than the SAT encoding).
