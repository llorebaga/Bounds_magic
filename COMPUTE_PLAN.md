# Remaining work: what was done, what needs a cluster, what is out of reach (17 Oct 2026)

Measured throughput (16 cores, Julia kernel `fast_enum.jl`): 1.2e8 signed codes per second, i.e. 7.5e6 per
core-second. Exhaustive searches are CPU-bound and embarrassingly parallel (independent code structures,
megabytes per worker). Memory matters only for the Python structure enumeration at n >= 10: subspaces of
F2^n number 1.1e8 (n=10, 20-40 GB as Python sets) and 3.5e9 (n=11, about 1 TB). The kernel needs 128-bit
lift vectors when a*n > 64 (small change).

## A. Done on this machine (17 Oct 2026, each under 2 h)
| Task | Result |
|---|---|
| One T state, d=2, n=8, last sector (a=5) | no T protocol: n_min(T,2) >= 9 |
| One T state, d=2, n=9, sectors a <= 6 | 96 codes encode T: nine-input factory, n_min(T,2) = 9 (Prop. 5) |
| In-class MIP, T, n=9, s=6 (the shape of the nine-input factory) | infeasible (869 s): the factory is not generalized triorthogonal; s=1..6 all infeasible at n=9 |
| Two-level SAT, cubic CCZ below 36 inputs | 3 cube + 4 T (28 inputs): UNSAT for CCZ and CS12CS13 forms. 32 inputs (4 cube; 3 cube + 8 T; 3 cube + 2 Jones; 2 cube + 4 Jones; 8 Jones): UNSAT for the CS12CS13 form, TIMEOUT (15 min) for the CCZ form |
| Two-CCZ in-class MIP, d=2, n=11-13 (s=3) and n=12 (s=4,5) | all TIMEOUT (15-20 min each) |
| Leading-coefficient MIP, cubic CCZ n=36 | no design below 120 in 1 h; relaxation bound 0 |
| Mixed T^k columns in the matrix model | proved equivalent to the plain model (parities only) |

## B. Doable, but needs a cluster or a large-memory node
| Task | Size | 1000 cores | Gain if negative / positive |
|---|---|---|---|
| Cubic CS, exclude n=9 (a=5: 358, a=6: 124, a=7: 10 structures) | 1.2e13 signed codes | ~30 min (about 27 h here) | LB 9 -> 10 |
| Cubic T, exclude n=9 (only a=7: 83 and a=8: 7 structures needed) | 6.4e13 | ~2.5 h (about 6 days here) | LB 9 -> 10 |
| Cubic and quartic CCZ, exclude n=10 (top sector a=7: 33 structures, 2^39 lifts) | 2.3e15 (+~10%) | ~4 days | LB 10 -> 11 for both |
| Two CCZ, d=2, exclude n=10 | ~1e12-1e13, enumeration-bound | minutes on a big-memory node | LB 10 -> 11 |
| Two CCZ, d=2, exclude n=11 | ~5e14 + 1 TB enumeration | ~1 day | LB -> 12 |
| Two-level SAT, CCZ form, 32-input cubic and 40-input quartic designs | undecided at 15 min | cube-and-conquer, hours-days | a 32-input cubic or 40-input quartic CCZ factory, or UNSAT within the model |
| Two-CCZ in-class MIP/SAT, n=11-13, s=3-6 | undecided at 15-20 min | many long runs, or a SAT encoding with cube-and-conquer | a 12- or 13-input factory (UB 14) |

## C. Out of reach with present methods (even on a large cluster)
| Task | Size | Why |
|---|---|---|
| Cubic T, exclude n=10 | ~7e16 | ~110 days on 1000 cores |
| Cubic CS, exclude n=10 | ~1e16 | weeks on a cluster, then each further n x1000 |
| Cubic/quartic CCZ, exclude n=11 | >1e18 | top sector 2^49 lifts per structure |
| Closing 10-36 (cubic CCZ), 10-48 (quartic), 9-35 (cubic CS), 9-15 (cubic T) | - | needs analytic progress (e.g. the Z^n-stabilizer conjecture) or new search ideas, not compute |
| Proving optimality of the 864 p^4 / 120 p^3 coefficients by MIP | - | LP relaxation is zero; needs a combinatorial reformulation |

## D. Further ways to use MIP/SAT (assessment)
1. **Larger check-row counts for d=2 targets.** The Wills-Jain-Singh classification covers d >= 3 only, so
   in-class searches at d=2 with many check rows are new territory. This settled the structure of the
   nine-input T factory (s=6 infeasible). For two CCZ states, s=4,5 at n=12 timed out; worth a longer run.
2. **Two-level and mixed-input designs (SAT).** Most promising constructive route: inner factories now include
   the eight-input CS factory and the nine-input T factory. Cube-and-conquer would decide the 32/40-input
   instances.
3. **SAT/MIP proposal + exact check (counterexample loop).** Encode only cheap linear necessary conditions on
   a code (no hidden logical, sector structure, for T the condition that complements of tolerated patterns
   encode Z|T>), let the solver propose candidates, test each exactly with the kernel and add a blocking
   clause. This reaches non-triorthogonal codes like the nine-input T factory, which no linear matrix model
   captures.
4. **Not useful:** coefficient minimization at large n (weak relaxation); mixed T^k columns (equivalent);
   MIP for general lower bounds (collapse conditions are not linear).

Sizing files: `results/structs_r2_n9_d3_a{5,6,7}.txt`, `results/structs_r3_n10_d3_a7.txt`,
`results/structs_nogal_r1_n9_d3_a{7,8}.txt`. Logs: `results/mip/scan2_*`, `scan3_*`, `opt_*`, `t_d2.txt`,
`results/sat/C*.jsonl`.
