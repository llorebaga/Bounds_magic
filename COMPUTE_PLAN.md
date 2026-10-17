# What a large cluster would and would not decide (17 Oct 2026)

Throughput measured on this machine (16 cores, Julia kernel `fast_enum.jl`): about 1.2e8 signed codes per
second, i.e. 7.5e6 per core-second. The exhaustive searches are CPU-bound and embarrassingly parallel
(code structures are independent), and each worker needs only megabytes. A cluster therefore helps in
proportion to its core count. Large memory matters only for the Python stage that enumerates code
structures: the subspaces of F2^n number 1.1e8 for n=10 (20-40 GB as Python sets) and 3.5e9 for n=11
(about 1 TB). A streaming rewrite of that stage would remove the need. The kernel must also be widened to
128-bit integers when a*n > 64 (e.g. cubic CCZ n=10 a=7, T n=9 a=8); this is a small change.

| Open entry | What decides the next step | Size (signed codes) | This machine | 1000 cores | 1 TB RAM needed? | Verdict |
|---|---|---|---|---|---|---|
| T, d=3 (LB 9, UB 15) | exclude n=9: only sectors a=7 (83 structures, 2^27 lifts) and a=8 (7, 2^35); a<=6 already follows from the d=2 run (all 96 T-codes there have harmful weight-2 faults) | 6.4e13 | ~6 days | ~2.5 h | no | Worth doing: LB 10. n=10 is ~7e16 (~110 days on 1000 cores): out of reach. |
| CS, d=3 (LB 9, UB 35) | exclude n=9 (a=5: 358 structures, a=6: 124, a=7: 10 with 2^33 lifts) | 1.2e13 | ~27 h | ~30 min | no | Feasible even here: LB 10. n=10 is ~1e16: weeks on a cluster. |
| CCZ, d=3 and d=4 (LB 10, UB 36 / 48) | exclude n=10 (top sector a=7: 33 structures, 2^39 lifts, 128 signs) plus smaller sectors | 2.3e15 (+~10%) | ~8 months | ~4 days | helpful for the a<=6 structure enumeration (1e8 subspaces) | Cluster job: LB 11 for both orders. n=11 is >1e18: out of reach. |
| CCZ x CCZ, d=2, lower bound (LB 10, UB 14) | exclude n=10 (r=6, lifts <= 2^24): compute small, enumeration of 5e7 subspaces of F2^10 | ~1e12-1e13 (est.) | hours once enumeration fits in RAM | minutes | yes for the current code (or rewrite) | Easy on a big-memory node: LB 11. n=11 needs ~1 TB for enumeration and ~2e4 core-h: LB 12 possible. |
| CCZ x CCZ, d=2, upper bound (UB 14) | in-class MIP at n=11-13, s=3: all 9 instances timed out at 20 min | - | - | many long runs / seeds | no | Uncertain. Better: SAT encoding with cube-and-conquer on many cores. |
| Two-level SAT (5 cube blocks = 40-input quartic; 4 cube + 2 Jones; 9 Jones cubic) | undecided after 15 min | - | - | cube-and-conquer, hours-days | no | Reasonable cluster job; outcome uncertain either way. |
| Leading-coefficient MIP (cubic CCZ n=36) | LP bound stays 0 after 1 h | - | - | - | no | More compute does not help; needs a combinatorial reformulation. |
| T, d=2, n=9 sectors a=7, a=8 | not needed | - | - | - | - | Settled: the nine-input factory gives n_min(T,2)=9. |

Sizing files: `results/structs_r2_n9_d3_a{5,6,7}.txt` (cubic CS), `results/structs_r3_n10_d3_a7.txt` (cubic CCZ),
`results/structs_nogal_r1_n9_d3_a{7,8}.txt` (cubic T). MIP logs: `results/mip/scan2_*`, `results/mip/opt_*`.
