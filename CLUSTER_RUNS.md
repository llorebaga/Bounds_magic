# Cluster runs worth trying

Only runs that are realistic on a cluster (or a large-memory node) and would change an entry of Table I.
Status as of 22 Oct 2026. Nothing here is mentioned in the paper. Background and the full history are in
`COMPUTE_PLAN.md`. Scripts to run A1, A2, A3, B1 and B2 on the RCI cluster of FEL CTU: `cluster/README.md`.

Conventions: commands run from `scripts/`. Measured kernel throughput: 1.2e8 signed codes per second on 16
cores (7.5e6 per core-second); exhaustive searches are embarrassingly parallel over code structures.
Lemma 13 of the paper (n_min(tau, d) >= n_min(tau, s) + d - s for non-stabilizer targets) propagates every
exclusion at order s to all higher orders.

## A. Lower bounds (exhaustive search)

### A1. Cubic CS: exclude n = 9  (CS, d = 3: LB 9 -> 10)
- Status: X-ranks a <= 5 done, no survivors (`results/log_fast_cs_n9_d3_a{3,4,5}.txt`). Remaining: a = 6
  (124 structures, run started but not finished: `results/log_fast_cs_n9_d3_a6.txt`) and a = 7 (10 structures).
- Inputs (already exported): `results/structs_r2_n9_d3_a6.txt`, `results/structs_r2_n9_d3_a7.txt`.
- Commands:
  ```bash
  julia -t 64 fast_enum.jl ../results/structs_r2_n9_d3_a6.txt 9 3 7
  julia -t 64 fast_enum.jl ../results/structs_r2_n9_d3_a7.txt 9 3 7
  ```
  (m = n - r = 7 for CS, r = 2.) Split a structure file into chunks of lines to spread it over nodes.
- Cost: about 1.2e13 signed codes; about 27 h on 16 cores, about 30 min on 1000 cores. Feasible on one
  workstation over a weekend.
- If no survivors: Proposition 5 gets "or with order three from n <= 9"; Table I CS d = 3 LB 10 (bold, source
  e, l); Table SII new row; Discussion gap "cubic CS (10 to 35)".

### A2. Two CCZ states, d = 2: exclude n = 10  (LB 10 -> 11)
- Command (Python search with the two-CCZ target filter; shard over X-ranks or with the shard arguments):
  ```bash
  R=6 TARGET=ccz2 python enum_codes.py 10 2
  ```
- Cost: about 1e12-1e13 signed codes. The Python search does about 1e5 signed codes per core-second
  (measured on the n = 9 runs), i.e. 3e3-3e4 core-hours, so use the kernel instead: export the structures
  (`R=6 python export_structs.py 10 2 a`, a = 0..4, 20-40 GB RAM each), then
  `julia fast_enum.jl ../results/structs_r6_n10_d2_a<a>.txt 10 2 4` (40-400 core-hours). Kernel survivors still
  need the two-CCZ target test.
- If no survivors: Proposition 6 "for quadratic suppression, at least eleven noisy ones"; Table I CCZ^{x2}
  d = 2 LB 11; Table SII row.

### A3. Cubic CCZ: exclude n = 10  (cubic LB 10 -> 11; by Lemma 13 quartic 11 -> 12 and the d >= 5 row
becomes max(2d+1, d+8), i.e. 13 at d = 5 and 14 at d = 6)
- Run without a target filter, as at n = 9, so that a null result again holds for every rational 3-qubit
  target of nullity three (Proposition 6).
- Sectors a <= 6: Python (`R=3 python enum_codes.py 10 3 0 6` with shards) or export plus kernel
  (`R=3 python export_structs.py 10 3 a`, then `julia fast_enum.jl ../results/structs_r3_n10_d3_a<a>.txt 10 3 7`).
- Top sector a = 7: 33 structures, 2^39 lifts each, input already exported:
  `results/structs_r3_n10_d3_a7.txt`.
- Code change done (22 Oct 2026): `fast_enum.jl` reads lift vectors as `UInt128` (a = 7, n = 10 gives 70 bits) and
  takes `SHARD=k NSHARDS=K` to split one structure's lift range over many jobs (each structure of the top
  sector is about 2,600 core-hours). Validated on the T search at n = 8, a = 5: full and three-way sharded runs
  reproduce the stored log exactly (910,163,968 signed codes, the same 346,136 survivors).
- Cost: about 2.3e15 signed codes (+~10 % for the lower sectors); about 4 days on 1000 cores (about 1e5
  core-hours).
- If no survivors: Proposition 6 "none has suppression order three with n <= 10"; Table I CCZ d = 3 LB 11,
  d = 4 LB 12, d >= 5 max(2d+1, d+8); Table SII row; Discussion gaps.

## B. Upper bounds (constructions; a result needs exact certification, see Sec. S5)

### B1. Two-level SAT: cubic CCZ below 36 inputs, quartic CCZ below 48
- Undecided after 15 min per instance on the laptop. A result would be a new bold upper bound (36 -> 32
  cubic, 48 -> 40 quartic) or UNSAT within the two-level model (not used in Table I).
- Instances (`'target D s nT nJ nC'`, try s from 3 to 7; for cubic also the `cs12cs13` form):
  ```bash
  # cubic, 32 inputs
  python sat_sweep.py c32.jsonl 86400 "ccz 3 4 0 0 4" "ccz 3 4 8 0 3" "ccz 3 4 0 2 3" "ccz 3 4 0 4 2" "ccz 3 4 0 8 0"
  # quartic, 40 inputs
  python sat_sweep.py q40.jsonl 86400 "ccz 4 5 0 0 5" "ccz 4 5 0 2 4"
  ```
  Each instance is independent; run one per core. For hard instances, cube-and-conquer (for example
  march_cu to split, CaDiCaL on the cubes) parallelizes a single instance over many cores.
- If SAT: certify the design exactly (`verify_*` scripts as for the six-block design), then update Table I UB,
  the main-text two-level paragraph and Sec. S5.

### B2. Two CCZ states, d = 2: in-class designs with 11-13 inputs  (UB 14 -> 12 or 13)
- Undecided after 15-20 min per instance (`results/mip/ccz2_d2.txt`, `scan2_*`, `scan3_*`).
- Command (one long run per (n, s); a SAT encoding of the same constraints is an alternative):
  ```bash
  python mip_scan.py ccz2 2 11 13 3,4,5,6 86400
  ```
- If feasible: certify exactly, then Table I CCZ^{x2} d = 2 UB.

## C. Not worth running (no gain or out of reach)
- Cubic T, exclude n = 9: no gain, Lemma 13 already gives LB 10. Excluding n = 10 needs about 7e16 signed
  codes (about 110 days on 1000 cores).
- Cubic CS, exclude n = 10: about 1e16, weeks on a cluster, then each further n costs x1000.
- Cubic/quartic CCZ, exclude n = 11: above 1e18.
- Optimality of the 120 p^3 and 864 p^4 coefficients by integer programming: the LP relaxation is zero.
- None of the runs in A or B closes a gap; each moves one bound by one or a few inputs.

## After any run
1. Store logs and outputs in `results/` (gzip large logs) and add the files to the matching page in
   `scripts/make_index.py`, then `python make_index.py`.
2. Update the paper: the proposition statement, Table I (bold, source letter), Table SII, the Discussion gaps,
   and the abstract or introduction if a headline number changes.
3. Add a README row with the exact command and output file.
