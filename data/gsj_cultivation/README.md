# Distance-three cultivation circuit of Gidney, Shutty and Jones

Files in this directory are taken from https://github.com/Strilanc/magic-state-cultivation at commit
`871e68ff6df2f75190b1bfd6351459d1b5a037e3` (Apache License 2.0, copy in `LICENSE`):

- `d3_init_unitary.stim`: the stim circuit of `make_chunk_d3_init_unitary` in
  `src/cultiv/_construction/_injection_stage.py` (unitary injection into the distance-three colour code,
  including its stabilizer round), verbatim apart from leading whitespace.
- `d3_double_cat_check.stim`: the stim circuit of `make_chunk_d3_double_cat_check` in
  `src/cultiv/_construction/_cultivation_stage.py` (the T check), verbatim apart from leading whitespace.
- `stats_d3_unitary.csv`: the rows of `assets/stats.csv` for the circuit `inject[unitary]+cultivate` with
  `d1 = 3`, decoders `vec_intercept_t`, `vec_intercept_s`, `vec_intercept_z` (state-vector simulation with
  T, S or Z gates at the marked positions) and `perfectionist` (stabilizer simulation, S gates).

In these circuits S and S_DAG stand for T and T^dagger. `scripts/cultivation_gsj.py` composes the two chunks
(the MPP lines at the chunk boundaries are the boundary conditions of the chunks and are dropped, except the
final stabilizer measurements, which are kept as an ideal decoder), replaces S and S_DAG by T and T^dagger,
and analyses the result with the characterization of `paper/cultivation.tex`. The original code composes the
chunks with `gen.compile_chunks_into_circuit` and re-layers the result; we did not run it, so our idle-noise
layering follows the TICKs of the chunk text.
