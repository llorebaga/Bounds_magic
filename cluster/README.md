# Running the cluster searches on RCI (FEL CTU)

Scripts for the runs of `CLUSTER_RUNS.md` on the RCI Slurm cluster. Everything is started from your
computer with `from_pc.sh` (MobaXterm local terminal or Git Bash; needs only `ssh` and `tar`), or directly on
the cluster with `rci.sh` (for example in a MobaXterm SSH tab, from `~/magic_runs`).

## First time
1. Get an RCI account (request form on the RCI wiki, login.rci.cvut.cz/wiki, FEL SSO).
2. In `cluster/config.sh` set `RCI_USER`. `RCI_HOST` defaults to `login3.rci.cvut.cz` (AMD nodes).
3. In the MobaXterm local terminal, go to the repository
   (`cd /drives/c/Users/<you>/Documents/2026/Magic/Bounds_magic`), then:
   ```bash
   bash cluster/from_pc.sh keys     # one password prompt, then key login
   bash cluster/from_pc.sh setup    # copies the code, builds the Python venv, finds Julia, runs a self-test
   bash cluster/from_pc.sh check    # partitions, cores and memory per node, modules, quota
   ```
   The self-test must print `signed candidates 536870912 survivors 0`.

## Runs (in order of cost)
| Run | What | Cost | Changes if negative/positive |
|---|---|---|---|
| `A1` | cubic CS, exclude n = 9 (kernel, X-ranks 6, 7) | ~430 core-h, 5 tasks | CS d = 3 LB 9 -> 10 |
| `B1` | two-level SAT, CCZ at 32 (cubic) and 40 (quartic) inputs, 35 instances | 35 x 23 h on 1 core | new UB if SAT (certified exactly) |
| `A2-export`, then `A2` | two CCZ, d = 2, exclude n = 10 (export, then kernel) | export 5 x <= 24 h, 120 GB; kernel sized after export | CCZ^2 d = 2 LB 10 -> 11 |
| `B2` | two-CCZ in-class MIP, n = 11-13, s = 3-6 (only with a Gurobi module) | 12 x 23 h on 16 cores | UB 14 -> 12 or 13 if feasible |
| `A3-export`, `A3-low`, `A3-top` | cubic CCZ, exclude n = 10 | ~95,000 core-h (top sector 86,000) | CCZ d = 3 LB 10 -> 11, d = 4 11 -> 12 |

```bash
bash cluster/from_pc.sh submit A1
bash cluster/from_pc.sh submit B1
bash cluster/from_pc.sh submit A2-export   # when finished (status shows 'exported ...'):
bash cluster/from_pc.sh submit A2
bash cluster/from_pc.sh status
bash cluster/from_pc.sh pull               # logs to results/cluster/
```
`A3-top` is refused unless you add `--yes`. Ask the RCI admins before such a run.

Every `submit` first copies the current `scripts/` and `cluster/` to the cluster. If you submit a run again,
only its unfinished tasks are resubmitted, so the same command resumes after a timeout or a failure.
Partitions are chosen from `sinfo`: the CPU partition (names starting with `amd` preferred) with the shortest
time limit that fits. `PARTITION=... bash cluster/from_pc.sh submit A1` overrides the choice. Other settings
are in `config.sh`. At most `ARRAY_THROTTLE` = 20 tasks of a run run at the same time.

## Reading the results
- Kernel runs: `status` sums the `RESULT` lines (signed codes, survivors). The kernel tests only the
  collapse conditions. With zero survivors the exclusion holds. Survivors still need the target test
  (`SURVIVOR` lines in `out/<run>/task_*.log`).
- B1/B2: `status` counts the `status` fields and lists any task that found a design.
- After a result, follow "After any run" in `CLUSTER_RUNS.md`.

## Kernel options used here
`scripts/fast_enum.jl` takes `SHARD=k NSHARDS=K` to run piece k of K of every structure's lift range, and it
reads lift vectors as `UInt128` (needed for X-rank 7 at n = 10, 70 bits). On the T search at n = 8
(X-rank 5) the full run and a three-way sharded run reproduce the stored log exactly: 910,163,968 signed codes
and the same 346,136 survivors.
