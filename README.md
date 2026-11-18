# Bounds_magic

Code and numerical results for two papers by Llorenç Balada Gaggioli and Jakub Mareček on the limits of magic-state
preparation.

| Paper | Directory |
|---|---|
| *The limits of magic-state distillation*: every stabilizer protocol reduces to one code on the inputs; exhaustive lower bounds, constructions and exact certificates | [`distillation/`](distillation/README.md) |
| *The limits of magic-state cultivation*: an exact theory of which faults are harmful with real T gates, the loss of half the distance in the checks in use, and checks that keep all of it | [`cultivation/`](cultivation/README.md) |

Each directory stands on its own. Its README lists the computational claims of the paper with the command that
reproduces each one and the file that holds the output.

```
distillation/
  scripts/   enumeration, verifiers, integer and SAT programs
  results/   logs, certificates and designs
  sdp/       stabilizer-cone relaxations and exact certificates
  by_target/ one page per entry of Table I
cultivation/
  checks/    state-vector checks of the theorems, the nine-T circuit, the published distance-three circuit
  evaluator/ the exact evaluator for large circuits: src, scripts, results, external
  data/      data of the published distance-three circuit
```

## License

MIT, see [`LICENSE`](LICENSE). Published circuits of other authors keep their own licences, listed in
[`cultivation/README.md`](cultivation/README.md).
