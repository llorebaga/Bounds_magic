# Bounds_magic

Code and numerical results for *The limits of magic-state distillation*, by Llorenç Balada Gaggioli and
Jakub Mareček: every stabilizer protocol reduces to one code on the inputs; exhaustive lower bounds,
constructions and exact certificates.

Everything is in [`distillation/`](distillation/README.md), the directory the paper cites. Its README lists every
computational claim of the paper with the command that reproduces it and the file that holds the output.

```
distillation/
  scripts/   enumeration, verifiers, integer and SAT programs
  results/   logs, certificates and designs
  sdp/       stabilizer-cone relaxations and exact certificates
  by_target/ one page per entry of Table I
```

## License

MIT, see [`LICENSE`](LICENSE).
