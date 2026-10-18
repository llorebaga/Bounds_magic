# CCZ (and any rational non-stabilizer target), d >= 5

Table I: prior LB d | LB **2d+1** (analytic, Theorem 4) | UB O(d) | prior UB -

[All targets](../README.md)

## Lower bound

Theorem 4: n >= 2d + 1 for rational targets (witnesses |0>^n and the GHZ-type state G_N), and n >= d + nu + 1 for
dyadic targets (low-nullity witness). Exact checks of the witness identities below.

Further files (logs, kernel inputs and outputs, certificates):

- [`results/free_witness_checks.json`](../../results/free_witness_checks.json)
- [`results/low_nullity_witness_checks.json`](../../results/low_nullity_witness_checks.json)

Reproduce (from `scripts/` unless stated):

```bash
python 41_check_free_witnesses.py
python 52_check_low_nullity_witness.py 10
```

## Upper bound (explicit construction)

Asymptotically good codes with transversal CCZ fed with four-T CCZ preparations (literature).
