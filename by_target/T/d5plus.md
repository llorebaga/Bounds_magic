# One T state, d >= 5

Table I: prior LB d | LB **max(2d, d+7)** (Theorem 4 and Lemma 13: 12 at d = 5, 13 at d = 6) | UB 49 at d = 5 (Bravyi-Haah) | prior UB 49 at d = 5

[All targets](../README.md)

## Lower bound

Lemma 13 (trading order for inputs): for non-stabilizer tau, n_min(tau, d) >= n_min(tau, s) + d - s, by feeding d - s inputs with maximally mixed qubits I/2 = rho_T(1/2). With the d = 2 search it gives n >= d + 7.
Theorem 4, last claim: every order-d protocol (d >= 2) uses n >= 2d inputs. Proof (Sec. S3): single branch, the
sqrt2 -> -sqrt2 automorphism forces complements of harmless patterns to output exactly Z|T>, and the two-qubit
inequality g01 - eta^2 g10 >= eta |g00 - g11| (eta = 3 - 2 sqrt2, checked on all 60 two-qubit stabilizer states)
applied to singlets x |+->^{2t} gives an induction ending in zero acceptance at n = 2d - 1. Found as the
certificate of a linear-programming relaxation of the stabilizer cone; independent exact certificates of that
relaxation for d = 2..7. For d = 5 this gives 10 (the search bound is 9).

Further files (logs, kernel inputs and outputs, certificates):

- [`sdp/proof_check.py`](../../sdp/proof_check.py)
- [`sdp/exact_cert_fast.py`](../../sdp/exact_cert_fast.py)
- [`sdp/theorem_T2d.tex`](../../sdp/theorem_T2d.tex)
- [`sdp/README.md`](../../sdp/README.md)

Reproduce (from `scripts/` unless stated):

```bash
cd sdp
python proof_check.py
python exact_cert_fast.py 2,3,4,5,6,7
```

## Upper bound (explicit construction)

Bravyi-Haah [[49,1,5]] (literature) at d = 5; no explicit bound listed for d >= 6.
