# Directions to strengthen the cultivation paper

Running list from the section-by-section review of `paper/cultivation.tex` (started 2026-10-31).
Each entry: what, why it helps, rough cost. Tags: **[theory]**, **[compute]**, **[external]**, **[writing]**.
Status: open unless marked.

## Highest impact

1. **Converse of Corollary 18: code-relative fault tolerance is sufficient** [theory, hard].
   Today Cor. 18 gives only the upper bound FD <= s + delta - |u|. A matching lower bound would make the
   design rule an exact criterion, and Proposition 6 (verified cat, NN ladder) would hold for every delta
   instead of being certified up to 7. Possible route: Theorem 14's bound FD >= min(d_C, d_sp) with a
   code-relative spreading distance in place of d_sp, evaluated through the Gauss sums of Theorem 11.
   Caveat: the naive one-branch screen gives distance one on every circuit, so the proof must use the
   cancellations g_1 = 0.
2. **The loss is universal: apply the criterion to surface-code cultivation** [compute + external, medium].
   Run Theorems 1-2 on the published circuits of Chen et al. (RP^2, PRX Quantum 7, 010315) and Sahay et al.
   (fold-transversal, PRX Quantum 7, 033006). If their real-T fault distance is also about half the proxy,
   "half the distance is lost" covers every published family. First check that their stim circuits are public.
   Takada et al. (2609.16929) simulate fold-transversal surface-code cultivation but give no fault-distance
   verdict; their figures use p^((f+1)/2) guide lines.
3. **Classification of malignant configurations** [theory, medium].
   Every malignant configuration found (folds at delta = 3, 5, 7, the published circuits, Google's circuit)
   is Lemma 4 or one of its two neighbouring cases (Y completion; X on all of l with odd Z). Proving this
   for GHZ-fold checks would make "(delta+1)/2" a theorem rather than a pattern.

3b. **(delta+1)/2 upper bound for every delta** [theory, medium] (from "Real T gates halve the distance").
   Construct, for a GHZ fold of the kind used, (delta-1)/2 undetected hooks that each put X on two qubits of
   one minimum-weight logical support, plus one data fault; Cor. 18 then gives FD <= (delta+1)/2 for every
   delta, not only 3, 5, 7. Needs a condition on the fold tree (pairs of sibling leaves inside a logical
   support) that we can check for the trees we use. Turns Prop. 5(ii) partly into a theorem.

3c. **A full-distance grid check for every delta** [theory, medium-hard] (from "Keeping the distance").
   The ring and ladder generalise to a family (path grown from its middle, end parity, rungs and diagonals).
   Prove that the family is code-relatively fault tolerant for every delta; together with direction 1 this
   gives nearest-neighbour checks of full real-T fault distance at every distance, not only 3, 5 and >= 6 of 7.
   Even without direction 1, a proof that one fault reaches at most one data qubit and two faults at most two
   (the rungs and diagonals argument) is a clean statement.

## Theory, within the framework

4. **Theorem 2 for r-qubit targets** [theory, cheap]. With deterministic checks on an r-qubit target the
   same proof gives 2^r Gauss sums. Makes the exact test cover |CS> and |CCZ> cultivation (Min and Heo,
   2609.09993, 2609.09994). A remark plus a few lines in Sec. S2.
5. **Proposition 7 for rates** [theory, cheap]. The leading coefficient of an adaptive protocol is a
   probability-weighted combination of the branch coefficients; partly settles the "Rates" open problem.
6. **Theorem 1 beyond Pauli noise** [theory, medium]. State what survives for local non-Pauli noise and
   spell out the coherent-noise limit (Hines et al. 2603.18457 show coherent errors raise the cost).

6b. **Gate lower bounds for real-T checks** [theory, hard] (open problem in the Outlook). Code-relative fault
   tolerance is weaker than fault tolerance of the cat, so SpiderCat-type bounds (2603.05391) apply only at
   delta = 3 (>= 8 cat CNOTs). For delta >= 5 a new constraint problem over code-relative conditions is needed
   (minimum CNOTs / ancillas for a check of full real-T fault distance). Would complement the T-count theorem
   with a two-qubit-gate bound; notes in `cultivation2/notes/lower_bounds_direction.md`.

## Computation

7. **Fault distance of the delta = 7 ladder** [compute, running time]. Finish the weight-6 search
   (`python fd_k6.py ../results/nn_cat_d7_r1_v1_unf.stim 14 7`, killed earlier). If clean, FD = 7 exactly
   and Table I / Prop 6 lose their ">=".
8. **|CS> cultivation as a non-T target** [compute, medium]. Run Theorem 1 on Min and Heo's circuit.
9. **Independent cross-check with merlin** [external + compute, cheap]. merlin (Cercelescu, Koch, Pesah,
   2610.06811) is open source. Reproducing one or two fault distances gives a third, outside implementation.
   Also: the two in-house implementations are cross-checked only on the delta = 3 Gidney circuit; extend
   the cross-check to the delta = 3 verified cat and the ring.
10. **A 15-T circuit with better acceptance** [compute, medium]. Our fault-distance-three 15-T circuit
    accepts one fault-free run in 1024. The integer program of Sec. S5 could search for a better one.

10b. **A delta = 9 fold data point** [compute, medium-hard]. Fig. 3 has delta = 3, 5, 7. An explicit malignant
    configuration of 5 faults on the [[61,1,9]] fold is cheap (direction 3b); certifying that no 4 faults are
    harmful is a weight-4 search like those at delta = 7 (minutes to hours). Makes "the gap grows" visible.
10c. **Best fold tree** [compute, cheap]. The leading term of the delta = 7 fold ranges from 0.011 to 15 p^4 over
    six trees, three orders of magnitude at no cost. A tree search (and the same for Google's grid fold)
    would give experimentalists a free improvement while the distance loss remains. Report it as a side result.

10d. **Full nearest-neighbour protocol at delta = 5** [compute, medium-hard]. Only delta = 3 is done with every
    operation noisy (FD 3). At delta = 5 the state must be prepared and grown on the grid; this is what makes
    the ladder a real protocol and would replace "start from an ideally encoded state" in Prop. 6.
10e. **Cheaper stabilizer measurements on the grid** [compute + theory, medium]. The complete delta = 3 protocol
    loses to Google's circuit above p ~ 3e-4 only because of slow stabilizer rounds. Ideas: measure stabilizers
    with the same path cat, interleave them with the un-growing, or use the route through |0>_L and a
    projecting first check. Already listed as an open problem in Sec. S13.
10f. **Head-to-head with Chan et al.'s repaired circuits** [external + compute, medium]. Chan et al. restore
    fault distance 4 at delta = 5 with 13 two-qubit flags (5 with a fourth stabilizer round). Comparing their
    circuits with the ladder (qubits, layers, acceptance, leading term under SI1000) would show which repair is
    cheaper. Needs their circuits (check whether public).
10g. **Logical error of the un-grown ladder** [compute, cheap-medium]. Table SIX omits epsilon for the un-grown
    ladder because the p^6 order dominates at p = 1e-3 under SI1000. Computing the weight-6 term (or sampling)
    would give the number the cost comparison is missing.
10h. **Proxy fault distance of the delta = 7 checks** [compute, cheap]. Table I lists the delta = 7 proxies as
    >= 5 while the real-T ladder is >= 6, which looks odd. A weight-5 (and weight-6) proxy search with stim
    would remove the inconsistency.

## Writing

11. **State the leading term early** [writing, cheap]. One sentence in "Cultivation and fault distance":
    eps ~ c_d p^d, with c_d summing malignant configurations weighted by acceptance and infidelity.
12. **Quote the Gauss-sum speed-up** [writing, cheap]. About 1 ms per class against 0.22 s for direct
    summation at kappa = 20.
12b. **Takada's guide lines** [external, cheap]. Takada et al. plot p^((f+1)/2) guides for surface-code
    cultivation; check whether their data actually follow (f+1)/2 and, if so, cite as independent evidence.
12c. **Main-text figure of a delta = 5 malignant configuration** [writing, cheap]. Fig. 2 shows delta = 3 only;
    a small panel with two hooks and one completion at delta = 5 would show the general mechanism.
12d. **Archive the code with a DOI** [writing, cheap, before submission]. Make a Zenodo release of the
    repository (GitHub integration) and cite it in the data-availability statement instead of only the URL.
13. **Escape stage** [external, large]. Combine Mehta et al.'s noisy-syndrome postselection
    (2610.06828) with the exact real-T evaluation, to remove the ideal decoder from our comparisons.
