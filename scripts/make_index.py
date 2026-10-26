"""Generate by_target/: one page per (target, suppression order d) of Table I.

Each page gives the Table I entry, how the lower bound is proved (with an automatically generated table of
every exhaustive run for that target and d, read from results/enum_*.json), the explicit construction behind
the upper bound, the commands that reproduce both, and related explorations that are not used in Table I.

Run from scripts/:  python make_index.py
"""
import glob
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "by_target")


def classify(fname):
    """Map an enum_*.json file name to (target, tolerated-set description)."""
    b = os.path.basename(fname)
    if b.startswith("enum_nogal_r1_t_"):
        return "T", "weights < d"
    if b.startswith("enum_nogal_r2_tt_"):
        return "TT", "weights < d"
    if b.startswith("enum_nogal_n"):
        return "CCZ", "weights < d only (any 3-qubit target with nu = 3)"
    if b.startswith("enum_r2_cs_"):
        return "CS", "weights < d and > n-d (complements)"
    if b.startswith("enum_r4_cccz_"):
        return "CCCZ", "{0, 1} (complements), target filter CCCZ"
    if b.startswith("enum_r4_n"):
        return "CCCZ", "{0, 1} (complements), any 4-qubit target with nu = 4"
    if b.startswith("enum_r6_ccz2_"):
        return "CCZ2", "weights < d and > n-d, target filter two CCZ"
    if b.startswith("enum_r6_n"):
        return "CCZ2", "{0, 1} (complements), any 6-qubit target with nu = 6"
    if b.startswith("enum_ccz_n"):
        return "CCZ", "{0, 1} (complements), target filter CCZ"
    if b.startswith("enum_n"):
        return "CCZ", "weights < d and > n-d (complements)"
    return None, None


def enum_runs():
    runs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "enum_*.json"))):
        target, H = classify(f)
        if target is None:
            continue
        try:
            st = json.load(open(f))["stats"]
        except Exception:
            continue
        per_a = st.get("per_a", {})
        ranks = sorted(int(a) for a in per_a)
        signed = sum(v.get("signed_candidates", 0) for v in per_a.values())
        surv = st.get("survivors")
        if isinstance(surv, list):
            surv = len(surv)
        m = re.search(r"_s(\d+)of(\d+)", f)
        shard = f"shard {int(m.group(1)) + 1}/{m.group(2)}" if m else ""
        runs.setdefault((target, st["d"]), []).append(dict(
            n=st["n"], ranks=f"{ranks[0]}-{ranks[-1]}" if ranks else "", signed=signed, surv=surv,
            sec=st.get("seconds", 0), H=H, file=os.path.relpath(f, ROOT).replace("\\", "/"), shard=shard))
    return runs


def link(path, depth=2):
    return f"[`{path}`]({'../' * depth}{path})"


# --------------------------------------------------------------------------------------------------
# Catalog: Table I entries (prior LB, LB, UB, prior UB) and how each is established.
# Paths are relative to the repository root.
# --------------------------------------------------------------------------------------------------
CAT = {}

CAT[("T", "d2")] = dict(
    title="One T state, quadratic suppression (d = 2)", row="prior LB 2 | LB **9** (exhaustive) | UB **9** (new factory, 27/2 p^2) | prior UB 10 (Meier-Eastin-Knill)",
    lb="""No stabilizer protocol with n <= 8 inputs (Proposition 5). Tolerated set: weights < 2 (T is not rational, so no
complements). n <= 7: Python search with target test. n = 8: X-rank <= 4 in Python (target test), X-ranks 5, 6, 7
with the compiled Julia kernel (collapse test only), then the 890,940 (a=6) and 346,136 (a=5) collapsing codes
are classified exactly by their logical Bloch moduli: all stabilizer states except 928 codes with moduli
{0, 1/3, 2 sqrt2/3}; none is Clifford-equivalent to |T>. Total 1.32e11 signed codes.""",
    lb_files=["results/log_t_n7_d2.txt", "results/log_t_n8_d2_a0-5.txt", "results/structs_nogal_r1_n8_d2_a5.txt",
              "results/structs_nogal_r1_n8_d2_a6.txt", "results/structs_nogal_r1_n8_d2_a7.txt",
              "results/log_export_t_n8_a5.txt", "results/log_fast_t_n8_d2_a5.txt.gz", "results/log_fast_t_n8_d2_a6.txt.gz",
              "results/log_fast_t_n8_d2_a7.txt", "results/log_fast_t_n8_d2_a5.txt.nonstab.json",
              "results/t_n8_survivor_classes.json"],
    lb_cmd="""GALOIS=0 R=1 TARGET=t python enum_codes_v2.py 7 2
GALOIS=0 R=1 TARGET=t python enum_codes_v2.py 8 2 0 4
GALOIS=0 R=1 TARGET=t python export_structs.py 8 2 5   # and 6, 7
GALOIS=0 julia -t 16 fast_enum.jl ../results/structs_nogal_r1_n8_d2_a5.txt 8 2 7   # and a6, a7
gunzip -k ../results/log_fast_t_n8_d2_a5.txt.gz; python check_t_survivors3.py ../results/log_fast_t_n8_d2_a5.txt 8""",
    ub="""Nine-input factory found by the exhaustive search at n = 9 (X-ranks <= 6, 2.2e11 signed codes, 2.5e7 collapsing
codes classified; 96 encode |T>, all equivalent). Proposition 7: measure the eight stabilizers listed in the paper,
postselect on +1 (probability 1/128), decode; error 27/2 p^2 + O(p^3). Exact certificate: single-fault minors,
logical Bloch moduli {0, 1/sqrt2, 1/sqrt2}, full error polynomial (harmful patterns 36, 45, 90, 75, 76, 15, 6, 1
at weights 2..9).""",
    ub_files=["results/log_fast_t_n9_d2_a6.txt.T.json", "results/t9_factory_cert.txt",
              "results/log_fast_t_n9_d2_a1.txt.gz", "results/log_fast_t_n9_d2_a6.txt.gz",
              "results/structs_nogal_r1_n9_d2_a6.txt"],
    ub_cmd="""GALOIS=0 R=1 TARGET=t python export_structs.py 9 2 6
GALOIS=0 julia -t 16 fast_enum.jl ../results/structs_nogal_r1_n9_d2_a6.txt 9 2 8
python classify_parallel.py ../results/log_fast_t_n9_d2_a6.txt 9 16
python verify_t_factory.py ../results/log_fast_t_n9_d2_a6.txt.T.json 5 9""",
    other="""- In-class integer program: no generalized triorthogonal T factory with n <= 9 and <= 6 check rows
  ([`results/mip/t_d2.txt`](../../results/mip/t_d2.txt), `python mip_scan.py t 2 ...`). Not used in Table I.
- Stabilizer-cone LP relaxation (sdp/) excludes only n <= 2d - 1 = 3 here; the search is much stronger at d = 2.""")

CAT[("T", "d3")] = dict(
    title="One T state, cubic suppression (d = 3)", row="prior LB 3 | LB **10** (Lemma 13 + d = 2 search) | UB 15 | prior UB 15 (35 p^3, Bravyi-Kitaev 15-to-1)",
    lb="""Lemma 13 (trading order for inputs): for non-stabilizer tau, n_min(tau, d) >= n_min(tau, s) + d - s, by feeding d - s inputs with maximally mixed qubits I/2 = rho_T(1/2).
With n_min(T, 2) >= 9 ([d2](d2.md)) this gives n >= 10 at d = 3. Excluding n = 10 needs new searches.""",
    lb_files=["results/structs_nogal_r1_n9_d3_a7.txt", "results/structs_nogal_r1_n9_d3_a8.txt"],
    lb_cmd="(see d2)",
    ub="""Bravyi-Kitaev 15-to-1 (literature). Our integer program recovers it and proves that 35 harmful weight-3 patterns
is optimal for n = 15, s = 4 (generalized triorthogonal form).""",
    ub_files=["results/mip/opt_t_n15_s4.json"],
    ub_cmd="""python tri_mip.py t 15 4 3
python tri_mip_opt.py t 15 4 300""",
    other="")

CAT[("T", "d4")] = dict(
    title="One T state, quartic suppression (d = 4)", row="prior LB 4 | LB **11** (Lemma 13 + d = 2 search) | UB 48 | prior UB 48 (Wills-Jain-Singh)",
    lb="Lemma 13 (trading order for inputs): for non-stabilizer tau, n_min(tau, d) >= n_min(tau, s) + d - s, by feeding d - s inputs with maximally mixed qubits I/2 = rho_T(1/2). With n_min(T, 2) >= 9 ([d2](d2.md)) this gives n >= 11; the analytic bound 2d = 8 is weaker here.",
    lb_files=[], lb_cmd="(see d2)",
    ub="Wills-Jain-Singh list a 48-input single-T protocol with d_Z = 4 (literature, arXiv:2609.30860, Table 2); the Bravyi-Haah [[49,1,5]] code (order 5) also gives order 4 with 49 inputs.",
    ub_files=[], ub_cmd="", other="")

CAT[("T", "d5plus")] = dict(
    title="One T state, d >= 5", row="prior LB d | LB **max(2d, d+7)** (Theorem 4 and Lemma 13: 12 at d = 5, 13 at d = 6) | UB 49 at d = 5 (Bravyi-Haah) | prior UB 49 at d = 5",
    lb="""Lemma 13 (trading order for inputs): for non-stabilizer tau, n_min(tau, d) >= n_min(tau, s) + d - s, by feeding d - s inputs with maximally mixed qubits I/2 = rho_T(1/2). With the d = 2 search it gives n >= d + 7.
Theorem 4, last claim: every order-d protocol (d >= 2) uses n >= 2d inputs. Proof (Sec. S3): single branch, the
sqrt2 -> -sqrt2 automorphism forces complements of harmless patterns to output exactly Z|T>, and the two-qubit
inequality g01 - eta^2 g10 >= eta |g00 - g11| (eta = 3 - 2 sqrt2, checked on all 60 two-qubit stabilizer states)
applied to singlets x |+->^{2t} gives an induction ending in zero acceptance at n = 2d - 1. Found as the
certificate of a linear-programming relaxation of the stabilizer cone; independent exact certificates of that
relaxation for d = 2..7. For d = 5 this gives 10 (the search bound is 9).""",
    lb_files=["sdp/proof_check.py", "sdp/exact_cert_fast.py", "sdp/theorem_T2d.tex", "sdp/README.md"],
    lb_cmd="""cd sdp
python proof_check.py
python exact_cert_fast.py 2,3,4,5,6,7""",
    ub="Bravyi-Haah [[49,1,5]] (literature) at d = 5; no explicit bound listed for d >= 6.",
    ub_files=[], ub_cmd="", other="")

CAT[("CS", "d1")] = dict(
    title="Controlled-S state CS|++>, d = 1 (ideal T states)", row="prior LB 2 | LB 3 (dagger: from Beverland et al.'s canonical form) | UB 3 | prior UB 3",
    lb="""Rational target with nu = 2: L(CS) >= nu + 1 = 3 (Theorem 4). The search also finds no code on fewer inputs.""",
    lb_files=["results/52_unused"] if False else [], lb_cmd="R=2 TARGET=cs python enum_codes_v2.py 3 1",
    ub="Measure Z1Z2Z3 on three T states, keep +1, decode (worked example, Sec. S1); error 3p(1-p).",
    ub_files=["scripts/worked_example.py"], ub_cmd="python worked_example.py", other="")

CAT[("CS", "d2")] = dict(
    title="Controlled-S state, quadratic suppression (d = 2)", row="prior LB 2 | LB **8** (exhaustive) | UB 8 (dagger: CCZ factory + Y measurement, Beverland et al. Prop. A.4) | prior UB 12",
    lb="""No protocol with n <= 7 (Proposition 5). Tolerated set includes complements (Theorem 2). Signed codes tested at
n = 7: 99,429,248.""",
    lb_files=["results/log_cs_d2.txt"],
    lb_cmd="R=2 TARGET=cs python enum_codes_v2.py 7 2",
    ub="""Eight-input CCZ cube factory followed by a Y measurement of one output qubit (Proposition 10: projecting the third
qubit of |CCZ> onto (|0> +- i|1>)/sqrt2 leaves CS^{+-1}|++>). Order two is inherited from the cube factory.""",
    ub_files=["scripts/control_cube.py", "scripts/worked_example.py"],
    ub_cmd="python control_cube.py",
    other="""- In-class integer program: no generalized triorthogonal CS factory with n <= 10 and <= 3 check rows
  ([`results/mip/cs_d2.txt`](../../results/mip/cs_d2.txt)). Not used in Table I.""")

CAT[("CS", "d3")] = dict(
    title="Controlled-S state, cubic suppression (d = 3)", row="prior LB 3 | LB **9** (exhaustive; also Lemma 13 + d = 2 search) | UB 35 | prior UB 35 (Nezami-Haah code, restricted as in Wills-Jain-Singh)",
    lb="""No protocol with n <= 8 (Proposition 5); 9,240,117,248 signed codes at n = 8 with complements.""",
    lb_files=["results/log_cs_n8_d3.txt"],
    lb_cmd="R=2 TARGET=cs python enum_codes_v2.py 8 3",
    ub="35-input triorthogonal-family factory (literature: Nezami-Haah, Wills-Jain-Singh).",
    ub_files=[], ub_cmd="",
    other="""- n = 9 (open): structure files and partial kernel runs (X-ranks 3-5 complete with no collapsing code; X-rank 6
  partial): `results/structs_r2_n9_d3_a*.txt`, `results/log_export_cs_n9_d3_a*.txt`, `results/log_fast_cs_n9_d3_a*.txt`.
  Not used in the paper.""")

CAT[("CCZ", "d1")] = dict(
    title="CCZ, d = 1 (ideal T states)", row="prior LB 3 | LB 4 (dagger) | UB 4 | prior UB 4 (Jones)",
    lb="Rational target with nu = 3: L(CCZ) >= nu + 1 = 4 (Theorem 4).",
    lb_files=[], lb_cmd="python enum_codes.py 4 1",
    ub="Jones' four-T CCZ preparation; recovered by the search at n = 4.",
    ub_files=[], ub_cmd="", other="")

CAT[("CCZ", "d2")] = dict(
    title="CCZ, quadratic suppression (d = 2)", row="prior LB 3 | LB **8** (exhaustive) | UB 8 (28 p^2) | prior UB 8 (Eastin, Jones, Haah-Hastings)",
    lb="""No protocol with n <= 7 (Proposition 6). At n = 7 two independent runs: (i) tolerated set = weights <= 1 only,
valid for any 3-qubit target with nu = 3 (17,161,440 signed codes); (ii) with complements (7,699,696 signed codes).
Confirmed by a separately written Julia implementation. Worked example Step 6 follows one rejected n = 7 code.""",
    lb_files=["results/log_n7_d2.txt", "results/structs_r3_n7_d2_a4.txt", "scripts/independent_check.jl"],
    lb_cmd="""GALOIS=0 R=3 python enum_codes_v2.py 7 2
python enum_codes.py 7 2
julia -t 16 independent_check.jl 7 2""",
    ub="""Eight-T cube factory (stabilizers: X on even and Y on odd cube vertices, Z on faces); error 28 p^2; among codes of
X-rank <= 4 it is the unique collapsing code up to symmetry. Worked example Step 7 and exact control script.""",
    ub_files=["results/log_n8_d2_a0-4.txt", "results/structs_r3_n8_d2_a1.txt", "scripts/control_cube.py", "scripts/worked_example.py"],
    ub_cmd="""python control_cube.py
python enum_codes.py 8 2 0 4
python tri_mip.py ccz 8 1 2""",
    other="")

CAT[("CCZ", "d3")] = dict(
    title="CCZ, cubic suppression (d = 3)", row="prior LB 3 | LB **10** (exhaustive) | UB 36 (120 p^3) (dagger: Wills-Jain-Singh CS12CS13 code + Howard-Campbell equivalence) | prior UB 47 (236 p^3)",
    lb="""No protocol with n <= 9 (Proposition 6), complements included. n = 7, 8 in Python (X-rank 5 at n = 8 in 12
shards); n = 9: X-ranks <= 5 in Python (X-rank 5 in 10 shards) and X-rank 6 with the Julia kernel (9.13e11 signed
codes in total at n = 9). Validated against a second implementation at n = 7.""",
    lb_files=["results/log_n7_d3.txt", "results/log_n8_d3_a0-4.txt", "results/log_n8_d3_a5_s0.txt",
              "results/structs_r3_n8_d3_a5.txt", "results/log_n9_d3_a0-4.txt", "results/log_n9_d3_a5_s0.txt",
              "results/structs_r3_n9_d3_a6.txt", "results/log_export_n9_d3_a6.txt", "results/log_fast_n9_d3_a6.txt"],
    lb_cmd="""python enum_codes.py 7 3
python enum_codes.py 8 3 0 4; python enum_codes.py 8 3 5 5 s 12   # shards 0..11
python enum_codes.py 9 3 0 4; python enum_codes.py 9 3 5 5 s 10   # shards 0..9
R=3 python export_structs.py 9 3 6; julia -t 16 fast_enum.jl ../results/structs_r3_n9_d3_a6.txt 9 3 6""",
    ub="""36-input factory: the Wills-Jain-Singh CS12CS13 class code is a CCZ factory because CS12CS13|+++> is
Clifford-equivalent to |CCZ> (Proposition 3). Our integer program finds such a matrix independently (Sec. S5);
exact certificate: Clifford correction, order 3, 120 of 7140 weight-3 patterns harmful.""",
    ub_files=["results/mip/cs12cs13_36.json", "results/mip/cs12cs13_36_cert.json", "results/mip/cs12cs13_d3_n36.txt",
              "scripts/verify_orbit.py", "scripts/clifford_orbit_ccz.py"],
    ub_cmd="""python verify_orbit.py
python tri_mip.py cs12cs13 36 6 3 3600
python verify_tri2.py ../results/mip/cs12cs13_36.json 3""",
    other="""- Leading-coefficient MIP at n = 36: no design below 120 within an hour (dual bound 0):
  `results/mip/opt_cs12cs13_n36_s6.json`, `results/mip/opt_cs12cs13_36.log`.
- Two-level SAT below 36 inputs (CCZ and CS12CS13 forms): 28 inputs UNSAT, 32 inputs undecided after 15 min:
  `results/sat/C1..C6.jsonl` (instance lists `listC*.txt`; fields target D s nT nJ nC).
- Jones-block-only designs: `results/sat/B*.jsonl`; Gurobi version `results/blockmip/jones_d3.txt`; mixed-input
  enumerations `results/log_mixed_*.txt`.
- n = 10 sizing (open): `results/structs_r3_n10_d3_a7.txt`, `results/log_export_ccz_n10_d3_a7.txt`.""")

CAT[("CCZ", "d4")] = dict(
    title="CCZ, quartic suppression (d = 4)", row="prior LB 4 | LB **11** (Lemma 13 + cubic search) | UB 48 (**864 p^4**) | prior UB 48 (2816 p^4)",
    lb="""Lemma 13 (trading order for inputs): for non-stabilizer tau, n_min(tau, d) >= n_min(tau, s) + d - s, by feeding d - s inputs with maximally mixed qubits I/2 = rho_T(1/2). With n_min(CCZ, 3) >= 10 ([d3](d3.md)) this gives n >= 11.
The quartic run at n = 9 (complements, 1.39e11 signed codes) is an independent consistency check.""",
    lb_files=["results/log_n9_d4_a0-5.txt", "results/structs_r3_n9_d4_a6.txt", "results/log_export_n9_d4_a6.txt",
              "results/log_fast_n9_d4_a6.txt"],
    lb_cmd="""python enum_codes.py 9 4 0 5
julia -t 16 fast_enum.jl ../results/structs_r3_n9_d4_a6.txt 9 4 6""",
    ub="""Two-level design: six eight-T cube blocks (48 T inputs) in an outer code with 3 logical + 5 check registers, found
by SAT and certified exactly; 54 harmful two-block error configurations, each contributing 16 p^4, so 864 p^4.""",
    ub_files=["results/sat/quartic_ccz_6cube_s5.json", "results/sat/quartic_ccz_6cube_s4.json",
              "results/sat/quartic_ccz_6cube_s6.json", "results/sat/quartic_ccz_6cube_s7.json", "results/sat/A2.jsonl"],
    ub_cmd="""python sat_sweep.py out.jsonl 600 "ccz 4 5 0 0 6"
python verify_block.py ../results/sat/quartic_ccz_6cube_s5.json""",
    other="""- Cube-block sweeps (fields target D s nT nJ nC): `results/sat/A*.jsonl` / `listA*.txt`; two and three cube blocks
  UNSAT; six SAT with 4-7 check rows (54-98 harmful two-block configurations); five cube blocks undecided after 15 min.
- Mixtures with Jones blocks or T columns: `results/sat/M4.jsonl` (undecided).
- Gurobi version of the two-level model: `results/blockmip/cube_d4.txt` (two cube blocks infeasible in 459 s).""")

CAT[("CCZ", "d5plus")] = dict(
    title="CCZ (and any rational non-stabilizer target), d >= 5", row="prior LB d | LB **max(2d+1, d+7)** for CCZ (Theorem 4 and Lemma 13: 12 at d = 5); 2d+1 for any rational target | UB O(d) | prior UB -",
    lb="""Theorem 4: n >= 2d + 1 for rational targets (witnesses |0>^n and the GHZ-type state G_N), and n >= d + nu + 1 for
dyadic targets (low-nullity witness). Exact checks of the witness identities below.""",
    lb_files=["results/free_witness_checks.json", "results/low_nullity_witness_checks.json"],
    lb_cmd="""python 41_check_free_witnesses.py
python 52_check_low_nullity_witness.py 10""",
    ub="Asymptotically good codes with transversal CCZ fed with four-T CCZ preparations (literature).",
    ub_files=[], ub_cmd="", other="")

CAT[("CCZ2", "d1")] = dict(
    title="Two CCZ states, d = 1 (ideal T states)", row="prior LB 6 | LB **8** (exhaustive) | UB 8 | prior UB 8 (two Jones preparations)",
    lb="No code on 7 inputs encodes two CCZ states (Proposition 6).", lb_files=[],
    lb_cmd="R=6 TARGET=ccz2 python enum_codes.py 7 1",
    ub="Two copies of Jones' four-T preparation.", ub_files=[], ub_cmd="", other="")

CAT[("CCZ2", "d2")] = dict(
    title="Two CCZ states, quadratic suppression (d = 2)", row="prior LB 6 | LB **10** (exhaustive) | UB 14 | prior UB 14 (Campbell-Howard)",
    lb="No protocol with n <= 9 (Proposition 6); 3,678,216 signed codes at n = 9 with complements.",
    lb_files=["results/log_ccz2_n9_d2.txt"], lb_cmd="R=6 TARGET=ccz2 python enum_codes.py 9 2",
    ub="Campbell-Howard 14-input protocol (literature).", ub_files=[], ub_cmd="",
    other="""- In-class MIP scans at n = 11-13 (undecided): `results/mip/ccz2_d2.txt`, `results/mip/scan2_*`, `results/mip/scan3_*`.
- Two-level SAT with Jones blocks and T columns (UNSAT within the model): `results/sat/D.jsonl`, `listD.txt`.""")

CAT[("CCCZ", "d1")] = dict(
    title="CCCZ state, d = 1 (ideal T states)", row="prior LB 4 | LB **6** (exhaustive) | UB 6 | prior UB 6 (Beverland et al.)",
    lb="No code on 5 inputs encodes CCCZ (Proposition 6); 44 signed codes survive pruning and none collapses.",
    lb_files=[], lb_cmd="R=4 TARGET=cccz python enum_codes.py 5 1",
    ub="""Proposition 11: project |T>^6 onto the [[6,4,2]] code with Z^6 = +1 and X^6 = -1 (probability 1/4); the result is
X^{(x)4}|CCCZ> up to Clifford phases. Exact check below.""",
    ub_files=["scripts/verify_cccz.py"], ub_cmd="python verify_cccz.py", other="")

NAMES = {"T": "One T state", "CS": "CS|++>", "CCZ": "CCZ", "CCZ2": "Two CCZ states", "CCCZ": "CCCZ"}
ORDER = [("T", "d2"), ("T", "d3"), ("T", "d4"), ("T", "d5plus"), ("CS", "d1"), ("CS", "d2"), ("CS", "d3"),
         ("CCZ", "d1"), ("CCZ", "d2"), ("CCZ", "d3"), ("CCZ", "d4"), ("CCZ", "d5plus"),
         ("CCZ2", "d1"), ("CCZ2", "d2"), ("CCCZ", "d1")]


def page(target, dkey, runs):
    e = CAT[(target, dkey)]
    L = [f"# {e['title']}", "", f"Table I: {e['row']}", "", "[All targets](../README.md)", "",
         "## Lower bound", "", e["lb"], ""]
    dnum = int(dkey[1]) if dkey[1:].isdigit() else None
    rs = sorted(runs.get((target, dnum), []), key=lambda r: (r["n"], r["file"])) if dnum else []
    if rs:
        L += ["Exhaustive runs for this target and d (generated from `results/enum_*.json`; signed codes = codes not",
              "removed by Lemma 8 and tested in exact arithmetic; surviving codes = codes satisfying the collapse",
              "conditions, and, in runs whose file name names the target (e.g. `_cs_`, `_cccz_`), also its Pauli spectrum):", "",
              "| n | X-ranks | tolerated set | signed codes tested | surviving codes | time (s) | file |",
              "|---|---|---|---|---|---|---|"]
        for r in rs:
            L.append(f"| {r['n']} | {r['ranks']} | {r['H']}{' (' + r['shard'] + ')' if r['shard'] else ''} | "
                     f"{r['signed']:,} | {r['surv']} | {r['sec']:.0f} | {link(r['file'])} |")
        L.append("")
    if e["lb_files"]:
        L += ["Further files (logs, kernel inputs and outputs, certificates):", ""]
        L += [f"- {link(f)}" for f in e["lb_files"]]
        L.append("")
    if e["lb_cmd"]:
        L += ["Reproduce (from `scripts/` unless stated):", "", "```bash", e["lb_cmd"], "```", ""]
    L += ["## Upper bound (explicit construction)", "", e["ub"], ""]
    if e["ub_files"]:
        L += [f"- {link(f)}" for f in e["ub_files"]]
        L.append("")
    if e["ub_cmd"]:
        L += ["```bash", e["ub_cmd"], "```", ""]
    if e["other"]:
        L += ["## Related explorations (not used in Table I)", "", e["other"], ""]
    return "\n".join(L)


def main():
    runs = enum_runs()
    os.makedirs(OUT, exist_ok=True)
    idx = ["# Results by target and suppression order", "",
           "One page per entry of Table I of the paper. Each page states the bound, how it is proved (exhaustive runs",
           "per n, with the files they produced), the explicit construction behind the upper bound, the commands that",
           "reproduce both, and related explorations not used in the paper. Generated by `scripts/make_index.py`.", "",
           "| target | d | page |", "|---|---|---|"]
    for target, dkey in ORDER:
        os.makedirs(os.path.join(OUT, target), exist_ok=True)
        with open(os.path.join(OUT, target, f"{dkey}.md"), "w", encoding="utf-8") as fh:
            fh.write(page(target, dkey, runs))
        d = "d >= 5" if dkey == "d5plus" else dkey[1:]
        idx.append(f"| {NAMES[target]} | {d} | [{target}/{dkey}.md]({target}/{dkey}.md) |")
    idx += ["", "Explorations without a Table I entry: T (x) T at d = 2 (`results/enum_nogal_r2_tt_*.json`,",
            "`results/log_tt_d2.txt`, `results/mip/tt_d2.txt`) and the stabilizer-cone relaxations in `sdp/`.", ""]
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(idx))
    print("wrote", len(ORDER), "pages")


if __name__ == "__main__":
    main()
