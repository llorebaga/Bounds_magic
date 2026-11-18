"""Nearest-neighbour checks with resets and measurements gathered into few moments (Builder.window / .period).

SI1000 charges every idle qubit 2p in a moment that contains a reset or a measurement, and in the ASAP schedule of
the ladder at delta = 5 most moments do. Here the same checks are scheduled with resets and measurements allowed
only every P-th moment (period P) or delayed by up to W moments to share one (window W); idle noise is then added
to the data (src/idle.py). For each variant: moments, moments with a reset or measurement, real-T fault distance
up to KMAX (uniform and SI1000), and the SI1000 logical error per accepted state at p from the rare-event estimator.
Usage: python nn_layers.py d3|d5|d7|full_d3|fold_n7_c|fold_n19_b1|vcat_d3|ucat_d5|vcat_d5|vcat_d7 KMAX P SAMPLES VARIANT...   (VARIANT = pN or wN, then d to defer the terminal
                                                                          measurements and u for the un-grown cat)
-> results/didle/nn_<d>_<variant>_didle.stim, results/didle/layers.json"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
import nn_cat
from fold import merge_final_mpps, regenerate_detectors
from colorcode import color_code_faces
from idle import add_data_idle, _segments
from noise import si1000
from tdem import TDEM
from mitm_rate import MITM
from mc import sample
from nn_cat_fd import certify

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "didle")


def build_other(which, variant):
    """the GHZ folds and all-to-all cats of the comparison without stabiliser rounds (data_idle_make.py), with the
    same Builder schedule options set on the class"""
    from cult_circuits import Builder
    from vcat import make_vcat
    from fold import make_fold
    from data_idle_make import tree, same
    v = variant.rstrip("d")
    Builder.defer = variant.endswith("d")
    Builder.window, Builder.period = (int(v[1:]), 0) if v[0] == "w" else (0, int(v[1:]))
    try:
        if which == "full_d3":                   # the complete protocol: noisy injection, stabiliser cats, 3 checks
            import nn_full
            return nn_full.build("d3", 3, 1, 0, 1, True)
        if which.startswith("fold"):
            stem, bw, sd = {"fold_n7_c": ("fold_n7_c", 3, "c"), "fold_n19_b1": ("fold_n19_b1", 5, "b1")}[which]
            faces, n, parent, root, reps = tree(bw, sd)
            Builder.window, Builder.period, Builder.defer, saved = 0, 0, False, (Builder.window, Builder.period,
                                                                               Builder.defer)
            stored = stim.Circuit.from_file(os.path.join(OUT, "..", stem + ".stim"))
            dag = [g for g in (False, True) if same(make_fold(faces, n, inject_dag=g, parent=parent, root=root,
                                                              reps=reps), stored)][0]
            Builder.window, Builder.period, Builder.defer = saved
            return make_fold(faces, n, inject_dag=dag, parent=parent, root=root, reps=reps, stab_rounds=False)
        name, bw, verify, dag = {"vcat_d3": ("vcat_d3_r1", 3, True, True), "ucat_d5": ("ucat_d5_r1", 5, False, False),
                                 "vcat_d5": ("vcat_d5_r1", 5, True, False), "vcat_d7": ("vcat_d7_r1", 7, True, True)}[which]
        faces, n = color_code_faces(bw)
        return merge_final_mpps(make_vcat(faces, n, rounds=1, inject_dag=dag, verify=verify, stab_rounds=False))
    finally:
        Builder.window, Builder.period, Builder.defer = 0, 0, False


def build(which, variant):
    if not which.startswith("d"):
        return build_other(which, variant)
    if which == "d3":
        faces, n, L, dag = nn_cat.STEANE, 7, nn_cat.RING3, True
    else:
        faces, n = color_code_faces(int(which[1:]))
        L, dag = nn_cat.ladder({5: 8, 7: 17}[int(which[1:])]), which == "d7"
    unf = variant.endswith("u")
    v = variant.rstrip("u")
    kw = dict(defer=v.endswith("d"))
    v = v.rstrip("d")
    kw.update(dict(period=int(v[1:])) if v[0] == "p" else dict(window=int(v[1:])))
    c, G = nn_cat.make_nn_cat(faces, n, inject_dag=dag, rounds=1, ver_rounds=1, unfold=unf, **kw, **L)
    return regenerate_detectors(merge_final_mpps(c))


if __name__ == "__main__":
    which, kmax, p, samples = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(float(sys.argv[4]))
    path = os.path.join(OUT, "layers.json")
    for variant in sys.argv[5:]:
        t0 = time.time()
        c, _ = add_data_idle(build(which, variant))
        stem = f"nn_{which}_{variant}_didle"
        c.to_file(os.path.join(OUT, stem + ".stim"))
        segs = _segments(c)
        heavy = sum(any(i.name in ("R", "RX", "M", "MX") for i in s) for s in segs)
        fu = certify(c, kmax, proxy=False)["T"]
        cs = si1000(c, p)
        fs = certify(cs, kmax, proxy=False)["T"]
        m = TDEM(cs, magic="T", pattern=False)
        pb, se, rows = MITM(m).P_bad(1.0, 8, samples)
        acc = sample(TDEM(cs, magic="T", pattern=False), 300000, seed=1)["P_acc"]
        row = dict(circuit=stem, ticks=len(segs), heavy=heavy, fd_uniform=fu["fd"], fd_si1000=fs["fd"],
                   coef_si1000=fs["coef_p1e3"], p=p, P_acc=acc, eps=pb / acc, se_eps=se / acc,
                   strata=[(k, a, s) for k, a, s in rows], seconds=round(time.time() - t0))
        print(json.dumps({k: row[k] for k in row if k != "strata"}), flush=True)
        res = json.load(open(path)) if os.path.exists(path) else []
        res = [x for x in res if (x["circuit"], x["p"]) != (stem, p)] + [row]
        json.dump(res, open(path, "w"), indent=1)
