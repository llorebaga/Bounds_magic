"""Scan schedules of the measurement-prepared nearest-neighbour protocol (src/nn_cat.py, inject="measure").
Usage: python nn_sched.py d3 SCHED[,SCHED...] [unfold]"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import multiprocessing as mp
import nn_cat
from noise import si1000
from fold import merge_final_mpps, regenerate_detectors

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def run(args):
    which, sched, unfold = args
    from nn_cat_fd import certify
    if which == "d3":
        faces, n, L = nn_cat.STEANE, 7, nn_cat.RING3
    else:
        from colorcode import color_code_faces
        faces, n = color_code_faces(5)
        L = nn_cat.ladder(8)
    c, G = nn_cat.make_nn_cat(faces, n, inject="measure", schedule=sched, unfold=unfold, **L)
    c = regenerate_detectors(merge_final_mpps(c))
    name = f"nn_meas_{which}_{sched}" + ("_unf" if unfold else "")
    c.to_file(os.path.join(R, name + ".stim"))
    try:
        a = certify(c, 3, proxy=False)["T"]
        b = certify(si1000(c, 1e-3), 3, proxy=False)["T"]
    except Exception as e:
        return dict(name=name, err=str(e))
    return dict(name=name, ticks=c.num_ticks, fd=a["fd"], mal=a["malignant"], c_unif=a["coef_p1e3"],
                c_si=b["coef_p1e3"])


if __name__ == "__main__":
    which = sys.argv[1]
    unfold = len(sys.argv) > 3 and sys.argv[3] == "unfold"
    jobs = [(which, s, unfold) for s in sys.argv[2].split(",")]
    with mp.Pool(min(len(jobs), 14)) as pool:
        for r in pool.imap(run, jobs):
            print(json.dumps(r), flush=True)
