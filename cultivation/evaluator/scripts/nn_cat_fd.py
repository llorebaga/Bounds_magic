"""Real-T and proxy fault distance of the nearest-neighbour verified-cat checks (src/nn_cat.py).
Usage: python nn_cat_fd.py d3 [KMAX]"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
from tdem import TDEM
from fd_fast import fault_distance_fast
from branch import FastExact
from fold import merge_final_mpps, regenerate_detectors
import nn_cat

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def certify(c, kmax, proxy=True):
    out = {}
    for magic in ("T", "S") if proxy else ("T",):
        m = TDEM(c, magic=magic, pattern=False)
        ev = FastExact(m) if magic == "T" else m
        k, f = fault_distance_fast(ev, kmax, m.mechanisms(), verbose=False)
        coef = sum(x[4] * x[3] for x in f if x[4] is not None) / 1e-3 ** k if k else None
        out[magic] = dict(fd=k if k else f">{kmax}", malignant=len(f), coef_p1e3=coef, n0=m.n0, t=m.t)
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if which == "d3":
        # the ring of the paper: one check round, end parity measured twice (rounds=2, ver=1: nn_cat_d3_dag.stim)
        from noise import si1000
        L = nn_cat.RING3
        for rounds, vr, name in ((1, 2, "nn_cat_d3_r1_v2"), (1, 1, "nn_cat_d3_r1_v1"), (2, 1, "nn_cat_d3_dag")):
            for dag in (False, True):
                c, G = nn_cat.make_nn_cat(nn_cat.STEANE, 7, inject_dag=dag, rounds=rounds, ver_rounds=vr, **L)
                c = regenerate_detectors(merge_final_mpps(c))
                try:
                    r = certify(c, kmax)
                    r["T_si1000"] = certify(si1000(c, 1e-3), kmax, proxy=False)["T"]
                except Exception as e:
                    print(name, "inject_dag", dag, "ERR", e); continue
                print(name, "inject_dag", dag, "qubits", c.num_qubits, "ticks", c.num_ticks, r, flush=True)
                c.to_file(os.path.join(R, name + ".stim"))
    elif which.startswith("d5"):
        # d5[-nodiag][-vN]: ladder layout, optional diagonals, N verification rounds
        from colorcode import color_code_faces
        faces, n = color_code_faces(5)
        L = nn_cat.ladder(8, diagonals="nodiag" not in which)
        vr = int(which.split("-v")[1]) if "-v" in which else 1
        for dag in (False, True):
            c, G = nn_cat.make_nn_cat(faces, n, inject_dag=dag, ver_rounds=vr, **L)
            c = regenerate_detectors(merge_final_mpps(c))
            try:
                r = certify(c, kmax, proxy=len(sys.argv) > 3)
            except Exception as e:
                print("inject_dag", dag, "ERR", e); continue
            print(which, "inject_dag", dag, "qubits", c.num_qubits, r, flush=True)
            c.to_file(os.path.join(R, f"nn_cat_{which}_{'dag' if dag else 'T'}.stim"))
