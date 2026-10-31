"""Full nearest-neighbour cultivation protocols: noisy injection with a path-relayed encoder, Z-stabiliser rounds
through the path, and verified-cat checks (src/nn_cat.py). Certifies the real-T and proxy fault distance.
Usage: python nn_full.py d3|d5 ROUNDS ZROUNDS ZBETWEEN VER [KMAX]"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import stim
import nn_cat
from noise import si1000
from fold import merge_final_mpps, regenerate_detectors
from nn_cat_fd import certify

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def build(which, rounds, zr, zb, vr, dag, zkind="shor", inject="routed", unfold=False):
    if which == "d3":
        faces, n, L = nn_cat.STEANE, 7, nn_cat.RING3
    else:
        from colorcode import color_code_faces
        faces, n = color_code_faces(int(which[1:]))
        L = nn_cat.ladder({5: 8, 7: 17}[int(which[1:])])
    c, G = nn_cat.make_nn_cat(faces, n, inject_dag=dag, rounds=rounds, ver_rounds=vr, inject=inject,
                              zrounds=zr, zbetween=zb, zkind=zkind, unfold=unfold, **L)
    return regenerate_detectors(merge_final_mpps(c))


if __name__ == "__main__":
    which, rounds, zr, zb, vr = sys.argv[1], *map(int, sys.argv[2:6])
    kmax = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    zkind = sys.argv[7] if len(sys.argv) > 7 else "shor"
    inject = sys.argv[8] if len(sys.argv) > 8 else "routed"
    unfold = len(sys.argv) > 9 and sys.argv[9] == "unfold"
    name = f"nn_full_{which}_r{rounds}_z{zr}_b{zb}_v{vr}" + ("" if zkind == "shor" else "_" + zkind) +         ("" if inject == "routed" else "_" + inject) + ("_unf" if unfold else "")
    for dag in (False, True):
        c = build(which, rounds, zr, zb, vr, dag, zkind, inject, unfold)
        try:
            r = certify(c, kmax, proxy=os.environ.get("PROXY") == "1")
            r["T_si1000"] = certify(si1000(c, 1e-3), min(kmax, 3), proxy=False)["T"] if kmax <= 3 else None
        except Exception as e:
            print(name, "inject_dag", dag, "ERR", e, flush=True); continue
        c.to_file(os.path.join(R, name + ".stim"))
        print(json.dumps(dict(name=name, inject_dag=dag, qubits=c.num_qubits, ticks=c.num_ticks, **r)), flush=True)
