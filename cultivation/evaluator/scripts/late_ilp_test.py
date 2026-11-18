"""Validate the integer-programming certificate (late_ilp.py) against exact real-T certificates (nn_cat_fd.certify).
Usage: python late_ilp_test.py KIND DELTA N SEED [CERT]
KIND: endp (un-grown path cat with end parity only), ladder (rungs and diagonals, un-grown) or vcat (all-to-all
verified cat without stabilizer rounds)."""
import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import nn_cat
from colorcode import color_code_faces
from fold import merge_final_mpps, regenerate_detectors
from late_ilp import certify as ilp
from nn_cat_fd import certify as exact
import endparity_predict
from vcat import make_vcat

LADDER = {5: 8, 7: 17, 9: 29, 11: 44}


def build(kind, delta, order, dag):
    if kind == "endp":
        return endparity_predict.build(delta, order, dag)[0]
    faces, n = color_code_faces(delta)
    pos = {q: i for i, q in enumerate(order)}
    lf = [[pos[q] for q in f] for f in faces]
    if kind == "vcat":                              # all-to-all verified cat, one check, no stabilizer rounds
        return regenerate_detectors(merge_final_mpps(make_vcat(lf, n, rounds=1, inject_dag=dag, stab_rounds=False)))
    c, G = nn_cat.make_nn_cat(lf, n, unfold=True, rounds=1, inject_dag=dag, **nn_cat.ladder(LADDER[delta]))
    return regenerate_detectors(merge_final_mpps(c))


if __name__ == "__main__":
    kind, delta, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    cert = len(sys.argv) <= 5 or sys.argv[5] != "0"
    faces, n = color_code_faces(delta)
    rng = random.Random(seed)
    for k in range(N):
        order = list(range(n))
        if k:
            rng.shuffle(order)
        pos = {q: i for i, q in enumerate(order)}
        lf = [[pos[q] for q in f] for f in faces]
        dag = False
        for dag in (False, True):                   # the injected state whose check is logical
            c = build(kind, delta, order, dag)
            if exact(c, 1, proxy=False)["T"]["fd"] != 1:
                break
        t0 = time.time()
        r = ilp(c, lf, n, delta)
        out = dict(k=k, inject_dag=dag, qubits=c.num_qubits, ilp=r["bound"], X=r.get("X"), Z=r.get("Z"), C=r.get("C"), cuts=r["cuts"],
                   pieces=r["pieces"], t_ilp=round(time.time() - t0, 1))
        if cert:
            e = exact(c, delta - 1, proxy=False)["T"]["fd"]
            out["certified"] = e if isinstance(e, int) else delta
            out["match"] = out["certified"] == out["ilp"]
        print(json.dumps(out), flush=True)
