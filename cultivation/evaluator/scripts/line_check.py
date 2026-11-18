"""Nearest-neighbour line checks whose cat is verified delta - 1 times (the construction of Corollary 24).

Layout on a square grid: data qubit q at (q, -1), its cat site at (q, 0), ancillas at (q, 1). The cat is grown from
the middle of the path (q, 0), q = 0..n-1, and verified by the n - 1 adjacent parities Z_(x,0) Z_(x+1,0), each
through the two ancillas (x, 1), (x+1, 1) (even x, then odd x), in ver_rounds rounds. The data are coupled once,
the cat is read out by X measurements, and the check sits between an ideally encoded input and the ideal decoder
(src/nn_cat.py). With ver_rounds = delta - 1 the gadget is that of Lemma 18, so Theorem 6 gives real-T fault
distance delta at every delta; this script certifies it at small delta with the exact evaluator.
Usage: python line_check.py BASE_WIDTH VER_ROUNDS KMAX [proxy]   -> results/line_d<delta>_v<rounds>.json"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from tdem import TDEM
from fd_fast import fault_distance_fast
from branch import FastExact
from fold import merge_final_mpps, regenerate_detectors
import nn_cat

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def line(n):
    pairs = [(x, x + 1) for x in range(0, n - 1, 2)] + [(x, x + 1) for x in range(1, n - 1, 2)]
    return dict(path=[(x, 0) for x in range(n)], data_sites=[(x, -1) for x in range(n)], cat_of=list(range(n)),
                root=n // 2, ver_pairs=pairs, ver_sites=[((x, 1), (x + 1, 1)) for x, _ in pairs])


if __name__ == "__main__":
    bw, vr, kmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    proxy = len(sys.argv) > 4 and sys.argv[4] == "proxy"
    faces, n = color_code_faces(bw)
    for dag in (False, True):
        t0 = time.time()
        c, G = nn_cat.make_nn_cat(faces, n, inject_dag=dag, rounds=1, ver_rounds=vr, **line(n))
        c = regenerate_detectors(merge_final_mpps(c))
        m = TDEM(c, magic="T", pattern=False)
        fe = FastExact(m)
        if len(fe.A) != 2:                       # the other injection: the check projects instead of detecting
            continue
        mech = m.mechanisms()
        k, f = fault_distance_fast(fe, kmax, mech, verbose=False)
        row = dict(delta=bw, n=n, ver_rounds=vr, inject_dag=dag, qubits=c.num_qubits, ticks=c.num_ticks, t=m.t,
                   classes=len(mech), fd_real=k if k else f">{kmax}", malignant=len(f),
                   seconds_real=round(time.time() - t0))
        if proxy:
            t1 = time.time()
            mp = TDEM(c, magic="S", pattern=False)
            kp, fp = fault_distance_fast(mp, kmax, mp.mechanisms(), verbose=False)
            row.update(fd_proxy=kp if kp else f">{kmax}", seconds_proxy=round(time.time() - t1))
        print(json.dumps(row), flush=True)
        c.to_file(os.path.join(R, f"line_d{bw}_v{vr}.stim"))
        json.dump(row, open(os.path.join(R, f"line_d{bw}_v{vr}.json"), "w"), indent=1)
