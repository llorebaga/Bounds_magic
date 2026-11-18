"""The late-error integer program also bounds the S proxy: with S gates a late error X^A crossing U keeps the single
family member w = A of the program, and an X^a Z^b before U^dag acts like X^a Z^{a+b} between the layers, on the same
qubits. So a run of late_ilp.certify without cuts that finds no solution below K certifies proxy fault distance >= K.
Test: on random placements of the end-parity check, the value of the program (when no cut was needed) must not exceed
the exact proxy fault distance from the detector error model (proxy_ilp.py).
Usage: python late_ilp_proxy_test.py DELTA N SEED"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from late_ilp import certify
from late_ilp_test import build
from proxy_ilp import mechanisms, min_weight

delta, N, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
faces, n = color_code_faces(delta)
rng = random.Random(seed)
ok = tested = 0
for k in range(N):
    order = list(range(n))
    if k:
        rng.shuffle(order)
    pos = {q: i for i, q in enumerate(order)}
    lf = [[pos[q] for q in f] for f in faces]
    for dag in (False, True):
        c = build("endp", delta, order, dag)
        r = certify(c, lf, n, delta)
        w, st = min_weight(mechanisms(c), 0, delta + 1, "gurobi", 600, 0)
        row = dict(k=k, inject_dag=dag, late=r["bound"], cuts=r["cuts"], proxy=w, status=st)
        if r["cuts"] == 0:
            tested += 1
            ok += isinstance(w, int) and r["bound"] <= w
        print(json.dumps(row), flush=True)
print(json.dumps(dict(delta=delta, tested=tested, consistent=ok)), flush=True)
