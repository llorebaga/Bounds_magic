"""Gurobi (late_ilp.certify) against HiGHS (late_ilp_highs.certify) on random placements of the end-parity check,
which include placements below the full distance, so that both the bounds and the witnesses are compared.
Usage: python late_ilp_highs_test.py DELTA N SEED"""
import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from late_ilp import certify as grb, exact_harm
from late_ilp_highs import certify as hig
from late_ilp_test import build

delta, N, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
faces, n = color_code_faces(delta)
rng = random.Random(seed)
agree = 0
for k in range(N):
    order = list(range(n))
    if k:
        rng.shuffle(order)
    pos = {q: i for i, q in enumerate(order)}
    lf = [[pos[q] for q in f] for f in faces]
    rows = []
    for dag in (False, True):
        c = build("endp", delta, order, dag)
        t0 = time.time(); g = grb(c, lf, n, delta); t1 = time.time(); h = hig(c, lf, n, delta); t2 = time.time()
        if h["witness"]:
            W = h["witness"]
            assert exact_harm(W["A"], W["Ap"], W["B"], lf, n)
        rows.append(dict(inject_dag=dag, gurobi=g["bound"], highs=h["bound"], t_grb=round(t1 - t0, 1), t_highs=round(t2 - t1, 1)))
    ok = all(r["gurobi"] == r["highs"] for r in rows)
    agree += ok
    print(json.dumps(dict(k=k, rows=rows, agree=ok)), flush=True)
print(json.dumps(dict(delta=delta, placements=N, agree=agree)), flush=True)
