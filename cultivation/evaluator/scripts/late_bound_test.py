"""Compare the late-error bound with exact certificates: un-verified path cats (noverify_probe.py) and the
end-parity check (endparity_predict.py). Usage: python late_bound_test.py KIND DELTA N SEED [CERT]"""
import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from late_bound import bound, CodeTools
from nn_cat_fd import certify
import noverify_probe, endparity_predict

kind, delta, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
cert = len(sys.argv) <= 5 or sys.argv[5] != "0"
faces, n = color_code_faces(delta)
ct = CodeTools(faces, n)
rng = random.Random(seed)
for k in range(N):
    order = list(range(n))
    if k:
        rng.shuffle(order)
    pos = {q: i for i, q in enumerate(order)}
    lf = [[pos[q] for q in f] for f in faces]          # the circuit's data qubit i is code qubit order[i]
    ctl = CodeTools(lf, n)
    builder = noverify_probe.build if kind == "nover" else endparity_predict.build
    t0 = time.time()
    c, root = builder(delta, order, False)
    b, wit, npieces = bound(c, lf, n, delta, ct=ctl)
    out = dict(k=k, late_bound=b, witness_cost=wit[0] if wit else None, pieces=npieces, tb=round(time.time() - t0, 1))
    if kind == "endp":
        out["family_pred"] = endparity_predict.predict(order, faces, n, root, delta)
    if cert:
        for dag in (False, True):
            c, root = builder(delta, order, dag)
            r = certify(c, delta - 1, proxy=False)["T"]
            if r["fd"] != 1:
                break
        out["certified"] = r["fd"] if isinstance(r["fd"], int) else delta
    print(json.dumps(out), flush=True)
