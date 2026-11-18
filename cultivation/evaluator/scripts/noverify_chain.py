"""Computer-assisted bound for un-verified path cats grown from one end on the [[19,1,5]] code.

A single X fault on path site i, before it grows site i+1, spreads X to the suffix of the path from i, which is
then coupled to its data qubits: a late pattern A = the set of data qubits of the suffix, of cost one (equivalently
its complement, since X^n is the measured operator). A placement keeps fault distance delta only if every suffix
S (1 <= |S| <= n-1) has completion cost comp(S) >= delta - 1 under the late-error theorem. We decide by dynamic
programming over subsets whether a chain {} < S_1 < ... < S_n of such sets exists, that is, whether any order
of the data along the path passes even the single-fault test. Usage: python noverify_chain.py DELTA"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces
from late_bound import CodeTools

delta = int(sys.argv[1])
faces, n = color_code_faces(delta)
ct = CodeTools(faces, n)
allq = (1 << n) - 1
need = delta - 1
ok = lambda S: min(ct.comp(S, need), ct.comp(S ^ allq, need)) >= need
level = {1 << q for q in range(n) if ok(1 << q)}
print(json.dumps(dict(size=1, sets=len(level))), flush=True)
for m in range(2, n):
    nxt = set()
    for S in level:
        for q in range(n):
            if not (S >> q) & 1:
                T = S | (1 << q)
                if T not in nxt and ok(T):
                    nxt.add(T)
    level = nxt
    print(json.dumps(dict(size=m, sets=len(level))), flush=True)
    if not level:
        break
print(json.dumps(dict(delta=delta, order_exists=bool(level))), flush=True)
