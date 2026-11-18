"""Condition (i) of Theorem (late errors dominate) for a one-check circuit with stabilizer rounds before the check.

Every fault location of the circuit's own noise channels before the check (on a data qubit before its U^dag gate,
or on an ancilla of a gadget that touches the data only before their U^dag gates) is propagated through the
Clifford gates. We record the Pauli it leaves on each data qubit when that qubit reaches its U^dag gate, and the
records of the earlier gadgets that it flips (the input is ideally encoded, so every such record is deterministic
on its own). Condition (i*): for every combination of s <= KMAX faults that flips no early record and leaves Z^b X^a
on the data, some such combination G that leaves only Z^c satisfies |G| + w(Z^(b+c) X^a) <= s, where w is the
weight up to stabilizers. Usage: python early_check.py PATH DELTA
[KMAX]"""
import os, sys, json, itertools
import numpy as np
import stim
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from colorcode import color_code_faces

PAULI1 = {"DEPOLARIZE1": ["X", "Y", "Z"], "X_ERROR": ["X"], "Z_ERROR": ["Z"], "Y_ERROR": ["Y"]}
PAULI2 = [a + b for a in "IXYZ" for b in "IXYZ"][1:]


def main(path, delta, kmax):
    faces, n = color_code_faces(delta)
    ins = list(stim.Circuit.from_file(path).flattened())
    tg = {}
    for k, op in enumerate(ins):
        if op.name in ("S", "S_DAG"):
            for t in op.targets_copy():
                tg.setdefault(t.value, []).append(k)
    data = sorted(q for q in tg if len(tg[q]) >= 2)
    assert len(data) == n, (len(data), n)
    dpos = {q: i for i, q in enumerate(data)}
    t1 = {q: tg[q][-2] for q in data}
    # gadget episodes: an ancilla between a reset and a measurement; early if it touches data only before U^dag
    rec, nrec, episode_start, early_rec, touch, partner = {}, 0, {}, set(), {}, {}
    for k, op in enumerate(ins):
        nm = op.name
        tq = [t.value for t in op.targets_copy()] if nm != "MPP" else []
        if nm in ("R", "RX"):
            for q in tq:
                episode_start[q] = k; touch[q] = []; partner[q] = []
        if nm in ("CX", "CNOT"):
            for c, t in zip(tq[::2], tq[1::2]):
                for a, b in ((c, t), (t, c)):
                    if a not in dpos and b in dpos:
                        touch.setdefault(a, []).append(k < t1[b])
                    elif a not in dpos and b not in dpos:
                        partner.setdefault(a, []).append(b)
        if nm in ("M", "MX", "MPP"):
            cnt = len(op.target_groups()) if nm == "MPP" else len(tq)
            rec[k] = list(range(nrec, nrec + cnt))
            if nm != "MPP":
                for j, q in enumerate(tq):
                    early = touch.get(q) and all(touch[q])
                    if not touch.get(q):                     # a flag: early if it couples to an early ancilla
                        early = any(touch.get(a) and all(touch[a]) for a in partner.get(q, []))
                    if q not in dpos and early:
                        early_rec.add(nrec + j)
            nrec += cnt
        if nm in ("M", "MX"):
            for q in tq:
                partner.pop(q, None)
    erec = {r: i for i, r in enumerate(sorted(early_rec))}

    def propagate(k0, xs, zs):
        """Pauli frame from just after instruction k0; returns (data X mask, data Z mask, early record flips, kind)"""
        x, z = set(xs), set(zs)
        X, Z, flips, hit_early, hit_late = 0, 0, 0, False, False
        for k in range(k0 + 1, len(ins)):
            op = ins[k]
            nm = op.name
            if nm == "MPP":
                break
            tq = [t.value for t in op.targets_copy()]
            if nm in ("CX", "CNOT"):
                for c, t in zip(tq[::2], tq[1::2]):
                    if c in x:
                        x ^= {t}
                    if t in z:
                        z ^= {c}
            elif nm == "H":
                for q in tq:
                    a, b = q in x, q in z
                    if a != b:
                        if a:
                            x.discard(q); z.add(q)
                        else:
                            z.discard(q); x.add(q)
            elif nm in ("S", "S_DAG"):
                for q in tq:
                    if q in dpos and k == t1[q]:
                        if q in x or q in z:
                            hit_early = True
                            X |= (1 << dpos[q]) if q in x else 0
                            Z |= (1 << dpos[q]) if q in z else 0
                        x.discard(q); z.discard(q)          # stop following this data qubit
                    elif q in x:
                        z ^= {q}
            elif nm in ("R", "RX"):
                for q in tq:
                    x.discard(q); z.discard(q)
            elif nm in ("M", "MX"):
                for j, q in enumerate(tq):
                    r = rec[k][j]
                    flipped = (q in x) if nm == "M" else (q in z)
                    if flipped and r in erec:
                        flips ^= 1 << erec[r]
                    elif flipped and r not in erec and q not in dpos:
                        pass
            if any(q in dpos for q in (x | z)) and any(k >= t1[q] for q in (x | z) if q in dpos):
                hit_late = True
        return X, Z, flips, hit_early, hit_late

    pieces = set()
    nloc = 0
    for k, op in enumerate(ins):
        nm = op.name
        if nm not in PAULI1 and nm != "DEPOLARIZE2":
            continue
        tq = [t.value for t in op.targets_copy()]
        groups = [[q] for q in tq] if nm in PAULI1 else [tq[i:i + 2] for i in range(0, len(tq), 2)]
        for g in groups:
            if any(q in dpos and k > t1[q] for q in g):
                continue                                     # data fault after its U^dag: late
            paulis = PAULI1[nm] if nm in PAULI1 else PAULI2
            for p in paulis:
                xs = [q for q, l in zip(g, p) if l in "XY"]
                zs = [q for q, l in zip(g, p) if l in "ZY"]
                X, Z, fl, he, hl = propagate(k, xs, zs)
                if hl and not he:
                    continue                                 # a fault of the check gadget: late
                if (X or Z or fl):
                    nloc += 1
                    pieces.add((X, Z, fl))
    pieces = sorted(pieces)
    F = [sum(1 << q for q in f) for f in faces]
    span = [0]
    for f in F:
        span += [s ^ f for s in span]
    S = np.array(span, dtype=np.uint64)
    POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int16)

    def popv(v):
        out = np.zeros(v.shape, dtype=np.int16)
        for s in range(4):
            out += POP[((v >> np.uint64(16 * s)) & np.uint64(0xFFFF)).astype(np.int64)]
        return out

    def minweight(X, Z):
        xs = S ^ np.uint64(X)
        zs = S ^ np.uint64(Z)
        return int(popv(xs[:, None] | zs[None, :]).min())

    # condition (i*): an accepted early configuration of s faults leaving Z^b X^a is fine if some accepted early
    # configuration G leaving only Z^c on the data has |G| + w(Z^(b+c) X^a) <= s (w = weight up to stabilizers);
    # G = the configuration itself covers every case with a = 0.
    states = {}                                      # accepted states (X, Z) -> least number of faults
    level = {(0, 0, 0)}
    seen = {(0, 0, 0)}
    states[(0, 0)] = 0
    for s in range(1, kmax + 1):
        nxt = set()
        for (X, Z, fl) in level:
            for (x, z, f) in pieces:
                key = (X ^ x, Z ^ z, fl ^ f)
                if key not in seen:
                    nxt.add(key)
        seen |= nxt
        for (X, Z, fl) in nxt:
            if fl == 0 and (X, Z) not in states:
                states[(X, Z)] = s
        level = nxt
        print(json.dumps(dict(faults=s, accepted_states=sum(1 for k in nxt if k[2] == 0))), flush=True)
    zonly = sorted((t, Z) for (X, Z), t in states.items() if X == 0)
    worst, checked = None, 0
    for (X, Z), s in states.items():
        if X == 0 or bin(X | Z).count("1") <= s:
            continue
        checked += 1
        ok = False
        for t, c in zonly:
            if t + bin(X).count("1") > s:
                break
            if t + minweight(X, Z ^ c) <= s:
                ok = True
                break
        if not ok:
            w0 = minweight(X, Z)
            if worst is None or w0 - s > worst[0]:
                worst = (w0 - s, s, [q for q in range(n) if (X >> q) & 1], [q for q in range(n) if (Z >> q) & 1])
    print(json.dumps(dict(path=os.path.basename(path), pieces=len(pieces), early_records=len(erec),
                          accepted_states=len(states), zonly_states=len(zonly), checked=checked,
                          violation=worst, condition_i_star=worst is None)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else int(sys.argv[2]) - 1)
