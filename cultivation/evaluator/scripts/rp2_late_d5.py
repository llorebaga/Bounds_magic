"""Upper bound on the real-T fault distance of the distance-five RP^2 cultivation circuit of Chen et al.
(../external/rp2_circuits/rp_3_rp_5_T_cult.stim) from harmful late configurations of its last check.

A late configuration has all its faults inside the last check (between its two layers of magic gates) or on the data
just after its last layer. The circuit before the check is then fault free, so the check acts on the ideally encoded
state of the self-dual [[35,1,5]] code of the check, and everything after the last layer (the morph back to the RP^2
code and the final ideal decoder) is an ideal decoder of that code: its measurements are deterministic on the
fault-free output, a magic state, so they measure stabilizers. The late-error theorem then decides harm by Pauli
algebra (late_ilp.exact_harm), and a harmful late configuration of w faults is harmful in the whole circuit.

The script cuts the circuit after the last layer, reads the code from the noiseless proxy state there, checks the
hypotheses (CSS, self-dual, X^n logical, Z stabilizers of sign +1 and weights giving flat phases), extracts the late
pieces of every single fault (late_ilp.pieces) and searches for harmful configurations of at most KMAX faults.
LIMITATION: the pieces start at the first layer of the check, so Z-only faults before the check (which the late-error
theorem allows) are not included. Its ">3" result is therefore NOT a bound: the exact evaluation (rp2_fd_lean.py) finds
harmful triples whose first fault is a Z fault during the growth. Kept for the record of the signed flat-phase check.
Usage: python rp2_late_d5.py [KMAX]"""
import sys, os, itertools, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import stim
from late_ilp import pieces, exact_harm
from gf2 import rref

KMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 3
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "rp2_circuits", "rp_3_rp_5_T_cult.stim")
c = stim.Circuit.from_file(path)
ins = list(c.flattened())
last = max(k for k, op in enumerate(ins) if op.name in ("S", "S_DAG"))
data = sorted(t.value for t in ins[last].targets_copy())
n = len(data)
pos = {q: i for i, q in enumerate(data)}
print(f"last layer at instruction {last}, {n} data qubits", flush=True)

# truncated circuit: everything up to and including the last layer, without detectors on later records
cut = stim.Circuit()
for op in ins[:last + 1]:
    cut.append(op)
# the code of the check: stabilizers of the noiseless proxy state after the last layer, restricted to the data
sim = stim.TableauSimulator()
for op in cut.without_noise():
    if op.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"):
        continue
    sim.do(op)
stabs = sim.canonical_stabilizers()
on_data = []
for P in stabs:
    sup = [q for q in range(len(P)) if P[q]]
    if sup and all(q in pos for q in sup):
        on_data.append(P)
assert len(on_data) == n, "data not in a pure state"
from gf2 import nullspace
G = np.array([np.concatenate([np.array([P[q] in (1, 2) for q in data]), np.array([P[q] in (2, 3) for q in data])])
              for P in on_data], dtype=np.uint8)


def subgroup(cols):
    """products of generators whose part in cols vanishes (pure-Z: X part zero; pure-X: Z part zero)"""
    N = nullspace(G[:, cols].T)                       # combinations c with c . G[:, cols] = 0
    out = []
    for row in N:
        Q = stim.PauliString(len(on_data[0]))
        for i in np.nonzero(row)[0]:
            Q *= on_data[i]
        out.append(Q)
    return out


Z = subgroup(list(range(n)))
X = subgroup(list(range(n, 2 * n)))
print(f"stabilizers on the data: {len(on_data)}; pure-X subgroup rank {len(X)}, pure-Z subgroup rank {len(Z)}", flush=True)
assert len(X) + len(Z) == n - 1, "not a CSS code with one logical qubit"
mask = lambda P: sum(1 << pos[q] for q in data if P[q])
Xm, Zm = [mask(P) for P in X], [mask(P) for P in Z]
pc = lambda v: bin(v).count("1")


def span_rank(ms):
    M = np.array([[(m >> j) & 1 for j in range(n)] for m in ms], dtype=np.uint8)
    R, piv = rref(M)
    return len(piv)


selfdual = span_rank(Xm) == span_rank(Zm) == span_rank(Xm + Zm)
signs_ok = all(P.sign == 1 for P in Z)
allones = (1 << n) - 1
xn_logical = all(pc(z & allones) % 2 == 0 for z in Zm) and span_rank(Xm + [allones]) > span_rank(Xm)
# flat phases with signs: U X^n U^dag maps the code to itself iff phi(k) = s_k for every Z stabilizer k of sign s_k,
# where phi(k) = prod_{q in k} (-i sgn theta_q) for the last layer U (theta = +pi/4 for S read as T, -pi/4 for S_DAG).
# On generators: phi(k) = s_k, and phi(a+b) = phi(a) phi(b) (-1)^{|a & b|} = s_a s_b requires even overlaps.
u = -1j if ins[last].name == "S" else 1j
flat = all(abs(u ** pc(m) - P.sign) < 1e-12 for m, P in zip(Zm, Z)) and \
    all(pc(a & b) % 2 == 0 for a in Zm for b in Zm)
print(json.dumps(dict(self_dual=selfdual, z_signs_plus=signs_ok, Xn_logical_X=xn_logical, flat_phases_with_signs=flat,
                      last_layer=ins[last].name, z_weights=sorted(pc(z) for z in Zm),
                      z_signs=[int(P.sign.real) for P in Z])), flush=True)
assert selfdual and xn_logical and flat
faces = [[j for j in range(n) if (z >> j) & 1] for z in Zm]

# late pieces of every single fault inside the last check, on the full circuit: X, Z on the data are read just after
# the last layer, detector flips are followed to the end. Detectors that read a qubit which, after the last layer, is
# a data qubit or has touched one (the morph back to RP^2 and the final MPP) are stabilizer measurements of the ideal
# decoder; exact_harm imposes them. Every other detector is a gadget detector and must not fire.
from late_ilp import _records, TWO_Q, MEAS_Z, MEAS_X
rec_of = _records(ins)
touched = set(data)
meas_after = set()
for k in range(last + 1, len(ins)):
    op = ins[k]
    tg = [x.value for x in op.targets_copy() if not x.is_measurement_record_target and not x.is_combiner]
    if op.name in TWO_Q:
        for a, b in zip(tg[::2], tg[1::2]):
            if a in touched or b in touched:
                touched |= {a, b}
    if k in rec_of and (op.name == "MPP" or any(q in touched for q in tg)):
        meas_after |= set(rec_of[k]) if op.name == "MPP" else {r for r, q in zip(rec_of[k], tg) if q in touched}
dets, nr = [], 0
for op in ins:
    if op.name in MEAS_Z + MEAS_X + ("MY",):
        nr += len(op.targets_copy())
    elif op.name == "MPP":
        nr += len(op.target_groups())
    elif op.name == "DETECTOR":
        dets.append(frozenset(nr + x.value for x in op.targets_copy()))
gadget = [i for i, D in enumerate(dets) if not (D & meas_after)]
det_of_rec = {}
for i in gadget:
    for r in dets[i]:
        det_of_rec.setdefault(r, set()).add(i)
print(f"{len(dets)} detectors, {len(gadget)} gadget detectors kept", flush=True)
tg_of = {}
for k, op in enumerate(ins):
    if op.name in ("S", "S_DAG"):
        for x in op.targets_copy():
            tg_of.setdefault(x.value, []).append(k)
t1 = {q: tg_of[q][-2] for q in data}


def propagate(k0, xs, zs):
    x, z, flips, A, snap = set(xs), set(zs), set(), 0, None
    if k0 > last:
        snap = (sum(1 << pos[q] for q in x if q in pos), sum(1 << pos[q] for q in z if q in pos))
    for k in range(k0, len(ins)):
        op = ins[k]
        nm = op.name
        tg = [q.value for q in op.targets_copy()]
        if nm in ("CX", "CNOT"):
            for a, b in zip(tg[::2], tg[1::2]):
                if a in x: x ^= {b}
                if b in z: z ^= {a}
        elif nm == "CZ":
            for a, b in zip(tg[::2], tg[1::2]):
                if a in x: z ^= {b}
                if b in x: z ^= {a}
        elif nm == "H":
            for q in tg:
                xq, zq = q in x, q in z
                x.discard(q); z.discard(q)
                if zq: x.add(q)
                if xq: z.add(q)
        elif nm in ("S", "S_DAG"):
            for q in tg:
                if q in pos and k == t1[q] and q in x:
                    return None
                if q in pos and k == last and q in x:
                    A |= 1 << pos[q]
        elif nm in MEAS_Z:
            for j, q in enumerate(tg):
                if q in x: flips ^= det_of_rec.get(rec_of[k][j], set())
            if nm == "MR": x -= set(tg); z -= set(tg)
        elif nm in MEAS_X:
            for j, q in enumerate(tg):
                if q in z: flips ^= det_of_rec.get(rec_of[k][j], set())
            if nm == "MRX": x -= set(tg); z -= set(tg)
        elif nm == "MPP":
            for j, grp in enumerate(op.target_groups()):
                anti = 0
                for g in grp:
                    q = g.value
                    if (g.is_x_target or g.is_y_target) and q in z: anti ^= 1
                    if (g.is_z_target or g.is_y_target) and q in x: anti ^= 1
                if anti: flips ^= det_of_rec.get(rec_of[k][j], set())
        elif nm in ("R", "RX"):
            x -= set(tg); z -= set(tg)
        if k == last:
            snap = (sum(1 << pos[q] for q in x if q in pos), sum(1 << pos[q] for q in z if q in pos))
    fx, fz = snap
    return A, fx ^ A, fz, frozenset(flips)


start = min(t1.values()) + 1
P = set()
t0 = time.time()
for k in range(start, last + 1):
    op = ins[k]
    nm = op.name
    if nm in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS") or             nm.startswith(("DEPOLARIZE", "X_ERROR", "Z_ERROR", "Y_ERROR", "PAULI")):
        continue
    tg = [q.value for q in op.targets_copy() if not q.is_measurement_record_target]
    groups = [tg[i:i + 2] for i in range(0, len(tg), 2)] if nm in TWO_Q else [[q] for q in tg]
    for g in groups:
        if k == last:
            continue                                    # faults after the last layer are the data faults
        for paulis in itertools.product("IXYZ", repeat=len(g)):
            if all(p == "I" for p in paulis):
                continue
            xs = [q for q, p in zip(g, paulis) if p in "XY"]
            zs = [q for q, p in zip(g, paulis) if p in "ZY"]
            if any(q in pos and k < t1[q] for q in xs):
                continue
            r = propagate(k + 1, xs, zs)
            if r is not None and any(r[:3]) | bool(r[3]):
                P.add(r)
    if k in rec_of and nm in MEAS_Z + MEAS_X:
        for r in rec_of[k]:
            fl = frozenset(det_of_rec.get(r, set()))
            if fl:
                P.add((0, 0, 0, fl))
P = sorted(P, key=lambda p: (len(p[3]), p[0], p[1], p[2]))
print(f"{len(P)} distinct late pieces ({time.time() - t0:.0f}s)", flush=True)
byD = {}
for A, Ap, B, D in P:
    byD.setdefault(D, []).append((A, Ap, B))
single = byD.get(frozenset(), [])
print(f"undetected single pieces: {len(single)}", flush=True)


def data_faults(w):
    for qs in itertools.combinations(range(n), w):
        for ps in itertools.product((1, 2, 3), repeat=w):
            yield (sum(1 << q for q, p in zip(qs, ps) if p & 1), sum(1 << q for q, p in zip(qs, ps) if p & 2))


def harmful_with(A, Ap, B, budget):
    for w in range(budget + 1):
        for x, z in data_faults(w):
            if exact_harm(A, Ap ^ x, B ^ z, faces, n):
                return w, x, z
    return None


best = None
# one piece, or two pieces with cancelling detectors, completed by data faults just after the last layer
combos = [((a,), 1) for a in single]
combos += [((a, b), 2) for D, lst in byD.items() for a, b in itertools.combinations(lst, 2)]
combos += [((a, b), 2) for a, b in itertools.combinations(single, 2)]
triples = set()
if KMAX >= 3:                                         # three pieces with cancelling detectors, no data fault
    Dlist = list(byD.items())
    for i, (D1, l1) in enumerate(Dlist):
        for D2, l2 in Dlist[i:]:
            D3 = D1 ^ D2
            if D3 not in byD:
                continue
            for a in l1:
                for b in l2:
                    ab = (a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2])
                    for c3 in byD[D3]:
                        triples.add((ab[0] ^ c3[0], ab[1] ^ c3[1], ab[2] ^ c3[2]))
    print(f"{len(triples)} distinct patterns from three pieces", flush=True)
for parts, s in sorted(combos, key=lambda x: x[1]):
    if s > KMAX:
        continue
    A = Ap = B = 0
    for a, ap, b in parts:
        A ^= a; Ap ^= ap; B ^= b
    r = harmful_with(A, Ap, B, min(KMAX - s, (best[0] - 1 - s) if best else KMAX - s))
    if r is not None:
        w, x, z = r
        if best is None or s + w < best[0]:
            best = (s + w, dict(pieces=s, data_faults=w, A=A, Ap=Ap ^ x, B=B ^ z,
                                X_before_layer=[data[j] for j in range(n) if A >> j & 1]))
            print(f"  harmful late configuration of {s + w} faults: {best[1]}", flush=True)
if best is None or best[0] > 3:
    for i, (A, Ap, B) in enumerate(sorted(triples)):
        if exact_harm(A, Ap, B, faces, n):
            best = (3, dict(pieces=3, data_faults=0, A=A, Ap=Ap, B=B, X_before_layer=[data[j] for j in range(n) if A >> j & 1]))
            print(f"  harmful late configuration of 3 faults: {best[1]}", flush=True)
            break
        if i % 200000 == 0:
            print(f"  {i} of {len(triples)} triple patterns tested ({time.time() - t0:.0f}s)", flush=True)
print("RESULT", json.dumps(dict(upper_bound=best[0] if best else f">{KMAX} among late configurations",
                               witness=best[1] if best else None, seconds=round(time.time() - t0))), flush=True)
