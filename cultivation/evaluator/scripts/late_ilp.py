"""Integer-programming certificate of the real-T fault distance of a one-round cultivation check that starts from
an ideally encoded state (late-error theorem and its dominance theorem in the paper).

By the dominance theorem, the real-T fault distance of such a check is min(delta, least weight of a harmful late
configuration). A late configuration consists of faults between the U^dag and U layers of the check, which may put
X on the data before U (pattern A, which then crosses U), X after U (pattern A') and Z on the data (pattern B), and
of single-qubit data faults immediately after U (X pattern x, Z pattern z, a Y fault counting once). By the
late-error theorem it is harmful if the gadget detectors cancel and
  (mu = 1)  A + A' + x lies in l_X + S_X, or
  (mu = 0)  A + A' + x lies in S_X and B + z + w lies in l_Z + S_Z for some w inside A.
The exact theorem adds a parity condition on w and a sign condition when A contains a Z logical support. Dropping
them can only make more configurations count as harmful, so the minimum below is a lower bound on the least weight
of a harmful late configuration, and it equals it when the minimizer satisfies them. If no configuration of fewer
than delta faults satisfies the relaxed conditions, the check has fault distance delta.

Pieces are the effects of every single fault of the check, a Pauli operator of any kind on the support of an
operation or a flip of a measurement, propagated through the Clifford gates of the check (S and S_DAG on the data
mark the U^dag and U layers). The code is a triangular color code, so X_L = X^n and Z_L = Z^n.

Usage (as a module): certify(circuit, faces, n, delta) -> dict"""
import itertools
import gurobipy as gp
from gurobipy import GRB

TWO_Q = ("CX", "CNOT", "CZ", "SWAP")
MEAS_Z = ("M", "MR")
MEAS_X = ("MX", "MRX")
RESETS = ("R", "RX", "MR", "MRX")


def _records(ins):
    rec_of, nrec = {}, 0
    for k, op in enumerate(ins):
        if op.name in MEAS_Z + MEAS_X + ("MY", "MPP"):
            ntg = len(op.target_groups()) if op.name == "MPP" else len(op.targets_copy())
            rec_of[k] = list(range(nrec, nrec + ntg))
            nrec += ntg
    return rec_of


def pieces(circuit):
    """dict (A, A', B, D) -> 1 for every single late fault; A, A', B bitmasks over data positions, D a frozenset of
    gadget detectors (detectors that do not involve the final MPP readout)"""
    ins = list(circuit.flattened())
    tg_of = {}
    for k, op in enumerate(ins):
        if op.name in ("S", "S_DAG"):
            for t in op.targets_copy():
                tg_of.setdefault(t.value, []).append(k)
    data = sorted(q for q in tg_of if len(tg_of[q]) >= 2)
    dpos = {q: i for i, q in enumerate(data)}
    t1 = {q: tg_of[q][-2] for q in data}
    t2 = {q: tg_of[q][-1] for q in data}
    end = next(k for k, op in enumerate(ins) if op.name == "MPP")
    start, stop = min(t1.values()) + 1, max(t2.values())
    rec_of = _records(ins)
    mpp_recs = {r for k in rec_of if ins[k].name == "MPP" for r in rec_of[k]}
    dets, nr = [], 0
    for op in ins:
        if op.name in MEAS_Z + MEAS_X + ("MY",):
            nr += len(op.targets_copy())
        elif op.name == "MPP":
            nr += len(op.target_groups())
        elif op.name == "DETECTOR":
            dets.append(frozenset(nr + t.value for t in op.targets_copy()))
    det_of_rec = {}
    for i, D in enumerate(dets):
        if not (D & mpp_recs):
            for r in D:
                det_of_rec.setdefault(r, set()).add(i)

    def propagate(k0, xs, zs):
        x, z, flips, A = set(xs), set(zs), set(), 0
        for k in range(k0, end):
            op = ins[k]
            nm = op.name
            tg = [t.value for t in op.targets_copy()]
            if nm in ("CX", "CNOT"):
                for c, t in zip(tg[::2], tg[1::2]):
                    if c in x:
                        x ^= {t}
                    if t in z:
                        z ^= {c}
            elif nm == "CZ":
                for a, b in zip(tg[::2], tg[1::2]):
                    if a in x:
                        z ^= {b}
                    if b in x:
                        z ^= {a}
            elif nm == "H":
                for q in tg:
                    xq, zq = q in x, q in z
                    x.discard(q); z.discard(q)
                    if zq:
                        x.add(q)
                    if xq:
                        z.add(q)
            elif nm in ("S", "S_DAG"):
                for q in tg:
                    if q in dpos:
                        if k <= t1[q] and q in x:
                            return None                 # X before U^dag: an early error, not a late piece
                        if k == t2[q] and q in x:
                            A |= 1 << dpos[q]          # X crossing the U layer (its Z part is free)
                    elif q in x:
                        z ^= {q}                        # S on a non-data qubit (not expected here)
            elif nm in MEAS_Z:
                for j, q in enumerate(tg):
                    if q in x:
                        flips ^= det_of_rec.get(rec_of[k][j], set())
                if nm == "MR":
                    x -= set(tg); z -= set(tg)
            elif nm in MEAS_X:
                for j, q in enumerate(tg):
                    if q in z:
                        flips ^= det_of_rec.get(rec_of[k][j], set())
                if nm == "MRX":
                    x -= set(tg); z -= set(tg)
            elif nm in ("R", "RX"):
                x -= set(tg); z -= set(tg)
        fin_x = sum(1 << dpos[q] for q in x if q in dpos)
        fin_z = sum(1 << dpos[q] for q in z if q in dpos)
        return A, fin_x ^ A, fin_z, frozenset(flips)

    out = set()
    for k in range(start, end):
        op = ins[k]
        nm = op.name
        if nm in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS") or \
                nm.startswith(("DEPOLARIZE", "X_ERROR", "Z_ERROR", "Y_ERROR", "PAULI")):
            continue
        tg = [t.value for t in op.targets_copy()]
        groups = [tg[i:i + 2] for i in range(0, len(tg), 2)] if nm in TWO_Q else [[q] for q in tg]
        for g in groups:
            if nm in ("S", "S_DAG") and any(q in dpos and k == t2[q] for q in g):
                continue                                # faults after U are the data faults x, z of the ILP
            for paulis in itertools.product("IXYZ", repeat=len(g)):
                if all(p == "I" for p in paulis):
                    continue
                xs = [q for q, p in zip(g, paulis) if p in "XY"]
                zs = [q for q, p in zip(g, paulis) if p in "ZY"]
                if any(q in dpos and k < t1[q] for q in xs):
                    continue                            # an X error on a data qubit before its U^dag
                r = propagate(k + 1, xs, zs)
                if r is not None and (r[0] or r[1] or r[2] or r[3]):
                    out.add(r)
        if k in rec_of and nm in MEAS_Z + MEAS_X:
            for r in rec_of[k]:
                fl = frozenset(det_of_rec.get(r, set()))
                if fl:
                    out.add((0, 0, 0, fl))
    return sorted(out, key=lambda p: (len(p[3]), p[0], p[1], p[2])), data


def _nullspace_in(faces, A, n):
    """basis (bitmasks) of the Z operators supported inside A that commute with every X face"""
    cols = [q for q in range(n) if A >> q & 1]
    rows = [sum(1 << j for j, q in enumerate(cols) if q in f) for f in faces]
    # Gaussian elimination on the face rows over the columns of A
    piv, red = [], []
    for r in rows:
        for pr, pc in zip(red, piv):
            if r >> pc & 1:
                r ^= pr
        if r:
            pc = r.bit_length() - 1
            for i in range(len(red)):
                if red[i] >> pc & 1:
                    red[i] ^= r
            red.append(r); piv.append(pc)
    free = [j for j in range(len(cols)) if j not in piv]
    basis = []
    for fj in free:
        v = 1 << fj
        for pr, pc in zip(red, piv):
            if pr >> fj & 1:
                v |= 1 << pc
        basis.append(sum(1 << cols[j] for j in range(len(cols)) if v >> j & 1))
    return basis


def _solve(rows, rhs, cols):
    """one solution (bitmask over the columns cols) of the GF(2) system rows . w = rhs, or None"""
    m = [(sum(1 << j for j, q in enumerate(cols) if r >> q & 1), b) for r, b in zip(rows, rhs)]
    piv = []
    sol_rows = []
    for row, b in m:
        for (pr, pb), pc in zip(sol_rows, piv):
            if row >> pc & 1:
                row ^= pr; b ^= pb
        if row:
            pc = row.bit_length() - 1
            for i in range(len(sol_rows)):
                if sol_rows[i][0] >> pc & 1:
                    sol_rows[i] = (sol_rows[i][0] ^ row, sol_rows[i][1] ^ b)
            sol_rows.append((row, b)); piv.append(pc)
        elif b:
            return None
    w = 0
    for (pr, pb), pc in zip(sol_rows, piv):
        if pb:
            w |= 1 << cols[pc]
    return w


def exact_harm(A, Ap, B, faces, n):
    """late-error theorem: is the late configuration with X^A crossing U, X^Ap after it and Z^B harmful?"""
    pc = lambda v: bin(v).count("1")
    Fm = [sum(1 << q for q in f) for f in faces]
    T = A ^ Ap
    if any(pc(T & f) & 1 for f in Fm):
        return False                                   # anticommutes with a Z face: rejected
    mu = pc(T) & 1
    K = _nullspace_in(faces, A, n)                     # Z stabilizers and Z logicals inside A
    K0 = []                                            # basis of the even (stabilizer) part
    odd = [k for k in K if pc(k) & 1]
    ellA = odd[0] if odd else None
    for k in K:
        K0.append(k ^ ellA if (pc(k) & 1) else k)
    K0 = [k for k in K0 if k]
    cols = [q for q in range(n) if A >> q & 1]
    rows = Fm + K0
    rhs = [pc(f & B) & 1 for f in Fm] + [0] * len(K0)
    w = _solve(rows, rhs, cols) if cols else (0 if all(r == 0 for r in rhs) else None)
    if w is None:
        return False                                   # no Z pattern of the family passes the decoder
    lam = pc(B ^ w) & 1
    if ellA is None:
        return (mu, lam) != (0, 0)
    if mu == 0:
        return True                                    # output of fidelity 1/2
    return (lam + pc((w ^ Ap) & ellA)) & 1 == 1


def certify(circuit, faces, n, delta, time_limit=3600, threads=0, verbose=False, max_cuts=2000, branches=("X", "Z", "C"), params=None):
    """least weight, capped at delta, of a harmful late configuration: relaxed integer programs per branch of the
    late-error theorem, each solution checked exactly and cut off when it is harmless"""
    P, data = pieces(circuit)
    assert len(data) == n
    dets = sorted({d for p in P for d in p[3]})
    di = {d: i for i, d in enumerate(dets)}
    res = dict(pieces=len(P), detectors=len(dets), cuts=0)
    best, wit = delta, None
    for branch in branches:
        m = gp.Model()
        m.Params.OutputFlag = 1 if verbose is True else 0
        m.Params.TimeLimit = time_limit
        if threads:
            m.Params.Threads = threads
        for k, v in (params or {}).items():                 # e.g. NodefileStart, NodefileDir for large trees
            m.setParam(k, v)
        y = m.addVars(len(P), vtype=GRB.BINARY)
        x = m.addVars(n, vtype=GRB.BINARY)
        z = m.addVars(n, vtype=GRB.BINARY)
        u = m.addVars(n, vtype=GRB.BINARY)
        a = m.addVars(n, vtype=GRB.BINARY)          # A, the X pattern crossing U
        ap = m.addVars(n, vtype=GRB.BINARY)         # A', the X pattern after U (pieces and data faults)
        bz = m.addVars(n, vtype=GRB.BINARY)         # B, the Z pattern (pieces and data faults)
        w = m.addVars(n, vtype=GRB.BINARY)          # the free Z pattern inside A
        g = m.addVar(vtype=GRB.BINARY)              # the gadget error times the measured X^n
        par = lambda expr: m.addConstr(expr == 2 * m.addVar(vtype=GRB.INTEGER, lb=0))
        for q in range(n):
            m.addConstr(u[q] >= x[q]); m.addConstr(u[q] >= z[q]); m.addConstr(w[q] <= a[q])
        by_det = {}
        for i, p in enumerate(P):
            for d in p[3]:
                by_det.setdefault(di[d], []).append(i)
        for d, idx in by_det.items():
            par(gp.quicksum(y[i] for i in idx))
        for q in range(n):
            bit = 1 << q
            par(gp.quicksum(y[i] for i, p in enumerate(P) if p[0] & bit) + g - a[q])
            par(gp.quicksum(y[i] for i, p in enumerate(P) if p[1] & bit) + x[q] - ap[q])
            par(gp.quicksum(y[i] for i, p in enumerate(P) if p[2] & bit) + z[q] - bz[q])
        # A and A + 1^n describe the same error (the gadget error times the measured X^n); n is odd, so exactly one
        # of them has at most (n-1)/2 qubits, and we keep that one
        m.addConstr(gp.quicksum(a[q] for q in range(n)) <= (n - 1) // 2)
        for f in faces:
            par(gp.quicksum(a[q] + ap[q] for q in f))            # A + A' commutes with every Z face
            par(gp.quicksum(bz[q] + w[q] for q in f))            # B + w commutes with every X face
        sumT = gp.quicksum(a[q] + ap[q] for q in range(n))
        sumZ = gp.quicksum(bz[q] + w[q] for q in range(n))
        if branch == "X":                                        # mu = 1
            m.addConstr(sumT == 2 * m.addVar(vtype=GRB.INTEGER, lb=0) + 1)
        else:                                                    # mu = 0
            par(sumT)
            if branch == "Z":                                    # lambda = 1
                m.addConstr(sumZ == 2 * m.addVar(vtype=GRB.INTEGER, lb=0) + 1)
            else:                                                # a Z logical inside A
                v = m.addVars(n, vtype=GRB.BINARY)
                for q in range(n):
                    m.addConstr(v[q] <= a[q])
                for f in faces:
                    par(gp.quicksum(v[q] for q in f))
                m.addConstr(gp.quicksum(v[q] for q in range(n)) == 2 * m.addVar(vtype=GRB.INTEGER, lb=0) + 1)
        obj = gp.quicksum(y[i] for i in range(len(P))) + gp.quicksum(u[q] for q in range(n))
        cap = m.addConstr(obj <= best - 1)
        m.setObjective(obj, GRB.MINIMIZE)
        found = "unknown"                                       # unless the loop ends by infeasibility or a witness
        for it in range(max_cuts):
            m.optimize()
            if m.Status == GRB.INFEASIBLE:
                found = None
                break
            if not m.SolCount:
                res[f"{branch}_status"] = int(m.Status)
                break
            A = sum(1 << q for q in range(n) if a[q].X > 0.5)
            Ap = sum(1 << q for q in range(n) if ap[q].X > 0.5)
            B = sum(1 << q for q in range(n) if bz[q].X > 0.5)
            if verbose == "progress":
                print(f"  {branch} it {it} obj {m.ObjVal:.0f} |A| {bin(A).count('1')} |A'| {bin(Ap).count('1')} "
                      f"|B| {bin(B).count('1')} t {m.Runtime:.1f}s", flush=True)
            if exact_harm(A, Ap, B, faces, n):
                found = int(round(m.ObjVal))
                if m.Status != GRB.OPTIMAL:              # a real witness, but maybe not the lightest
                    res[f"{branch}_status"] = int(m.Status)
                    found = f"<={found}"
                wit = dict(branch=branch, pieces=sum(1 for i in range(len(P)) if y[i].X > 0.5),
                           data=int(round(sum(u[q].X for q in range(n)))), A=A, Ap=Ap, B=B)
                break
            res["cuts"] += 1                                    # harmless: exclude this (A, A', B)
            lits = []
            for vec, var in ((A, a), (Ap, ap), (B, bz)):
                for q in range(n):
                    lits.append((1 - var[q]) if vec >> q & 1 else var[q])
            m.addConstr(gp.quicksum(lits) >= 1)
        if found == "unknown":
            res[branch] = "unknown"
            best = None
            break
        res[branch] = found if found is not None else f">={best}"
        if isinstance(found, str):                              # upper bound only: no certificate
            best = None
            break
        if isinstance(found, int):
            best = found
            m.remove(cap)
    res["bound"] = best
    res["witness"] = wit
    return res
