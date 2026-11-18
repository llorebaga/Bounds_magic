"""Late-error bound on the real-T fault distance of a one-round cultivation check (exact criterion for late errors).

A late configuration puts X errors on the data only between the U^dag and U layers of the last check round (where
X errors commute with the measured X^n), plus X and Z data faults after U. By the late-error theorem of the paper
(flat phases), such a configuration with X pattern A at the U layer, X^A' after it and Z^B is harmful iff
  A + A' lies in S_X (mu = 0) or l_X + S_X (mu = 1), and some w <= A orthogonal to every Z stabilizer inside A
  completes X^{A+A'} Z^{B+w} to a nontrivial logical,
with the special rule of the theorem when A contains a Z logical support. Hence s accepted faults leaving A give the
upper bound s + comp(A) on the fault distance, where comp(A) is the fewest further data faults:
  mu = 1: d(A, l_X + S_X), plus one Z fault inside A if A contains a Z-logical support (B = 0 is then harmless);
  mu = 0: d(A, S_X), plus nothing if A contains a Z-logical support (the output has fidelity 1/2), else
          min over Z logicals L of |L \\ A| + d_A(L cap A, K0_A^perp).
The family test of Theorem (Coherent family) drops the parity condition and the +1, so it can only be lower.

Pieces are found generically from the stim circuit (S/S_DAG stand for T/T^dag): every single X fault after every
operation of the check round on every qubit it touches is propagated through CX/H/resets to the U layer, recording
the data X pattern and the gadget detectors it flips (Z-basis measurements of the round); single measurement flips
are pieces with an empty pattern. Combinations of up to SMAX pieces with cancelling detectors are searched.
Usage (as a module): bound(circuit, faces, n, delta, smax)"""
import os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

pc = lambda m: bin(m).count("1")


class CodeTools:
    def __init__(self, faces, n):
        self.n = n
        F = [sum(1 << q for q in f) for f in faces]
        self.F = F
        span = [0]
        for f in F:
            span += [s ^ f for s in span]
        self.SX = np.array(span, dtype=np.uint64)
        self.LX = self.SX ^ np.uint64((1 << n) - 1)
        self._pop = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int16)
        self.SXl = [int(x) for x in self.SX]
        self.LXl = [int(x) for x in self.LX]
        self._SXs, self._LXs = set(self.SXl), set(self.LXl)
        self._ins = {}
        self.cache = {}

    def popv(self, x):
        out = np.zeros(x.shape, dtype=np.int16)
        m = np.uint64(0xFFFF)
        for k in range(4):
            out += self._pop[((x >> np.uint64(16 * k)) & m).astype(np.int64)]
        return out

    def comp(self, A, cap, Ap=0):
        """fewest further data faults (X or Z after the U layer) completing the crossing pattern A, with the X
        pattern Ap already after the layer, to a harmful late configuration (capped)"""
        if (A, Ap) in self.cache:
            val, exact = self.cache[(A, Ap)]
            if exact or cap <= val:
                return min(val, cap)
        a = np.uint64(A)
        t = np.uint64(A ^ Ap)
        inside_log = bool((self.popv(self.LX & ~a) == 0).any())   # A contains a Z (= X) logical support
        dX = int(self.popv(self.LX ^ t).min())
        dS = self.popv(self.SX ^ t)
        if min(dX, int(dS.min())) >= cap:          # both branches cost at least the cap
            self.cache[(A, Ap)] = (cap, False)
            return cap
        if inside_log:                             # B = 0 is harmless here: find the cheapest harmful Z part
            gi = int(np.argmin(self.popv(self.LX ^ t)))
            Ap2 = Ap ^ (A ^ Ap ^ int(self.LX[gi]))       # X completion after the layer onto that logical
            best = dX + self._zfix(A, Ap2, min(cap - dX - 1, 3))
        else:
            best = dX
        dmin = int(dS.min())
        if inside_log:
            best = min(best, dmin)
        elif dmin < best:
            K0 = [int(s) for s in self.SX[self.popv(self.SX & ~a) == 0] if s]
            basis = reduce_basis(K0)
            zl = self._zcost(A, basis, best - dmin)
            best = min(best, dmin + zl)
        exact = best < cap                         # values at the cap are only lower bounds
        best = min(best, cap)
        self.cache[(A, Ap)] = (best, exact)
        return best

    def harmful(self, A, Ap, B):
        """exact late-error criterion: (accepted, harmful) for X^A crossing the layer, X^Ap and Z^B after it"""
        tot = A ^ Ap
        if tot in self._SXs:
            mu = 0
        elif tot in self._LXs:
            mu = 1
        else:
            return False, False
        K0b, logs = self._inside(A)
        rows = [(f & A, pc(f & B) & 1) for f in self.F] + [(k, 0) for k in K0b]
        w = solve_gf2(rows)
        if w is None:
            return False, False
        lam = 0 if (B ^ w) in self._SXs else 1
        if not logs:
            return True, (mu, lam) != (0, 0)
        if mu == 0:
            return True, True
        cp_flip = pc((w ^ Ap) & logs[0]) & 1
        return True, not (mu == 1 and (lam ^ cp_flip) == 0)

    def _inside(self, A):
        """basis of the Z stabilizers inside A and the Z logicals inside A (cached)"""
        if A not in self._ins:
            a = np.uint64(A)
            K0 = [int(k) for k in self.SX[self.popv(self.SX & ~a) == 0] if k]
            logs = [int(l) for l in self.LX[self.popv(self.LX & ~a) == 0][:1]]
            self._ins[A] = (reduce_basis(K0), logs)
        return self._ins[A]

    def _zfix(self, A, Ap, wmax):
        """fewest Z faults (at most wmax) that make the configuration harmful; 99 if none (no witness)"""
        for wt in range(1, wmax + 1):
            for c in itertools.combinations(range(self.n), wt):
                acc, h = self.harmful(A, Ap, sum(1 << q for q in c))
                if h:
                    return wt
        return 99

    def _zcost(self, A, basis, cap):
        """min over Z logicals L of |L \\ A| + (fewest flips inside A fixing the parities of L cap A on K0)"""
        a = np.uint64(A)
        out = self.popv(self.LX & ~a)
        best = cap
        idx = np.nonzero(out < cap)[0]
        order = idx[np.argsort(out[idx])]
        U = 0
        for k in basis:
            U |= k
        for i in order:
            o = int(out[i])
            if o >= best:
                break
            v = int(self.LX[i]) & A
            syn = tuple(pc(v & k) & 1 for k in basis)
            best = min(best, o + parity_fix(syn, basis, U))
        return best


def reduce_basis(vecs):
    basis = []
    for v in vecs:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return basis


def solve_gf2(rows):
    piv = []
    for m, r in rows:
        for pm, pr, b in piv:
            if (m >> b) & 1:
                m ^= pm; r ^= pr
        if m == 0:
            if r:
                return None
            continue
        b = m.bit_length() - 1
        piv = [(pm ^ m, pr ^ r, pb) if (pm >> b) & 1 else (pm, pr, pb) for pm, pr, pb in piv]
        piv.append((m, r, b))
    return sum(1 << b for m, r, b in piv if r)


def parity_fix(syn, basis, U):
    if not any(syn):
        return 0
    qs = [q for q in range(U.bit_length()) if (U >> q) & 1]
    for w in range(1, 4):
        for c in itertools.combinations(qs, w):
            x = sum(1 << q for q in c)
            if tuple(pc(x & k) & 1 for k in basis) == syn:
                return w
    return 4


def pieces(circuit):
    """(A, A', flipped gadget detectors) for every single late X fault and every single measurement flip of the last
    check round. A: data X present at the data qubit's last T gate (the U layer); A': the rest of the final data X
    frame; a fault whose X reaches a data qubit at its second-to-last T gate (U^dag) is early and skipped."""
    ins = list(circuit.flattened())
    tgates = {}
    for k, op in enumerate(ins):
        if op.name in ("S", "S_DAG"):
            for t in op.targets_copy():
                tgates.setdefault(t.value, []).append(k)
    data = sorted(q for q in tgates if len(tgates[q]) >= 2)
    dpos = {q: i for i, q in enumerate(data)}
    t1 = {q: tgates[q][-2] for q in data}
    t2 = {q: tgates[q][-1] for q in data}
    end = next((k for k, op in enumerate(ins) if op.name == "MPP"), len(ins))
    start, stop = min(t1.values()) + 1, max(t2.values())
    rec_of, nrec = {}, 0
    for k, op in enumerate(ins):
        if op.name in ("M", "MX", "MY", "MR", "MRX", "MPP"):
            ntg = len(op.target_groups()) if op.name == "MPP" else len(op.targets_copy())
            rec_of[k] = list(range(nrec, nrec + ntg))
            nrec += ntg
    mpp_recs = set(r for k in rec_of if ins[k].name == "MPP" for r in rec_of[k])
    dets, nr = [], 0
    for op in ins:
        if op.name in ("M", "MX", "MY", "MR", "MRX"):
            nr += len(op.targets_copy())
        elif op.name == "MPP":
            nr += len(op.target_groups())
        elif op.name == "DETECTOR":
            dets.append(frozenset(nr + t.value for t in op.targets_copy()))
    det_of_rec = {}
    for i, D in enumerate(dets):
        if not (D & mpp_recs):                    # gadget detectors only; the ideal decoder is handled by comp
            for r in D:
                det_of_rec.setdefault(r, set()).add(i)

    def propagate(k0, xs):
        x, flips, A = set(xs), set(), 0
        for k in range(k0, end):
            op = ins[k]
            nm = op.name
            if nm == "MPP":
                break
            tg = [t.value for t in op.targets_copy()]
            if nm in ("CX", "CNOT"):
                for c, t in zip(tg[::2], tg[1::2]):
                    if c in x:
                        x ^= {t}
            elif nm in ("S", "S_DAG"):
                for q in tg:
                    if q in x and q in dpos:
                        if k == t1[q] or k < t1[q]:
                            return None
                        if k == t2[q]:
                            A |= 1 << dpos[q]
            elif nm in ("R", "RX", "MR", "MRX"):
                if nm == "MR":
                    for j, q in enumerate(tg):
                        if q in x:
                            flips ^= det_of_rec.get(rec_of[k][j], set())
                x -= set(tg)
            elif nm == "M":
                for j, q in enumerate(tg):
                    if q in x:
                        flips ^= det_of_rec.get(rec_of[k][j], set())
            elif nm == "H":
                if set(tg) & x:
                    return None                    # X turned into Z: not a pure late X piece
        fin = sum(1 << dpos[q] for q in x if q in dpos)
        return A, fin ^ A, frozenset(flips)

    out = set()
    for k in range(start, end):
        op = ins[k]
        if op.name in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "S", "S_DAG") or                 op.name.startswith(("DEPOLARIZE", "X_ERROR", "Z_ERROR", "Y_ERROR", "PAULI")):
            continue
        tg = [t.value for t in op.targets_copy()]
        for q in (set(tg) if k < stop else ()):
            if q in dpos and k < t1[q]:
                continue
            r = propagate(k + 1, [q])
            if r is not None:
                out.add(r)
        if k in rec_of and op.name in ("M", "MX", "MR", "MRX"):
            for r in rec_of[k]:
                fl = frozenset(det_of_rec.get(r, set()))
                if fl:
                    out.add((0, 0, fl))
    return [p for p in out if p[0] or p[1] or p[2]], data


def bound(circuit, faces, n, delta, smax=None, ct=None):
    """min over accepted combinations of at most smax late pieces of (#pieces + comp(A)), capped at delta"""
    smax = delta - 1 if smax is None else smax
    ct = ct or CodeTools(faces, n)
    P, data = pieces(circuit)
    assert len(data) == n
    dets = sorted({d for _, _, D in P for d in D})
    di = {d: i for i, d in enumerate(dets)}
    vec = sorted(set((a, ap, sum(1 << di[d] for d in D)) for a, ap, D in P))
    best, wit = delta, None
    level = {(0, 0, 0): ()}
    for s in range(1, smax + 1):
        nxt = {}
        for (A, Ap, Dv), w in level.items():
            for i, (a, ap, dv) in enumerate(vec):
                if w and i <= w[-1]:
                    continue
                key = (A ^ a, Ap ^ ap, Dv ^ dv)
                if key not in nxt and key not in level:
                    nxt[key] = w + (i,)
        for (A, Ap, Dv), w in nxt.items():
            if Dv == 0 and (A or Ap) and s < best:
                c = ct.comp(A, best - s, Ap)
                if s + c < best:
                    best, wit = s + c, (s, A, Ap)
        level = nxt
        if s + 1 >= best:
            break
    return best, wit, len(vec)
