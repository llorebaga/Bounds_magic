"""T-aware detector error model (TDEM): Theorem 2 of the draft, made constructive with stim.

Input: a Clifford+magic circuit in the S-proxy convention of Gidney-Shutty-Jones (every S / S_DAG gate is a
magic gate U / U^dag, U = T or, for the proxy itself, U = S). Every magic gate on qubit q is replaced by the
injection gadget  sqrt2 <0|_a CX_{q a} |q>|U>_a = U|q>,  with an ideal ancilla a, and U^dag by the gadget
followed by the ideal Clifford U^dag U^dag... = CORR (S_DAG for T, Z for S).  Each ancilla a_j is the half of a
Bell pair with a reference qubit r_j.  The result (the skeleton) is a Clifford circuit with stochastic Pauli
noise, which stim handles exactly:

  * DETECTORs = parities of measurement records that are deterministic in the noiseless skeleton (the map D);
  * the skeleton ends on the t references and one output qubit (the decoded logical qubit) in a stabilizer
    state |Phi> (the Choi state of the fault-free map M = c V Pi_C), with t+1 stabilizer generators g_i;
  * OBSERVABLE i = g_i, measured noiselessly at the end, plus the record parity that makes it deterministic.

For a fault configuration F with D(F)=0, the observable flips o(F) in F_2^{t+1} are the commutation pattern of
a Pauli R(F) on the references with the g_i, and the faulty (postselected) map is M R(F)^T: R(F) is E(F) of
Theorem 2.  The output for F is  v(o) = (I_out x <conj U|^t)(R x I)|Phi>,  evaluated exactly by writing |Phi>
as a sum over an affine subspace with quadratic phases (cost 2^k, k = rank of the X parts of the g_i).
"""
import numpy as np
import stim

from gf2 import rref, nullspace, right_inverse

PHASE = {"T": np.exp(1j * np.pi / 4), "S": 1j}
CORR = {"T": "S_DAG", "S": "Z"}          # U^dag = CORR . U


def sublattice_signs(n, faces, zsyn=None):
    """s in {+1,-1}^n with #(-1) on each face f = |f|/2 + [Z_f = -1] (mod 2): then (U X U^dag) on all qubits,
    with U^dag where s = -1, maps every X stabilizer to itself on the code space with Z-syndrome zsyn, i.e. it is
    a logical operator there (with real T this matters for weight-6 faces and for Z_f = -1)."""
    zsyn = zsyn or [1] * len(faces)
    from gf2 import solve
    A = np.zeros((len(faces), n), dtype=np.uint8)
    for i, f in enumerate(faces):
        A[i, f] = 1
    rhs = np.array([((len(f) // 2) + (z < 0)) % 2 for f, z in zip(faces, zsyn)], dtype=np.uint8)
    x = solve(A, rhs)
    if x is None:
        raise ValueError("no valid sign pattern")
    # prefer the solution of smallest weight among x + kernel when the kernel is small (matches the old search)
    return [(-1 if x[q] else 1) for q in range(n)]


def n_results(inst):
    if inst.name in ("M", "MX", "MY", "MR", "MRX"):
        return len(inst.targets_copy())
    if inst.name == "MPP":
        return len(inst.target_groups())
    return 0


def pauli_of_group(group, n):
    P = stim.PauliString(n)
    for t in group:
        if t.is_combiner:
            continue
        if t.is_inverted_result_target:
            raise ValueError("inverted MPP targets not supported")
        P[t.value] = "X" if t.is_x_target else ("Y" if t.is_y_target else "Z")
    return P


def wht(F):
    """unnormalised Walsh-Hadamard transform: H[beta] = sum_alpha F[alpha] (-1)^{alpha.beta}"""
    n = len(F)
    F = F.copy()
    h = 1
    while h < n:
        F = F.reshape(-1, 2, h)
        F = np.stack([F[:, 0, :] + F[:, 1, :], F[:, 0, :] - F[:, 1, :]], axis=1)
        h *= 2
    return F.reshape(n)


def popcount64(v):
    v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((v * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


class OpList:
    """instruction list with stable positions (stim.Circuit.append fuses adjacent instructions)"""
    def __init__(self):
        self.ops = []

    def append(self, name, targets=None, args=None):
        if isinstance(name, stim.CircuitInstruction):
            self.ops.append(name)
        else:
            tg = [x if isinstance(x, stim.GateTarget) else stim.GateTarget(x) for x in (targets or [])]
            self.ops.append(stim.CircuitInstruction(name, tg, list(args or [])))

    def __len__(self):
        return len(self.ops)

    def circuit(self, ops=None):
        c = stim.Circuit()
        for op in (self.ops if ops is None else ops):
            c.append(op)
        return c


class TDEM:
    def __init__(self, circuit, magic="T", pattern=True, verbose=False, feedback=None, overrides=None, light=False,
                 zsyn=None, complement=False, flip_gates=None):
        """feedback: {original record: 'X'|'Y'|'Z'} Pauli frame applied to the output when that record is 1
        (see find_frames); overrides: {original record: forced value} for the reference run; light: stop after
        the fault-free output (no detectors / observables)."""
        self.magic = magic
        self.verbose = verbose
        self.orig = circuit
        self.feedback = dict(feedback or {})
        self.overrides = dict(overrides or {})
        self.zsyn_given = zsyn
        self.complement = complement
        self.flip_gates = set(flip_gates or [])
        self._parse_original(pattern and magic == "T")
        self._build_body()
        self._reference_run()
        if light:
            self._affine(light=True)
            return
        self._append_generators()
        self._annotate()
        self._affine()

    # ------------------------------------------------------------------ parsing
    def _parse_original(self, pattern):
        c = self.orig.flattened()
        self.insts = list(c)
        # global record index of every result; detectors / observables in absolute indices
        rec = 0
        self.orig_dets, self.orig_obs = [], set()
        mpp_last = None
        for i, inst in enumerate(self.insts):
            if inst.name == "MPP":
                mpp_last = (i, rec)
            if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                refs = [rec + t.value for t in inst.targets_copy()]
                if inst.name == "DETECTOR":
                    self.orig_dets.append(refs)
                else:
                    self.orig_obs ^= set(refs)
            rec += n_results(inst)
        self.n_orig_rec = rec
        # final MPP: data qubits, stabilizers, logical products (those in the observable) to drop
        i_mpp, r0 = mpp_last
        groups = self.insts[i_mpp].target_groups()
        n0 = self.orig.num_qubits
        self.final_mpp_index = i_mpp
        self.drop_rec = {r0 + g for g in range(len(groups)) if r0 + g in self.orig_obs}
        stabs = [pauli_of_group(groups[g], n0) for g in range(len(groups)) if r0 + g not in self.drop_rec]
        self.data = sorted({t.value for t in self.insts[i_mpp].targets_copy() if not t.is_combiner})
        self.stabs = stabs
        faces = [[q for q in self.data if P[q] == 1] for P in stabs]      # X-type faces
        faces = [f for f in faces if f]
        self.faces = faces
        # magic-gate pattern: on S/S_DAG ticks covering all final data, flip U <-> U^dag where s = -1
        flip = set()
        if pattern:
            loc = {q: k for k, q in enumerate(self.data)}
            zs = self.zsyn_given or [1] * len(faces)
            s = sublattice_signs(len(self.data), [[loc[q] for q in f] for f in faces], zs)
            if self.complement:
                s = [-x for x in s]
            flip = {self.data[k] for k in range(len(self.data)) if s[k] < 0}
        self.flip_set = flip
        # ticks
        tick = 0
        self.tick_of = []
        for inst in self.insts:
            self.tick_of.append(tick)
            if inst.name == "TICK":
                tick += 1
        cover = {}
        for i, inst in enumerate(self.insts):
            if inst.name in ("S", "S_DAG"):
                cover.setdefault(self.tick_of[i], set()).update(t.value for t in inst.targets_copy())
        self.full_ticks = {tk for tk, qs in cover.items() if set(self.data) <= qs}

    # ------------------------------------------------------------------ skeleton
    def _build_body(self):
        n0 = self.orig.num_qubits
        mgates = []
        for i, inst in enumerate(self.insts):
            if inst.name in ("S", "S_DAG"):
                for t in inst.targets_copy():
                    q = t.value
                    dag = inst.name == "S_DAG"
                    if self.tick_of[i] in self.full_ticks and q in self.flip_set:
                        dag = not dag
                    if len(mgates) in self.flip_gates:
                        dag = not dag
                    mgates.append((i, q, dag))
        self.mgates = mgates
        t = len(mgates)
        self.t = t
        self.refs = list(range(n0, n0 + t))
        self.anc = list(range(n0 + t, n0 + 2 * t))
        self.nq = n0 + 2 * t
        body = OpList()
        body.append("R", self.refs + self.anc)
        body.append("H", self.refs)
        body.append("CX", [x for j in range(t) for x in (self.refs[j], self.anc[j])])
        body.append("TICK")
        self.rec_src = []          # per skeleton record: ("orig", r) | ("gadget", j) | ("gen", i)
        orig_rec = 0
        j = 0
        self.pos_after = []        # skeleton body length after emitting original instruction i
        self.rec_start = []        # first original record index of instruction i
        for i, inst in enumerate(self.insts):
            nm = inst.name
            self.rec_start.append(orig_rec)
            self.pos_after.append(len(body))
            if nm in ("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"):
                continue
            if nm in ("S", "S_DAG"):
                for tg in inst.targets_copy():
                    _, q, dag = mgates[j]
                    a = self.anc[j]
                    body.append("CX", [q, a])
                    body.append("M", [a])
                    self.rec_src.append(("gadget", j))
                    if dag:
                        body.append(CORR[self.magic], [q])
                    j += 1
                self.pos_after[-1] = len(body)
                continue
            if nm == "MPP" and i == self.final_mpp_index:
                groups = inst.target_groups()
                keep = []
                for g, grp in enumerate(groups):
                    r = orig_rec + g
                    if r in self.drop_rec:
                        continue
                    keep.append(grp)
                    self.rec_src.append(("orig", r))
                targets = []
                for grp in keep:
                    for k, x in enumerate([x for x in grp if not x.is_combiner]):
                        if k:
                            targets.append(stim.target_combiner())
                        targets.append(x)
                body.append("MPP", targets, inst.gate_args_copy())
                orig_rec += len(groups)
                self.pos_after[-1] = len(body)
                continue
            k = n_results(inst)
            for g in range(k):
                self.rec_src.append(("orig", orig_rec + g))
            orig_rec += k
            body.append(inst)
            self.pos_after[-1] = len(body)
        assert j == t
        # end: discard everything but data and references, decode the data block onto one qubit
        others = [q for q in range(self.nq) if q not in set(self.data) and q not in set(self.refs)]
        body.append("TICK")
        body.append("R", others)
        nd = len(self.data)
        loc_stabs = []
        for P in self.stabs:
            Q = stim.PauliString(nd)
            for k, q in enumerate(self.data):
                Q[k] = P[q]
            loc_stabs.append(Q)
        ZL = stim.PauliString("Z" * nd)
        tab = stim.Tableau.from_stabilizers(loc_stabs + [ZL])
        dec = tab.inverse().to_circuit("elimination")
        for inst in dec:
            body.append(inst.name, [stim.GateTarget(self.data[x.value]) for x in inst.targets_copy()],
                        inst.gate_args_copy())
        self.out = self.data[-1]
        nrec_now = len(self.rec_src)
        pos_of = {src[1]: k for k, src in enumerate(self.rec_src) if src[0] == "orig"}
        for r, P in sorted(self.feedback.items()):
            tgt = stim.target_rec(pos_of[r] - nrec_now)
            if P in ("X", "Y"):
                body.append("CX", [tgt, stim.GateTarget(self.out)])
            if P in ("Z", "Y"):
                body.append("CZ", [tgt, stim.GateTarget(self.out)])
        body.append("R", self.data[:-1])
        self.ops = body.ops
        self.body = body.circuit()
        self.n_body_rec = len(self.rec_src)
        if self.verbose:
            print(f"skeleton: {n0} qubits + {t} magic gates; {self.n_body_rec} records", flush=True)

    # ------------------------------------------------------------------ reference run
    def _reference_run(self):
        sim = stim.TableauSimulator()
        sim.set_num_qubits(self.nq)
        last_of = {}
        for D in self.orig_dets:
            if any(r in self.drop_rec for r in D):
                continue
            last_of.setdefault(max(D), []).append(D)
        val = {}
        ref_bits = []
        self.free_records = []
        self.zsyn_seen = None
        k = 0
        for inst in self.body.without_noise():
            nm = inst.name
            if nm in ("M", "MX", "MPP"):
                if nm == "MPP":
                    items = [("P", pauli_of_group(g, self.nq)) for g in inst.target_groups()]
                else:
                    items = [(nm, t.value) for t in inst.targets_copy()]
                for kind, x in items:
                    src = self.rec_src[k]
                    if kind == "M":
                        e = sim.peek_z(x)
                    elif kind == "MX":
                        e = sim.peek_x(x)
                    else:
                        e = sim.peek_observable_expectation(x)
                    if src[0] == "gadget":
                        if self.zsyn_seen is None and self.tick_of[self.mgates[src[1]][0]] in self.full_ticks:
                            n = self.nq
                            zs = []
                            for f in self.faces:
                                P = stim.PauliString(n)
                                for q in f:
                                    P[q] = "Z"
                                zs.append(int(sim.peek_observable_expectation(P)))
                            self.zsyn_seen = zs
                        want = 0
                        if e == -1:
                            raise RuntimeError("gadget outcome deterministically 1")
                    else:
                        r = src[1]
                        want = None
                        for D in last_of.get(r, []):
                            w = 0
                            for rr in D:
                                if rr != r:
                                    w ^= val[rr]
                            if want is None:
                                want = w
                            elif want != w:
                                raise RuntimeError("inconsistent original detectors")
                        if e == 0 and r in self.overrides and want is not None:
                            want = self.overrides[r]          # force a branch against the detectors (tests)
                        if e != 0:
                            det = int(e == -1)
                            if want is not None and want != det:
                                raise RuntimeError(f"record {r}: deterministic value contradicts a detector")
                            want = det
                        if want is None:
                            want = self.overrides.get(r, 0)
                            self.free_records.append(r)
                        val[r] = want
                    if e == 0:
                        if kind == "M":
                            sim.postselect_z(x, desired_value=bool(want))
                        elif kind == "MX":
                            sim.postselect_x(x, desired_value=bool(want))
                        else:
                            sim.postselect_observable(x, desired_value=bool(want))
                    ref_bits.append(want)
                    k += 1
            elif nm == "TICK":
                continue
            elif nm in ("CX", "CZ") and inst.targets_copy()[0].is_measurement_record_target:
                tg = inst.targets_copy()
                for a, b in zip(tg[::2], tg[1::2]):
                    if ref_bits[k + a.value]:
                        (sim.x if nm == "CX" else sim.z)(b.value)
            else:
                sim.do(inst)
        assert k == self.n_body_rec
        self.ref_bits = ref_bits
        # Choi state on references + output
        S = self.refs + [self.out]
        Sset = set(S)
        rows = []
        for P in sim.canonical_stabilizers():
            Q = stim.PauliString(len(S))
            ok = True
            for q in range(self.nq):
                p = P[q]
                if p == 0:
                    continue
                if q in Sset:
                    continue
                if p != 3:
                    ok = False
            if not ok:
                raise RuntimeError("non-subsystem qubit not in |0>")
            for k2, q in enumerate(S):
                Q[k2] = P[q]
            Q *= P.sign
            if Q.weight:
                rows.append(Q)
        # independent generators
        M = np.array([np.concatenate([Q.to_numpy()[0], Q.to_numpy()[1]]) for Q in rows], dtype=np.uint8)
        R, piv = rref(M.T)
        gens = [rows[p] for p in piv]
        if len(gens) != len(S):
            raise RuntimeError(f"Choi state has {len(gens)} generators on {len(S)} qubits")
        self.gens = gens
        self.S = S

    def _append_generators(self):
        full = []
        for Q in self.gens:
            P = stim.PauliString(self.nq)
            for k, q in enumerate(self.S):
                P[q] = Q[k]
            full.append(P)
        self.gens_full = full
        targets = []
        for P in full:
            first = True
            for q in range(self.nq):
                p = P[q]
                if p == 0:
                    continue
                if not first:
                    targets.append(stim.target_combiner())
                first = False
                targets.append([None, stim.target_x, stim.target_y, stim.target_z][p](q))
        self.noisy = self.body.copy()
        self.noisy.append("TICK")
        self.noisy.append("MPP", targets)
        for i in range(len(full)):
            self.rec_src.append(("gen", i))
        self.n_rec = len(self.rec_src)

    # ------------------------------------------------------------------ detectors / observables
    def _annotate(self, shots=None):
        clean = self.noisy.without_noise()
        M = self.n_rec
        shots = shots or (M + 256)
        s = clean.compile_sampler().sample(shots).astype(np.uint8)
        A = np.hstack([s, np.ones((shots, 1), dtype=np.uint8)])
        B = nullspace(A)[:, :M]                   # drop the constant column
        ng = len(self.gens)
        order = list(range(self.n_body_rec, M)) + list(range(self.n_body_rec))
        R, piv = rref(B[:, order])
        R = R[:len(piv)]
        inv = np.argsort(order)
        R = R[:, inv]
        gen_cols = set(range(self.n_body_rec, M))
        obs_rows, det_rows = {}, []
        for r, p in enumerate(piv):
            col = order[p]
            if col in gen_cols:
                obs_rows[col - self.n_body_rec] = R[r]
            else:
                det_rows.append(R[r])
        if len(obs_rows) != ng:
            raise RuntimeError(f"only {len(obs_rows)} of {ng} generators are deterministic")
        self.det_rows = np.array(det_rows, dtype=np.uint8).reshape(-1, M)
        self.obs_rows = np.array([obs_rows[i] for i in range(ng)], dtype=np.uint8)
        for row in self.det_rows:
            self.noisy.append("DETECTOR", [stim.target_rec(int(m) - M) for m in np.nonzero(row)[0]])
        for i, row in enumerate(self.obs_rows):
            self.noisy.append("OBSERVABLE_INCLUDE", [stim.target_rec(int(m) - M) for m in np.nonzero(row)[0]], i)
        self.n_det = len(self.det_rows)
        if self.verbose:
            print(f"skeleton detectors: {self.n_det}, observables: {ng} (original circuit: "
                  f"{len(self.orig_dets)} detectors)", flush=True)

    def dem(self):
        return self.noisy.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)

    def mechanisms(self):
        """list of (p, D bitmask, o bitmask) for every error mechanism of the stim DEM"""
        out = []
        for inst in self.dem().flattened():
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            D = 0
            o = 0
            for tg in inst.targets_copy():
                if tg.is_relative_detector_id():
                    D ^= 1 << tg.val
                elif tg.is_logical_observable_id():
                    o ^= 1 << tg.val
            out.append((p, D, o))
        return out

    # ------------------------------------------------------------------ exact evaluation
    def _affine(self, light=False):
        t, N = self.t, len(self.S)
        X = np.zeros((N, N), dtype=np.uint8)
        Z = np.zeros((N, N), dtype=np.uint8)
        lam = np.zeros(N, dtype=np.int64)
        for i, Q in enumerate(self.gens):
            x, z = Q.to_numpy()
            X[i], Z[i] = x, z
            lam[i] = (2 if Q.sign == -1 else 0) + int(np.sum(x & z))
        # Gaussian elimination on the X parts, tracking phases: P1 P2 = i^{l1+l2+2 z1.x2} X^{x1+x2} Z^{z1+z2}
        r = 0
        for c in range(N):
            rows = [i for i in range(r, N) if X[i, c]]
            if not rows:
                continue
            p = rows[0]
            if p != r:
                X[[r, p]] = X[[p, r]]
                Z[[r, p]] = Z[[p, r]]
                lam[[r, p]] = lam[[p, r]]
            for i in range(N):
                if i != r and X[i, c]:
                    lam[i] = (lam[i] + lam[r] + 2 * int(np.sum(Z[i] & X[r]))) % 4
                    X[i] ^= X[r]
                    Z[i] ^= Z[r]
            r += 1
        k = r
        self.k = k
        # Z constraints z.x = lam/2 ; particular solution x0
        from gf2 import solve
        Zc, bc = Z[k:], (lam[k:] // 2) % 2
        assert np.all(lam[k:] % 2 == 0)
        x0 = solve(Zc, bc) if len(Zc) else np.zeros(N, dtype=np.uint8)
        assert x0 is not None
        # enumerate the affine space with amplitudes: c(x + x_i) = i^{lam_i} (-1)^{z_i.x} c(x)
        xs = x0[None, :].copy()
        cs = np.ones(1, dtype=complex)
        ipow = np.array([1, 1j, -1, -1j])
        for i in range(k):
            sgn = 1 - 2 * ((xs.astype(np.int64) @ Z[i].astype(np.int64)) % 2)
            xs = np.vstack([xs, xs ^ X[i]])
            cs = np.concatenate([cs, ipow[lam[i] % 4] * sgn * cs])
        if self.verbose:
            print(f"Choi state: N={N}, affine dimension k={k} ({len(cs)} terms)", flush=True)
        W = (t + 63) // 64
        self.W = W
        packed = np.zeros((len(cs), W), dtype=np.uint64)
        for j in range(t):
            packed[:, j // 64] |= xs[:, j].astype(np.uint64) << np.uint64(j % 64)
        self.xr = packed
        self.xo = xs[:, t].astype(bool)
        # basis of the affine space (references part, packed) for the Walsh-Hadamard evaluation path
        self.kaff = k
        self.U_ref = np.array([self._pack(X[i][:t]) for i in range(k)], dtype=np.uint64).reshape(k, W)
        self.x0_ref = self._pack(x0[:t]) if k >= 0 else None
        self._wht = {}
        self._acount = {}
        self.cs = cs
        self.phase_pow = PHASE[self.magic] ** np.arange(t + 1)
        if light:
            self.v0 = self.output_ab(np.zeros(t, np.uint8), np.zeros(t, np.uint8))
            self.n0 = np.vdot(self.v0, self.v0).real
            return
        # o -> R (a | b) on references
        G = np.zeros((N, 2 * t), dtype=np.uint8)
        for i, Q in enumerate(self.gens):
            x, z = Q.to_numpy()
            G[i, :t] = z[:t]
            G[i, t:] = x[:t]
        self.Rinv = right_inverse(G)            # (2t) x N
        self.v0 = self.output_ab(np.zeros(t, np.uint8), np.zeros(t, np.uint8))
        n0 = np.vdot(self.v0, self.v0).real
        if n0 < 1e-12:
            raise RuntimeError("fault-free output vanishes")
        self.u0 = self.v0 / np.sqrt(n0)
        self.n0 = n0
        self._cache = {}

    def _pack(self, bits):
        w = np.zeros(self.W, dtype=np.uint64)
        for j in np.nonzero(bits)[0]:
            w[j // 64] |= np.uint64(1) << np.uint64(j % 64)
        return w

    def output_ab(self, a, b):
        """(I x <conj U|^t)(X^a Z^b x I)|Phi>, a 2-vector"""
        pa, pb = self._pack(a), self._pack(b)
        w = np.zeros(len(self.cs), dtype=np.int64)
        s = np.zeros(len(self.cs), dtype=np.int64)
        for q in range(self.W):
            w += popcount64(self.xr[:, q] ^ pa[q])
            s += popcount64(self.xr[:, q] & pb[q])
        term = self.cs * self.phase_pow[w] * (1 - 2 * (s & 1))
        return np.array([term[~self.xo].sum(), term[self.xo].sum()])

    def output_wht(self, a, b, threshold=2, max_tables=10):
        """same as output_ab, but once an X part a has been seen `threshold` times, the outputs for every Z part
        are read from a Walsh-Hadamard transform over the affine coordinates (one O(k 2^k) transform per a)"""
        ka = a.tobytes()
        tab = self._wht.get(ka)
        if tab is None:
            self._acount[ka] = self._acount.get(ka, 0) + 1
            if self._acount[ka] < threshold:
                return self.output_ab(a, b)
            pa = self._pack(a)
            w = np.zeros(len(self.cs), dtype=np.int64)
            for q in range(self.W):
                w += popcount64(self.xr[:, q] ^ pa[q])
            F = self.cs * self.phase_pow[w]
            tab = (wht(np.where(self.xo, 0, F)), wht(np.where(self.xo, F, 0)))
            if len(self._wht) >= max_tables:
                self._wht.pop(next(iter(self._wht)))
            self._wht[ka] = tab
        pb = self._pack(b)
        beta = 0
        for i in range(self.kaff):
            par = 0
            for q in range(self.W):
                par += int(popcount64(np.array([self.U_ref[i, q] & pb[q]]))[0])
            if par & 1:
                beta |= 1 << i
        sx = 0
        for q in range(self.W):
            sx += int(popcount64(np.array([self.x0_ref[q] & pb[q]]))[0])
        sg = -1 if sx & 1 else 1
        return np.array([sg * tab[0][beta], sg * tab[1][beta]])

    def R_of(self, o):
        ob = np.array([(o >> i) & 1 for i in range(len(self.gens))], dtype=np.uint8)
        R = (self.Rinv.astype(np.int64) @ ob) % 2
        return R[:self.t].astype(np.uint8), R[self.t:].astype(np.uint8)

    def evaluate(self, o):
        """(relative acceptance, bad) for observable-flip mask o, with D = 0"""
        if o in self._cache:
            return self._cache[o]
        if o == 0:
            res = (1.0, 0.0)
        else:
            a, b = self.R_of(o)
            v = self.output_wht(a, b)
            acc = np.vdot(v, v).real / self.n0
            good = abs(np.vdot(self.u0, v)) ** 2 / self.n0
            res = (acc, max(acc - good, 0.0))
        self._cache[o] = res
        return res

    def harmful(self, o, tol=1e-9):
        acc, bad = self.evaluate(o)
        return bad > tol

    def output_bloch(self):
        u = self.u0
        X = 2 * (np.conj(u[0]) * u[1]).real
        Y = 2 * (np.conj(u[0]) * u[1]).imag
        Zc = abs(u[0]) ** 2 - abs(u[1]) ** 2
        return X, Y, Zc


PAULI1 = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
          "Z": np.diag([1, -1])}


def find_frames(circuit, magic="T", verbose=False):
    """Pauli frame of the protocol: for every measurement that is random in the fault-free real circuit, the
    logical Pauli relating the output of the branch with that outcome flipped to the reference output.
    Returns {original record: 'X'|'Y'|'Z'} (records whose flip needs no correction are omitted)."""
    base = TDEM(circuit, magic=magic, light=True)
    u0 = base.v0 / np.sqrt(base.n0)
    frames = {}
    for r in base.free_records:
        try:
            alt = TDEM(circuit, magic=magic, light=True, overrides={r: 1})
        except RuntimeError:
            continue
        if alt.n0 < 1e-10 * base.n0:
            continue                      # deterministic in the real circuit
        u1 = alt.v0 / np.sqrt(alt.n0)
        best = max("IXYZ", key=lambda P: abs(np.vdot(PAULI1[P] @ u0, u1)))
        ov = abs(np.vdot(PAULI1[best] @ u0, u1))
        if ov < 1 - 1e-9:
            raise RuntimeError(f"record {r}: branch output is not a Pauli image of the reference (overlap {ov})")
        if verbose:
            print(f"  record {r}: random, relative weight {alt.n0 / base.n0:.3f}, frame {best}", flush=True)
        if best != "I":
            frames[r] = best
    return frames


def syndrome_pattern_tdem(circuit, magic="T", **kw):
    """TDEM whose T/T^dag pattern on the full-width magic layers matches the Z syndrome of the reference run
    there (a syndrome-dependent choice of T or T^dag, i.e. a classically controlled S^dag correction)."""
    probe = TDEM(circuit, magic=magic, light=True)
    zs = probe.zsyn_seen
    if zs is None or 0 in zs:
        raise RuntimeError(f"Z syndrome at the magic layer not deterministic: {zs}")
    m = TDEM(circuit, magic=magic, zsyn=zs, **kw)
    m.zsyn_used = zs
    return m


def best_reference(circuit, magic="T", verbose=False, **kw):
    """TDEM on the fault-free branch of largest weight.  The reference run fixes every outcome that is random in
    the skeleton; if one choice of the free outcomes leaves an X-type Pauli frame before the magic layers, the
    real-gate check is no longer deterministic in that branch (weight halved).  A real-gate protocol must undo
    such frames by feedback before the magic layers, which is the branch of maximal weight: undetected
    combinations of pure outcome flips with relative weight > 1 are applied to the reference until none is left."""
    from faults import FaultTable
    from gf2 import nullspace
    over = {}
    for _ in range(8):
        m = TDEM(circuit, magic=magic, overrides=over, **kw)
        ft = FaultTable(m)
        flips = [f for f in range(ft.N) if ft.desc[f][0][0] == "flip"]
        B = nullspace(ft.D[flips].astype(np.uint8).T)
        best, best_acc = None, 1 + 1e-9
        for mask in range(1, 1 << len(B)):
            row = np.zeros(B.shape[1], dtype=np.uint8)
            for i in range(len(B)):
                if (mask >> i) & 1:
                    row ^= B[i]
            o = 0
            sel = [flips[i] for i in np.nonzero(row)[0]]
            for f in sel:
                o ^= ft.okey[f]
            acc, _ = m.evaluate(o)
            if acc > best_acc:
                best, best_acc = sel, acc
        if best is None:
            m.overrides_used = over
            return m
        # records flipped by this combination that the reference run chose freely
        sk2orig = {k: src[1] for k, src in enumerate(m.rec_src) if src[0] == "orig"}
        flipped = set()
        for f in best:
            for k in np.nonzero(ft.Mflips[f])[0]:
                if k in sk2orig:
                    flipped ^= {sk2orig[k]}
        free = [r for r in flipped if r in m.free_records]
        if not free:
            raise RuntimeError("heavier branch is not reachable by flipping free outcomes")
        refb = {src[1]: m.ref_bits[k] for k, src in enumerate(m.rec_src[:m.n_body_rec]) if src[0] == "orig"}
        for r in free:
            over[r] = 1 - refb[r]
        if verbose:
            print(f"  heavier branch (relative weight {best_acc:.3g}): flipping free outcomes {free}", flush=True)
    raise RuntimeError("no maximal-weight branch found")


def max_neighbour_weight(m):
    """largest relative weight a(o) over undetected (D=0) combinations of pure outcome flips"""
    from faults import FaultTable
    from gf2 import nullspace
    ft = FaultTable(m)
    flips = [f for f in range(ft.N) if ft.desc[f][0][0] == "flip"]
    B = nullspace(ft.D[flips].astype(np.uint8).T)
    best = 0.0
    for mask in range(1, 1 << len(B)):
        row = np.zeros(B.shape[1], dtype=np.uint8)
        for i in range(len(B)):
            if (mask >> i) & 1:
                row ^= B[i]
        o = 0
        for i in np.nonzero(row)[0]:
            o ^= ft.okey[flips[i]]
        best = max(best, m.evaluate(o)[0])
    return best


def best_reference2(circuit, magic="T", verbose=False, **kw):
    """Reference branch by exhaustive choice of the free outcomes: the first assignment whose fault-free output
    exists and that has no undetected outcome-flip neighbour of larger weight (a locally maximal branch)."""
    import itertools
    base = TDEM(circuit, magic=magic, light=True)
    free = list(base.free_records)
    for bits in itertools.product([0, 1], repeat=len(free)):
        over = dict(zip(free, bits))
        try:
            m = TDEM(circuit, magic=magic, overrides=over, **kw)
        except RuntimeError:
            continue
        w = max_neighbour_weight(m)
        if verbose:
            print(f"  free outcomes {over}: max neighbour weight {w:.3g}", flush=True)
        if w <= 1 + 1e-9:
            m.overrides_used = over
            return m
    raise RuntimeError("no locally maximal branch")
