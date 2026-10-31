"""Physical single-fault table of a TDEM skeleton (every Pauli component of every noise channel, and every
measurement flip), with stim's FlipSimulator: detector flips D, observable flips o, and optionally the X part of
the propagated Pauli frame on the original qubits at every original tick boundary (for flag design)."""
import collections
import numpy as np
import stim

NOISE = ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR")
P2 = [(a, b) for a in "IXYZ" for b in "IXYZ" if a + b != "II"]


class FaultTable:
    def __init__(self, m, frames=False):
        self.m = m
        ops = list(m.noisy.flattened())
        # enumerate faults
        self.prob, self.desc, self.where = [], [], []      # where: index into ops
        for k, inst in enumerate(ops):
            nm = inst.name
            args = inst.gate_args_copy()
            tg = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if nm == "DEPOLARIZE1":
                for q in tg:
                    for P in "XYZ":
                        self._add(args[0] / 3, ((q, P),), k)
            elif nm == "DEPOLARIZE2":
                for a, b in zip(tg[::2], tg[1::2]):
                    for Pa, Pb in P2:
                        self._add(args[0] / 15, tuple((q, P) for q, P in ((a, Pa), (b, Pb)) if P != "I"), k)
            elif nm in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
                for q in tg:
                    self._add(args[0], ((q, nm[0]),), k)
            elif nm in ("M", "MX", "MY") and args and args[0] > 0:
                for q in tg:
                    self._add(args[0], (("flip", q),), k)
        N = len(self.prob)
        self.N = N
        self.prob = np.array(self.prob)
        by_op = collections.defaultdict(list)
        for f, k in enumerate(self.where):
            by_op[k].append(f)
        nq = m.nq
        sim = stim.FlipSimulator(batch_size=N, num_qubits=nq, disable_stabilizer_randomization=True)
        n0 = m.orig.num_qubits
        self.xframes = []
        self.zframes = []
        ntick = 0

        def inject(fs, flip_basis=None):
            masks = {P: np.zeros((nq, N), dtype=bool) for P in "XYZ"}
            any_ = False
            for f in fs:
                for item in self.desc[f]:
                    if item[0] == "flip":
                        if flip_basis is None:
                            continue
                        masks[flip_basis][item[1], f] = True
                    else:
                        masks[item[1]][item[0], f] = True
                    any_ = True
            if any_:
                for P, mk in masks.items():
                    if mk.any():
                        sim.broadcast_pauli_errors(pauli=P, mask=mk)

        for k, inst in enumerate(ops):
            nm = inst.name
            if nm in NOISE:
                inject(by_op.get(k, []))
                continue
            if nm in ("M", "MX", "MY") and k in by_op:
                basis = {"M": "X", "MX": "Z", "MY": "X"}[nm]
                inject(by_op[k], basis)
                sim.do(stim.CircuitInstruction(nm, inst.targets_copy()))
                inject(by_op[k], basis)
                continue
            if nm in ("M", "MX", "MY", "MPP") and inst.gate_args_copy():
                sim.do(stim.CircuitInstruction(nm, inst.targets_copy()))
                continue
            sim.do(inst)
            if nm == "TICK" and frames:
                xs, zs = sim.to_numpy(output_xs=True, output_zs=True)[:2]
                self.xframes.append(xs[:n0].T.copy())         # (N, n0) X bits after this tick
                self.zframes.append(zs[:n0].T.copy())
                ntick += 1
        self.Mflips = sim.get_measurement_flips().T             # (N, nrec) record flips
        self.D = sim.get_detector_flips().T                     # (N, ndet)
        self.O = sim.get_observable_flips().T                   # (N, nobs)
        if frames:
            # skeleton TICK 0 is after the Bell pairs; skeleton TICK i (i>=1) is original TICK i-1
            self.xframes = self.xframes[1:]
            self.zframes = self.zframes[1:]
        self.Dkey = [int.from_bytes(np.packbits(r, bitorder="little").tobytes(), "little") for r in self.D]
        self.okey = [int.from_bytes(np.packbits(r, bitorder="little").tobytes(), "little") for r in self.O]

    def _add(self, p, desc, k):
        self.prob.append(p)
        self.desc.append(desc)
        self.where.append(k)

    def classes(self):
        """group faults by (D, o): returns list of (total prob, D, o, [fault indices])"""
        g = collections.defaultdict(list)
        for f in range(self.N):
            if self.Dkey[f] or self.okey[f]:
                g[(self.Dkey[f], self.okey[f])].append(f)
        out = []
        for (D, o), fs in g.items():
            out.append((float(self.prob[fs].sum()), D, o, fs))
        return out
