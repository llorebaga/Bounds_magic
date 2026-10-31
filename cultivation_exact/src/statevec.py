"""Brute-force state-vector simulation of a Clifford+T circuit (S-proxy stim file read with real T) with
explicit Pauli faults inserted, postselected on the reference outcome of every measurement.  Used only to
validate the TDEM on small circuits (d1=3, 15 qubits)."""
import numpy as np
import stim

from tdem import n_results, pauli_of_group

W = np.exp(1j * np.pi / 4)
G1 = {
    "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "S": np.diag([1, 1j]), "S_DAG": np.diag([1, -1j]),
    "T": np.diag([1, W]), "T_DAG": np.diag([1, np.conj(W)]),
    "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]),
    "I": np.eye(2),
}


class SV:
    def __init__(self, n):
        self.n = n
        self.psi = np.zeros((2,) * n, dtype=complex)
        self.psi[(0,) * n] = 1

    def g1(self, U, q):
        self.psi = np.moveaxis(np.tensordot(U, self.psi, axes=([1], [q])), 0, q)

    def cx(self, c, t):
        idx = [slice(None)] * self.n
        idx[c] = 1
        sub = self.psi[tuple(idx)]
        tt = t if t < c else t - 1
        self.psi[tuple(idx)] = np.flip(sub, axis=tt)

    def project_pauli(self, P, bit):
        """project onto the (-1)^bit eigenspace of Pauli string P (stim.PauliString)"""
        phi = self.psi.copy()
        for q in range(self.n):
            p = P[q]
            if p:
                phi = np.moveaxis(np.tensordot(G1["IXYZ"[p]], phi, axes=([1], [q])), 0, q)
        s = -1 if bit else 1
        self.psi = (self.psi + s * phi) / 2

    def reset(self, q, basis="Z"):
        if basis == "X":
            self.g1(G1["H"], q)
        idx0 = [slice(None)] * self.n
        idx1 = [slice(None)] * self.n
        idx0[q] = 0
        idx1[q] = 1
        # the qubit is unentangled when reset: keep the rest state with the full norm
        A0, A1 = self.psi[tuple(idx0)].copy(), self.psi[tuple(idx1)].copy()
        n0, n1 = np.vdot(A0, A0).real, np.vdot(A1, A1).real
        if n0 + n1 < 1e-20:                       # branch already rejected
            self.psi[...] = 0
            return
        assert abs(abs(np.vdot(A0, A1)) ** 2 - n0 * n1) <= 1e-9 * max(n0 * n1, 1e-30) + 1e-24, "entangled reset"
        a = A0 * np.sqrt((n0 + n1) / n0) if n0 >= n1 else A1 * np.sqrt((n0 + n1) / n1)
        self.psi[tuple(idx0)] = a
        self.psi[tuple(idx1)] = 0
        if basis == "X":
            self.g1(G1["H"], q)


def run(m, faults, ref_bits=None):
    """Simulate the original circuit of TDEM m with real magic gates (same pattern as m), inserting
    faults = {instruction index: [(qubit, 'X'|'Y'|'Z'), ...] applied after it} and
    flips = measurement records to flip via {('flip', record): True}.  Every measurement is projected onto
    its reference value (m.ref_bits restricted to original records).  Returns the final data state vector
    (other qubits factored out) and its squared norm."""
    n = m.orig.num_qubits
    sv = SV(n)
    ref = {}
    for k, src in enumerate(m.rec_src[:m.n_body_rec]):
        if src[0] == "orig":
            ref[src[1]] = m.ref_bits[k]
    mg = {(i, q): dag for i, q, dag in m.mgates}
    U = "T" if m.magic == "T" else "S"
    rec = 0
    for i, inst in enumerate(m.insts):
        nm = inst.name
        tg = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if nm in ("S", "S_DAG"):
            for q in tg:
                sv.g1(G1[U + ("_DAG" if mg[(i, q)] else "")], q)
        elif nm in ("H", "X", "Y", "Z"):
            for q in tg:
                sv.g1(G1[nm], q)
        elif nm == "CX":
            for a, b in zip(tg[::2], tg[1::2]):
                sv.cx(a, b)
        elif nm in ("R", "RX"):
            for q in tg:
                sv.reset(q, "X" if nm == "RX" else "Z")
        elif nm in ("M", "MX"):
            for q in tg:
                P = stim.PauliString(n)
                P[q] = "Z" if nm == "M" else "X"
                bit = ref[rec] ^ faults.get(("flip", rec), 0)
                sv.project_pauli(P, bit)
                rec += 1
        elif nm == "MPP":
            for g in inst.target_groups():
                if rec not in m.drop_rec:
                    bit = ref[rec] ^ faults.get(("flip", rec), 0)
                    sv.project_pauli(pauli_of_group(g, n), bit)
                rec += 1
        elif nm in ("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK",
                    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR"):
            pass
        else:
            raise ValueError(nm)
        for q, P in faults.get(i, []):
            sv.g1(G1[P], q)
    # factor out non-data qubits: data state = dominant left singular vector
    data = m.data
    others = [q for q in range(n) if q not in data]
    A = np.transpose(sv.psi, data + others).reshape(2 ** len(data), -1)
    nrm = np.vdot(A, A).real
    if nrm < 1e-14:
        return None, 0.0
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    assert s[1] < 1e-7 * s[0] if len(s) > 1 else True
    return u[:, 0] * s[0], nrm
