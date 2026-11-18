"""Numerical check of the circuit-level characterization in paper/cultivation.tex (Theorem 2 there).

A circuit is non-adaptive and fully postselected: Clifford gates, T and T^dagger gates, Pauli projections
with fixed signs, and ideal removals <0| of qubits. Faults are Paulis right after a gate, Paulis right before
a projection, or flips of a projection's sign. For small random circuits we check:
  (i)   gadget: replacing each T by CNOT(q->a), <0|_a on an ideal |T>_a ancilla (times sqrt2) reproduces
        every faulty circuit, so K_F|0> = M_F |T>^t with M_F the faulty skeleton map on the t T-inputs;
  (ii)  for every fault configuration F, M_F = 0 or M_F = lambda M E for a Pauli E on the T-inputs
        (checked when M has full row rank, as Theorem 1 assumes);
  (iii) a Z fault next to T gate j gives E = Z_j up to stabilizers of the code;
  (iv)  E(F1 F2) = E(F1) E(F2) up to phase and stabilizers when both are undetected;
  (v)   'harmless' computed directly equals 'M_F = 0, or M E|T>^t parallel to M|T>^t';
  (vi)  for circuits with rational targets (CS|++>, CCZ), F is harmless iff F + (Z at every T gate) is.
Usage: python cultivation_check.py [n_random_circuits] [seed]
"""
import sys, itertools
import numpy as np

TOL = 1e-9
W = np.exp(1j * np.pi / 4)
P1 = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
G1 = {"H": np.array([[1, 1], [1, -1]]) / np.sqrt(2), "S": np.diag([1, 1j]), "Sdg": np.diag([1, -1j]),
      "T": np.diag([1, W]), "Tdg": np.diag([1, np.conj(W)])}
TSTATE = np.array([1, W]) / np.sqrt(2)


def apply1(psi, U, q):
    psi = np.tensordot(U, psi, axes=([1], [q]))
    return np.moveaxis(psi, 0, q)


def apply_cx(psi, c, t):
    psi = psi.copy()
    idx = [slice(None)] * psi.ndim; idx[c] = 1
    sub = psi[tuple(idx)]
    tt = t - 1 if t > c else t
    sub = np.flip(sub, axis=tt).copy()
    psi[tuple(idx)] = sub
    return psi


def apply_pauli(psi, ps, qs):
    for p, q in zip(ps, qs):
        if p != "I": psi = apply1(psi, P1[p], q)
    return psi


def project(psi, ps, qs, sign):
    return (psi + sign * apply_pauli(psi, ps, qs)) / 2


def remove0(psi, q):          # ideal <0| on qubit q, the qubit stays in |0>
    psi = psi.copy()
    idx = [slice(None)] * psi.ndim; idx[q] = 1
    psi[tuple(idx)] = 0
    return psi


def locations(ops):
    """fault locations: (op index, list of elementary faults) for every noisy op"""
    locs = []
    for i, op in enumerate(ops):
        if op[0] == "rm": continue
        sup = op_support(op)
        paulis = ["".join(p) for p in itertools.product("IXYZ", repeat=len(sup)) if set(p) != {"I"}]
        locs.append((i, paulis + (["flip"] if op[0] == "proj" else [])))
    return locs


def op_support(op):
    if op[0] in ("g1", "T", "Tdg", "rm"): return [op[-1]]
    if op[0] == "cx": return [op[1], op[2]]
    if op[0] == "proj": return list(op[2])


def run(ops, nq, F, gadget=False, anc_basis=None, t_count=0, extra_Z_at_T=False):
    """state after the circuit; F maps op index -> elementary fault.
    gadget=True: T gates consume ancillas nq.. nq+t-1 prepared in the basis state anc_basis."""
    n_tot = nq + (t_count if gadget else 0)
    psi = np.zeros([2] * n_tot, dtype=complex)
    if gadget:
        psi[tuple([0] * nq + list(anc_basis))] = 1
    else:
        psi[tuple([0] * nq)] = 1
    tj = 0
    for i, op in enumerate(ops):
        f = F.get(i)
        if op[0] == "proj":
            _, ps, qs, sign = op
            if f is not None and f != "flip": psi = apply_pauli(psi, f, qs)
            psi = project(psi, ps, qs, -sign if f == "flip" else sign)
            continue
        if op[0] == "g1": psi = apply1(psi, G1[op[1]], op[2])
        elif op[0] == "cx": psi = apply_cx(psi, op[1], op[2])
        elif op[0] == "rm": psi = remove0(psi, op[1])
        elif op[0] in ("T", "Tdg"):
            q = op[1]
            if gadget:
                a = nq + tj
                psi = apply_cx(psi, q, a)
                psi = remove0(psi, a) * np.sqrt(2)
                if op[0] == "Tdg": psi = apply1(psi, G1["Sdg"], q)
            else:
                psi = apply1(psi, G1[op[0]], q)
            tj += 1
        if f is not None: psi = apply_pauli(psi, f, op_support(op))
    return psi


def output_vec(psi, nq, removed, n_tot):
    idx = tuple(0 if (q in removed or q >= nq) else slice(None) for q in range(n_tot))
    return psi[idx].reshape(-1)


def skeleton_map(ops, nq, F, t):
    removed = {op[1] for op in ops if op[0] == "rm"}
    cols = []
    for b in itertools.product([0, 1], repeat=t):
        psi = run(ops, nq, F, gadget=True, anc_basis=b, t_count=t)
        cols.append(output_vec(psi, nq, removed, nq + t))
    return np.array(cols).T


def direct_out(ops, nq, F):
    removed = {op[1] for op in ops if op[0] == "rm"}
    return output_vec(run(ops, nq, F), nq, removed, nq)


def tpow(t):
    v = np.array([1.0 + 0j])
    for _ in range(t): v = np.kron(v, TSTATE)
    return v


def find_pauli(M, MF, t):
    """return (x, z) with MF proportional to M X^x Z^z, or None; bits of b are qubit 0 most significant"""
    nb = 2 ** t
    target = np.linalg.norm(M) * np.linalg.norm(MF)
    b = np.arange(nb)
    for x in range(nb):
        g = np.sum(np.conj(M[:, b ^ x]) * MF[:, b], axis=0)      # g_b = <M[:, b^x], MF[:, b]>
        h = g.copy(); step = 1                                     # Walsh-Hadamard: h_z = sum_b (-1)^{z.b} g_b
        while step < nb:
            for i in range(0, nb, 2 * step):
                a_, c_ = h[i:i + step].copy(), h[i + step:i + 2 * step].copy()
                h[i:i + step], h[i + step:i + 2 * step] = a_ + c_, a_ - c_
            step *= 2
        z = int(np.argmax(np.abs(h)))
        if abs(abs(h[z]) - target) < 1e-7 * max(1, target): return (x, z)
    return None


def pauli_mat(x, z, t):
    U = np.array([[1.0 + 0j]])
    for k in range(t):
        xb, zb = (x >> (t - 1 - k)) & 1, (z >> (t - 1 - k)) & 1
        U = np.kron(U, np.linalg.matrix_power(P1["X"], xb) @ np.linalg.matrix_power(P1["Z"], zb))
    return U


def parallel(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < TOL: return True
    return abs(abs(np.vdot(v, u)) - nu * nv) < 1e-8 * max(1, nu * nv)


def harmless_direct(ops, nq, F, target):
    return parallel(direct_out(ops, nq, F), target)


def configs(locs, max_w, rng=None, max_pairs=None):
    singles = [{i: f} for i, fs in locs for f in fs]
    yield from singles
    if max_w >= 2:
        pairs = [(a, b) for a, b in itertools.combinations(range(len(locs)), 2)]
        allp = [(locs[a][0], fa, locs[b][0], fb) for a, b in pairs for fa in locs[a][1] for fb in locs[b][1]]
        if max_pairs and len(allp) > max_pairs:
            sel = rng.choice(len(allp), max_pairs, replace=False); allp = [allp[k] for k in sel]
        for i, fi, j, fj in allp: yield {i: fi, j: fj}


def compose(f, g):
    """product of two elementary Paulis on the same support (phases dropped)"""
    tab = {("I", p): p for p in "IXYZ"}
    tab.update({(p, "I"): p for p in "IXYZ"}); tab.update({(p, p): "I" for p in "XYZ"})
    tab.update({("X", "Y"): "Z", ("Y", "X"): "Z", ("Y", "Z"): "X", ("Z", "Y"): "X", ("Z", "X"): "Y", ("X", "Z"): "Y"})
    return "".join(tab[(a, b)] for a, b in zip(f, g))


def check_circuit(ops, nq, name, rng, max_pairs=800, rational=False):
    t = sum(op[0] in ("T", "Tdg") for op in ops)
    target = direct_out(ops, nq, {})
    if np.linalg.norm(target) < 1e-6: return None
    target = target / np.linalg.norm(target)
    M = skeleton_map(ops, nq, {}, t)
    full = np.linalg.matrix_rank(M, tol=1e-8) == M.shape[0]
    T = tpow(t)
    locs = locations(ops)
    stats = dict(name=name, t=t, full_rank=bool(full), checked=0, skipped=0, gadget_ok=0, pauli_ok=0, harm_agree=0,
                 zT_ok=0, zT_tot=0, mult_ok=0, mult_tot=0, dual_ok=0, dual_tot=0)
    Ecache = {}
    def E_of(F):
        key = tuple(sorted(F.items()))
        if key not in Ecache:
            MF = skeleton_map(ops, nq, F, t)
            Ecache[key] = (MF, None if np.linalg.norm(MF) < TOL else find_pauli(M, MF, t))
        return Ecache[key]
    tlocs = [i for i, op in enumerate(ops) if op[0] in ("T", "Tdg")]
    for F in configs(locs, 2, rng, max_pairs):
        stats["checked"] += 1
        MF, E = E_of(F)
        out = direct_out(ops, nq, F)
        stats["gadget_ok"] += np.allclose(MF @ T, out, atol=1e-9)
        hd = harmless_direct(ops, nq, F, target)
        if np.linalg.norm(MF) < TOL:
            stats["pauli_ok"] += 1; stats["harm_agree"] += hd is True
        elif full and E is not None:
            stats["pauli_ok"] += 1
            hc = parallel(M @ pauli_mat(*E, t) @ T, M @ T)
            stats["harm_agree"] += hc == hd
        elif not full:
            stats["skipped"] += 1                                       # (ii) is claimed only for full rank
        if rational:
            G = dict(F)
            for i in tlocs:
                G[i] = compose(G[i], "Z") if i in G else "Z"
                if G[i] == "I": del G[i]
            if True:
                stats["dual_tot"] += 1
                stats["dual_ok"] += harmless_direct(ops, nq, G, target) == hd
    if full:
        for j, i in enumerate(tlocs):
            MF, E = E_of({i: "Z"}); stats["zT_tot"] += 1
            Zj = 1 << (t - 1 - j)
            stats["zT_ok"] += np.linalg.norm(MF) > TOL and parallel(MF.reshape(-1), (M @ pauli_mat(0, Zj, t)).reshape(-1))
        singles = [{i: f} for i, fs in locs for f in fs if f != "flip"]
        sel = rng.choice(len(singles), min(40, len(singles)), replace=False)
        for a, b in itertools.combinations(sel, 2):
            F1, F2 = singles[a], singles[b]
            if set(F1) & set(F2): continue
            (M1, E1), (M2, E2) = E_of(F1), E_of(F2)
            if E1 is None or E2 is None: continue
            M12, _ = E_of({**F1, **F2})
            stats["mult_tot"] += 1
            pred = M @ pauli_mat(*E1, t) @ pauli_mat(*E2, t)
            stats["mult_ok"] += np.linalg.norm(M12) > TOL and parallel(M12.reshape(-1), pred.reshape(-1))
    return stats


def random_circuit(rng, nq=2, length=9, n_t=3, n_proj=2):
    kinds = ["T"] * n_t + ["P"] * n_proj + ["C"] * (length - n_t - n_proj)
    rng.shuffle(kinds)
    ops = [("g1", "H", q) for q in range(nq)]
    for k in kinds:
        if k == "T": ops.append((rng.choice(["T", "Tdg"]), int(rng.integers(nq))))
        elif k == "C":
            if rng.random() < 0.4:
                c, tq = rng.choice(nq, 2, replace=False); ops.append(("cx", int(c), int(tq)))
            else: ops.append(("g1", rng.choice(["H", "S", "Sdg"]), int(rng.integers(nq))))
        else:
            w = int(rng.integers(1, nq + 1)); qs = tuple(int(q) for q in rng.choice(nq, w, replace=False))
            ops.append(("proj", "".join(rng.choice(list("XYZ"), w)), qs, 0))
    # fix every projection sign to the more likely fault-free outcome
    for i, op in enumerate(ops):
        if op[0] == "proj":
            best = max((1, -1), key=lambda s: np.linalg.norm(run(ops[:i] + [op[:3] + (s,)], nq, {})))
            ops[i] = op[:3] + (best,)
    return ops


def cs_circuit():
    """three T states, measure ZZZ, decode: output CS|++> up to Cliffords (worked example of the distillation paper)"""
    return [("g1", "H", 0), ("g1", "H", 1), ("g1", "H", 2), ("T", 0), ("T", 1), ("T", 2),
            ("proj", "ZZZ", (0, 1, 2), 1), ("cx", 0, 2), ("cx", 1, 2), ("rm", 2)]


def ccz_circuit():
    """|+++> followed by the standard seven-T CCZ circuit (rational target |CCZ>)"""
    ops = [("g1", "H", q) for q in range(3)]
    seq = [("cx", 1, 2), ("Tdg", 2), ("cx", 0, 2), ("T", 2), ("cx", 1, 2), ("Tdg", 2), ("cx", 0, 2),
           ("T", 1), ("T", 2), ("cx", 0, 1), ("T", 0), ("Tdg", 1), ("cx", 0, 1)]
    return ops + seq


if __name__ == "__main__":
    nrand = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    circuits = [("CS|++> (rational)", cs_circuit(), 3, True), ("CCZ (rational)", ccz_circuit(), 3, True)]
    while len(circuits) < nrand + 2:
        ops = random_circuit(rng, nq=2, length=int(rng.integers(8, 12)), n_t=int(rng.integers(2, 5)),
                             n_proj=int(rng.integers(1, 3)))
        if np.linalg.norm(direct_out(ops, 2, {})) > 1e-3:
            circuits.append((f"random #{len(circuits) - 1}", ops, 2, False))
    allok = True
    for name, ops, nq, rational in circuits:
        s = check_circuit(ops, nq, name, rng, rational=rational)
        n = s["checked"] - s["skipped"]
        ok = (s["gadget_ok"] == s["checked"] and s["pauli_ok"] == n and s["harm_agree"] == n
              and s["zT_ok"] == s["zT_tot"] and s["mult_ok"] == s["mult_tot"] and s["dual_ok"] == s["dual_tot"])
        allok &= ok
        print(("OK  " if ok else "FAIL"), s, flush=True)
    print("ALL OK" if allok else "SOME CHECK FAILED")
