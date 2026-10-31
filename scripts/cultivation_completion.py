"""State-vector checks of Lemma 4 (coherent completion) of cultivation.tex and its two neighboring cases.

On the [[7,1,3]] color code, for every T/T-dagger pattern whose transversal check
fixes the encoded T state and every minimum-weight logical support l:
- U X^l U^dag acts on the code as the check operator;
- X on u (|u| = 1, 2) before the last layer, completed by X after it: acceptance
  2^-|u|, fidelity 1/2; completed by Y on the last qubit: the same for |u| = 2,
  rejected for |u| = 1;
- X^l Z^y before the layer: acceptance 1/4, fidelity 1 for even |y| and 0 for odd |y|.
Prints ALL OK.
"""
import itertools
import numpy as np

w = np.exp(1j * np.pi / 4)
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
Y = 1j * X @ Z
T = np.diag([1, w])
Td = T.conj().T


def kron(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def op_on(n, single, qubits):
    return kron([single if q in qubits else I2 for q in range(n)])


# ---------------------------------------------------------------- Steane / [[7,1,3]] color code
n = 7
faces = [{0, 1, 2, 3}, {1, 2, 4, 5}, {2, 3, 5, 6}]
P = np.eye(2**n, dtype=complex)
for f in faces:
    P = P @ (np.eye(2**n) + op_on(n, X, f)) / 2 @ (np.eye(2**n) + op_on(n, Z, f)) / 2
zero = np.zeros(2**n, complex); zero[0] = 1
L0 = P @ zero; L0 /= np.linalg.norm(L0)
XL = op_on(n, X, set(range(n))); ZL = op_on(n, Z, set(range(n)))
L1 = XL @ L0
Tbar = (L0 + w * L1) / np.sqrt(2)
ell = [0, 1, 4]
for f in faces:
    assert len(f & set(ell)) % 2 == 0
assert np.allclose(op_on(n, X, set(ell)) @ Tbar, XL @ Tbar)


def run(pattern, u, completion):
    """pattern: tuple of +1 (T) / -1 (T^dag) per qubit for the last layer U.
    u: X error before the layer. completion: dict qubit->Pauli applied after."""
    U = kron([T if s > 0 else Td for s in pattern])
    psi = U.conj().T @ Tbar          # fault-free state before the last layer
    phi = U @ op_on(n, X, set(u)) @ psi
    for q, pa in completion.items():
        phi = op_on(n, pa, {q}) @ phi
    acc_vec = P @ phi
    acc = np.vdot(acc_vec, acc_vec).real
    fid = abs(np.vdot(Tbar, acc_vec))**2 / acc if acc > 1e-12 else float('nan')
    return acc, fid


Tperp = (L0 - w * L1) / np.sqrt(2)
check = np.outer(Tbar, Tbar.conj()) - np.outer(Tperp, Tperp.conj())


def restricted(A):
    return np.array([[np.vdot(a, A @ b) for b in (L0, L1)] for a in (L0, L1)])


def run2(pattern, xs, zs, after=None):
    """X on xs and Z on zs before the last layer U, then Paulis `after`."""
    U = kron([T if s > 0 else Td for s in pattern])
    phi = U @ op_on(n, Z, set(zs)) @ op_on(n, X, set(xs)) @ U.conj().T @ Tbar
    for q, pa in (after or {}).items():
        phi = op_on(n, pa, {q}) @ phi
    v = P @ phi
    acc = np.vdot(v, v).real
    return acc, (abs(np.vdot(Tbar, v))**2 / acc if acc > 1e-12 else None)


ells = [c for c in itertools.combinations(range(n), 3) if all(len(set(c) & f) % 2 == 0 for f in faces)]
Cr = restricted(check)
ok = True
for pat in itertools.product([1, -1], repeat=n):
    U = kron([T if s > 0 else Td for s in pat])
    B = restricted(P @ U @ XL @ U.conj().T @ P)
    k = np.vdot(Cr.flatten(), B.flatten()) / np.vdot(Cr.flatten(), Cr.flatten())
    if abs(k) < 1e-9 or not np.allclose(B, k * Cr):
        continue  # this pattern does not check the encoded T state
    for l in ells:
        l = list(l)
        Bl = restricted(P @ U @ op_on(n, X, set(l)) @ U.conj().T @ P)
        kl = np.vdot(Cr.flatten(), Bl.flatten()) / np.vdot(Cr.flatten(), Cr.flatten())
        ok &= abs(kl) > 1e-9 and np.allclose(Bl, kl * Cr)   # U X^l U^dag acts as the check
        for size in (1, 2):
            for u in itertools.combinations(l, size):
                rest = [q for q in l if q not in u]
                acc, fid = run2(pat, u, [], {q: X for q in rest})
                ok &= np.isclose(acc, 2.0**-size) and np.isclose(fid, 0.5)   # Lemma 4
                acc, fid = run2(pat, u, [], {**{q: X for q in rest[:-1]}, rest[-1]: Y})
                ok &= (np.isclose(acc, 2.0**-size) and np.isclose(fid, 0.5)) if size == 2 else np.isclose(acc, 0)
        for r in range(4):
            for y in itertools.combinations(l, r):
                acc, fid = run2(pat, l, y)
                ok &= np.isclose(acc, 0.25) and np.isclose(fid, 1.0 if r % 2 == 0 else 0.0)
print("ALL OK" if ok else "FAILED")
