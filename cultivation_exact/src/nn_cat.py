"""Nearest-neighbour verified-cat checks on a square grid.

Every two-qubit gate acts on grid neighbours (asserted). The check of a colour code with n data qubits uses a
cat path P = (p_0, ..., p_{m-1}) of grid sites: each data qubit q is a grid neighbour of its own path site
cat[q]; the other path sites are bridges (GHZ members that touch no data). The GHZ state is grown along P from
a root p_r outward, so a single X fault on p_i spreads to the suffix of P beyond p_i (towards the nearer end)
or stays on p_i. Verification measures Z_u Z_v for given pairs (u, v) of path sites through an ancilla that is a
grid neighbour of both; with the pair (p_0, p_{m-1}) every suffix that carries more than one X is detected, which
is the one-fault condition of a verified cat. Each data qubit then interacts with the check once
(CX cat[q] -> q), and the cat is read out in the X basis.

Protocol (the convention of src/vcat.py): ideal encoded input -> [stabiliser round] -> (check round ->
[stabiliser round]) x rounds -> noiseless MPP readout of all stabilisers and the logical Y."""
import sys

import stim

sys.path.insert(0, __file__.rsplit("nn_cat.py", 1)[0])
from cult_circuits import Builder          # noqa: E402
from tdem import sublattice_signs           # noqa: E402
from vcat import encoder_plan               # noqa: E402


def adjacent(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


class Grid:
    """qubit allocation by grid site; every CX is checked for adjacency"""

    def __init__(self, B):
        self.B, self.site, self.q = B, {}, {}

    def at(self, s):
        if s not in self.q:
            q = self.B.alloc(1)[0]
            self.q[s], self.site[q] = q, s
        return self.q[s]

    def cx(self, a, b):
        assert adjacent(self.site[a], self.site[b]), (self.site[a], self.site[b])
        self.B.g("CX", a, b)


def make_nn_cat(faces, n, data_sites, path, cat_of, root, ver_pairs, ver_sites, p=1e-3, rounds=2,
                inject_dag=False, stab=None, model="uniform", idle=True, ver_rounds=1, inject="ideal", zrounds=0,
                zbetween=0, zkind="shor", unfold=False, schedule=None):
    """data_sites[q]: site of data q; path: list of sites; cat_of[q]: index into path; root: index into path;
    ver_pairs: list of (i, j) path indices; ver_sites: per pair, an ancilla site that neighbours both, or a pair
    of adjacent sites (a1, a2) with a1 ~ path[i] and a2 ~ path[j]; ver_rounds: repetitions of the verification;
    stab: None (no stabiliser rounds) or a function(G, d, key) emitting one stabiliser round;
    inject: "ideal" (noiseless encoded input) or "routed" (noisy injection and CNOT encoder whose data-data CNOTs
    are relayed along the path); zrounds / zbetween: Z-stabiliser rounds through the path after the injection /
    after each check but the last; zkind: "shor" (each X_f and Z_f with the verified path cat) or "fold"
    (Z_f only, folded along the path); unfold: un-grow every cat onto its root after the coupling;
    schedule: with inject="measure", a string over {X, Z, C}: X_f round, Z_f round, check (overrides rounds)"""
    B = Builder(p, idle, model)
    G = Grid(B)
    d = [G.at(s) for s in data_sites]
    for q in range(n):
        assert adjacent(data_sites[q], path[cat_of[q]]), q
    assert len(set(cat_of)) == n
    mcount = [0]

    def key(tag):
        mcount[0] += 1
        return (tag, mcount[0])

    signs = sublattice_signs(n, faces)
    gens, piv, Lrow, carrier = encoder_plan(n, faces, list(range(n)))

    def U_layer(dag):
        for q in range(n):
            B.g("S" if (signs[q] > 0) != dag else "S_DAG", d[q])

    def check_round():
        U_layer(True)
        cat_measure(range(n), "X")
        U_layer(False)

    def cat_measure(support, ptype):
        """measure the X- (or Z-) type product on the data in `support` with the verified path cat: each of
        those data qubits gets one CX from its path site (conjugated by H on the data for Z type)"""
        P = [G.at(s) for s in path]
        m = len(P)
        B.g("RX", P[root])
        for i in range(m):
            if i != root:
                B.g("R", P[i])
        # grow outward from the root, one step per side per layer
        left, right = root, root
        grow = []
        while left > 0 or right < m - 1:
            if left > 0:
                grow.append((left, left - 1)); left -= 1
            if right < m - 1:
                grow.append((right, right + 1)); right += 1
        for a, b in grow:
            G.cx(P[a], P[b])
        for _ in range(ver_rounds):
            for (i, j), s in zip(ver_pairs, ver_sites):
                if isinstance(s[0], int):                 # one ancilla, a grid neighbour of both
                    a = G.at(s)
                    B.g("R", a)
                    G.cx(P[i], a)
                    G.cx(P[j], a)
                    k = key("ver"); B.g("M", a, key=k); B.detectors.append([k])
                else:                                     # two ancillas a1 ~ P[i], a2 ~ P[j], a1 ~ a2
                    a1, a2 = G.at(s[0]), G.at(s[1])
                    B.g("R", a1); B.g("R", a2)
                    G.cx(P[i], a1)
                    G.cx(P[j], a2)
                    G.cx(a2, a1)
                    G.cx(P[j], a2)                        # uncompute a2 (it returns to |0>)
                    k = key("ver"); B.g("M", a1, key=k); B.detectors.append([k])
                    k = key("ver2"); B.g("M", a2, key=k); B.detectors.append([k])
        for q in support:
            if ptype == "Z":
                B.g("H", d[q])
            G.cx(P[cat_of[q]], d[q])
            if ptype == "Z":
                B.g("H", d[q])
        if unfold:
            # un-grow the cat onto its root: the root holds the outcome (one X measurement), every other path
            # qubit returns to |0> (Z measurements, deterministic); faults here touch no data
            for a, b in reversed(grow):
                G.cx(P[a], P[b])
            k = key("cat"); B.g("MX", P[root], key=k); B.detectors.append([k])
            for i in range(m):
                if i != root:
                    k = key("catz"); B.g("M", P[i], key=k); B.detectors.append([k])
        else:
            ks = []
            for i in range(m):
                k = key("cat"); B.g("MX", P[i], key=k); ks.append(k)
            B.detectors.append(ks)

    def shor_round():
        for f in faces:
            cat_measure(f, "Z")
        for f in faces:
            cat_measure(f, "X")

    def foldx_round():
        """Z_f folded along the path (its hooks are Z errors), then X_f with the verified cat (one X per fault),
        which catches those Z hooks"""
        z_round()
        for f in faces:
            cat_measure(f, "X")

    def x_round():
        for f in faces:
            cat_measure(f, "X")

    stab_kinds = {"shor": lambda: shor_round(), "fold": lambda: z_round(), "foldx": lambda: foldx_round(),
                  "x": lambda: x_round()}

    P_all = [G.at(s) for s in path]

    def relay(x, sources, sinks):
        """apply CX(x -> t) for all data t in sinks, x a data qubit, via the path segment covering their sites:
        copy x onto its site, spread along the segment, CX onto the targets, unspread, uncopy, measure the
        segment (deterministically 0)"""
        idx = [cat_of[x]] + [cat_of[t] for t in sinks]
        lo, hi = min(idx), max(idx)
        seg = list(range(lo, hi + 1))
        for i in seg:
            B.g("R", P_all[i])
        c0 = cat_of[x]
        G.cx(d[x], P_all[c0])
        chain = []
        l, r = c0, c0
        while l > lo or r < hi:
            if l > lo:
                chain.append((l, l - 1)); l -= 1
            if r < hi:
                chain.append((r, r + 1)); r += 1
        for a, b in chain:
            G.cx(P_all[a], P_all[b])
        for t in sinks:
            G.cx(P_all[cat_of[t]], d[t])
        for a, b in reversed(chain):
            G.cx(P_all[a], P_all[b])
        G.cx(d[x], P_all[c0])
        for i in seg:
            k = key("relay"); B.g("M", P_all[i], key=k); B.detectors.append([k])

    def z_round():
        """Z_f for every face: CX(q -> its site), fold the segment parity onto its last site, measure it,
        unfold, uncopy, and measure the segment (deterministically 0)"""
        for f in faces:
            idx = sorted(cat_of[q] for q in f)
            lo, hi = idx[0], idx[-1]
            seg = list(range(lo, hi + 1))
            for i in seg:
                B.g("R", P_all[i])
            for q in f:
                G.cx(d[q], P_all[cat_of[q]])
            for i in seg[:-1]:
                G.cx(P_all[i], P_all[i + 1])
            k = key("zf"); B.g("M", P_all[hi], key=k); B.detectors.append([k])
            B.g("R", P_all[hi])
            for i in reversed(seg[:-2]):
                G.cx(P_all[i], P_all[i + 1])
            for q in f:
                if cat_of[q] != hi:
                    G.cx(d[q], P_all[cat_of[q]])
            for i in seg:
                k = key("zclr"); B.g("M", P_all[i], key=k); B.detectors.append([k])

    if inject == "measure":
        # |0>_L by measuring every X_f on |0>^n with the verified path cat (random outcomes = Z frames, which
        # commute with T); the first check then projects onto the T-type eigenstate (a -1 outcome is a Z_L frame)
        B.g("R", *d)
        if schedule:
            for tok in schedule:
                if tok == "C":
                    check_round()
                else:
                    for f in faces:
                        cat_measure(f, tok)
            rounds = 0
        else:
            for _ in range(max(zrounds, 1)):
                for f in faces:
                    cat_measure(f, "X")
    if inject == "routed":
        B.g("R", *d)
        for q in [carrier] + list(piv):
            B.g("H", d[q])
        B.g("S_DAG" if inject_dag else "S", d[carrier])
        relay(carrier, None, [q for q in Lrow if q != carrier])
        for g, pv in zip(gens, piv):
            relay(pv, None, [q for q in g if q != pv])
        for _ in range(zrounds):
            stab_kinds[zkind]()
    if stab:
        stab(G, d, key)
    for r_ in range(rounds):
        check_round()
        if stab:
            stab(G, d, key)
        if r_ < rounds - 1:
            for _ in range(zbetween):
                stab_kinds[zkind]()
    for f in faces:
        for Pn in "XZ":
            k = key("final")
            B.g("MPP", [stim.target_combined_paulis([stim.target_x(d[q]) if Pn == "X" else stim.target_z(d[q])
                                                     for q in f])], key=k)
            B.detectors.append([k])
    k = key("obs")
    B.g("MPP", [stim.target_combined_paulis([stim.target_y(q) for q in d])], key=k)
    B.observables.append([k])
    body = B.build()
    pre = stim.Circuit()
    for q, s in sorted(G.site.items()):
        pre.append("QUBIT_COORDS", [q], list(s))
    if inject in ("routed", "measure"):
        pre.append("TICK")
        out = pre.copy()
        for inst in body.flattened():
            out.append(inst)
        return out, G
    pre.append("R", d)
    pre.append("H", [d[carrier]] + [d[q] for q in piv])
    pre.append("S_DAG" if inject_dag else "S", [d[carrier]])
    for q in Lrow:
        if q != carrier:
            pre.append("CX", [d[carrier], d[q]])
    for g, pv in zip(gens, piv):
        for q in g:
            if q != pv:
                pre.append("CX", [d[pv], d[q]])
    pre.append("TICK")
    out = pre.copy()
    for inst in body.flattened():
        out.append(inst)
    return out, G


# ---- d = 3: Steane code, cat path around a 4 x 3 rectangle, verification ancilla closing the ring ----
STEANE = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
RING3 = dict(
    path=[(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (2, 2), (1, 2), (0, 2)],
    data_sites=[(-1, 0), (1, -1), (2, -1), (4, 1), (2, 3), (1, 3), (-1, 2)],
    cat_of=[0, 1, 2, 4, 6, 7, 8],
    root=4,
    ver_pairs=[(0, 8)],
    ver_sites=[(0, 1)],
)


def ladder(L, diagonals=True):
    """d = 5 (19 data, L = 8): U-shaped cat path on rows y = 0 (top, x = 0..L-1), the turn (L, 0), (L, 1), (L, 2),
    and y = 2 (bottom, x = L-1..0); the root is the turn centre, so the cat grows towards both ends. Data sit
    above the top row, below the bottom row and right of the turn. Row y = 1 holds the verification ancillas:
    rungs Z(x,0) Z(x,2) through (x,1), and diagonals Z(x,0) Z(x-1,2) through (x,1), (x-1,1), which break the
    top-bottom symmetry that rungs alone cannot see (equal suffixes grown on both sides)."""
    top = [(x, 0) for x in range(L)]
    turn = [(L, 0), (L, 1), (L, 2)]
    bot = [(x, 2) for x in range(L - 1, -1, -1)]
    path = top + turn + bot
    pos = {s: i for i, s in enumerate(path)}
    data_sites = [(x, -1) for x in range(L)] + [(L + 1, 0), (L + 1, 1), (L + 1, 2)] +                  [(x, 3) for x in range(L - 1, -1, -1)]
    cat_of = list(range(len(path)))
    ver_pairs, ver_sites = [], []
    for x in range(L):
        ver_pairs.append((pos[(x, 0)], pos[(x, 2)])); ver_sites.append((x, 1))
    if diagonals:
        # odd x first, then even x: diagonals that share no ancilla run in parallel
        for x in [x for x in range(1, L) if x % 2] + [x for x in range(1, L) if not x % 2]:
            ver_pairs.append((pos[(x, 0)], pos[(x - 1, 2)])); ver_sites.append(((x, 1), (x - 1, 1)))
    return dict(path=path, data_sites=data_sites, cat_of=cat_of, root=pos[(L, 1)], ver_pairs=ver_pairs,
                ver_sites=ver_sites)
