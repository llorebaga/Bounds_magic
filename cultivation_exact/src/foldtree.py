"""Hook-aware redesign of the GHZ fold tree in Gidney's d1=5 double check (zero added qubits).

The check pairs every data qubit with a partner (tick 76 CX partner -> data), folds one representative per pair
along a spanning tree onto a root (CX parent -> child, leaves first), measures the root (MX), resets it and
unfolds in reverse.  An X fault on tree node v during the fold/unfold spreads, after unfolding and unpairing,
to the data of (part of) v's subtree: a hook.  By the coherent-completion lemma, two hooks plus one data
fault whose X supports add up to a logical X (mod stabilizers) are malignant with real T.  Trees on the same
representatives and nearest-neighbour edges are generated, screened for such combinations, rebuilt into the
circuit and certified with the TDEM.
"""
import collections
import itertools
import random
import numpy as np
import stim

from gf2 import rref, solve


def split_ticks(c):
    T = [[]]
    for inst in c.flattened():
        if inst.name == "TICK":
            T.append([])
        else:
            T[-1].append(inst)
    return T


def cx_pairs(tick):
    out = []
    for inst in tick:
        if inst.name == "CX":
            tg = [t.value for t in inst.targets_copy()]
            out += list(zip(tg[::2], tg[1::2]))
    return out


class D5Check:
    def __init__(self, c, pair_tick=76, fold=(77, 82), meas_tick=83, unfold=(85, 90), unpair_tick=91):
        self.c = c
        self.T = split_ticks(c)
        self.coords = {q: tuple(v[:2]) for q, v in c.get_final_qubit_coordinates().items()}
        self.pairs = cx_pairs(self.T[pair_tick])                 # (partner, data)
        self.pairs0 = list(self.pairs)
        self.data_of = {}
        for a, b in self.pairs:
            self.data_of[a] = b
            self.data_of[b] = b
        self.fold_ticks = list(range(fold[0], fold[1] + 1))
        self.unfold_ticks = list(range(unfold[0], unfold[1] + 1))
        edges = [e for t in self.fold_ticks for e in cx_pairs(self.T[t])]
        self.root = [t.value for inst in self.T[meas_tick] if inst.name == "MX" for t in inst.targets_copy()][0]
        self.root0 = self.root
        self.parent = {child: par for par, child in edges}
        self.reps = sorted({x for e in edges for x in e})
        self.pair_tick, self.meas_tick, self.unpair_tick = pair_tick, meas_tick, unpair_tick
        self.pos = {v: k for k, v in enumerate(self.coords)}
        self.adj = collections.defaultdict(set)
        for a in self.reps:
            for b in self.reps:
                if a < b and abs(self.coords[a][0] - self.coords[b][0]) + abs(self.coords[a][1] - self.coords[b][1]) == 1:
                    self.adj[a].add(b)
                    self.adj[b].add(a)

    # ------------------------------------------------------------ trees and schedules
    def schedule(self, parent):
        """fold layers (lists of (parent, child)): a child is folded once its subtree is complete, a parent takes
        at most one child per layer, and nothing touches a node after it has been folded into its parent."""
        children = collections.defaultdict(list)
        for ch, par in parent.items():
            children[par].append(ch)
        pending = {v: set(children[v]) for v in self.reps}
        done = set()
        layers = []
        while len(done) < len(self.reps) - 1:
            busy = set()
            lay = []
            ready = [v for v in self.reps if v != self.root and v not in done and not pending[v]]
            # prefer the deepest-waiting parents first (critical path)
            ready.sort(key=lambda v: -self.height(v, children))
            for v in ready:
                p = parent[v]
                if v in busy or p in busy:
                    continue
                lay.append((p, v))
                busy |= {v, p}
            for p, v in lay:
                done.add(v)
                pending[p].discard(v)
            layers.append(lay)
        return layers

    def height(self, v, children, memo=None):
        if not children[v]:
            return 0
        return 1 + max(self.height(c, children) for c in children[v])

    def random_tree(self, rng, bias=None):
        """random spanning tree of the representative graph rooted at the root (randomised BFS / Prim)"""
        parent = {}
        seen = {self.root}
        frontier = [(self.root, n) for n in self.adj[self.root]]
        while len(seen) < len(self.reps):
            rng.shuffle(frontier)
            frontier.sort(key=lambda e: bias(e) if bias else 0)
            a, b = frontier.pop(0)
            if b in seen:
                continue
            parent[b] = a
            seen.add(b)
            frontier += [(b, n) for n in self.adj[b] if n not in seen]
        return parent

    # ------------------------------------------------------------ hook supports
    def hooks(self, parent, layers):
        """data X supports of single X faults on tree nodes at every fold / unfold time step"""
        children = collections.defaultdict(list)
        for ch, par in parent.items():
            children[par].append(ch)
        sub = {}

        def subtree(v):
            if v not in sub:
                s = {self.data_of[v]}
                for ch in children[v]:
                    s |= subtree(ch)
                sub[v] = frozenset(s)
            return sub[v]
        # unfold order: reverse of fold layers; X on v at unfold step s reaches children unfolded at steps > s
        unfold_layers = list(reversed(layers))
        when = {}
        for s, lay in enumerate(unfold_layers):
            for p, v in lay:
                when[v] = s
        out = set()
        for v in self.reps:
            steps = sorted({when[ch] for ch in children[v]})
            for s in [-1] + steps:
                later = [ch for ch in children[v] if when[ch] > s]
                h = {self.data_of[v]}
                for ch in later:
                    h ^= subtree(ch)
                out.add(frozenset(h))
        return out

    # ------------------------------------------------------------ circuit
    def build(self, parent, p=1e-3):
        layers = self.schedule(parent)
        nq = self.c.num_qubits
        out = stim.Circuit()
        old = new = 0
        skip = set(self.fold_ticks) | set(self.unfold_ticks)
        for t, tk in enumerate(self.T):
            if t == self.fold_ticks[0]:
                for lay in layers:
                    self._layer(out, lay, nq, p)
            if t == self.unfold_ticks[0]:
                for lay in reversed(layers):
                    self._layer(out, lay, nq, p)
            if t in skip:
                continue
            if t in (self.pair_tick, self.unpair_tick):
                tg = [x for pr in self.pairs for x in pr]
                for inst in tk:
                    if inst.name == "CX":
                        out.append("CX", tg)
                    elif inst.name == "DEPOLARIZE2":
                        out.append("DEPOLARIZE2", tg, inst.gate_args_copy())
                    else:
                        out.append(inst)
                if t < len(self.T) - 1:
                    out.append("TICK")
                continue
            for inst in tk:
                if t in (self.meas_tick, self.meas_tick + 1) and inst.name in ("MX", "RX", "DEPOLARIZE1", "Z_ERROR", "X_ERROR"):
                    tg = [self.root if x.value == self.root0 else (self.root0 if x.value == self.root else x.value)
                          for x in inst.targets_copy()]
                    out.append(inst.name, tg, inst.gate_args_copy())
                else:
                    out.append(inst)
            if t < len(self.T) - 1:
                out.append("TICK")
        return out, layers

    @staticmethod
    def _layer(out, lay, nq, p):
        tg = [x for e in lay for x in e]
        out.append("CX", tg)
        out.append("DEPOLARIZE2", tg, [p])
        out.append("DEPOLARIZE1", [q for q in range(nq) if q not in set(tg)], [p])
        out.append("TICK")


def logical_screen(hooks, data, x_stabs, logical_x, max_terms=2):
    """count combinations of <= max_terms hooks plus one data qubit whose X support is a logical X operator
    modulo the X stabilizers (necessary condition for a weight-(max_terms+1) coherent-completion failure)"""
    idx = {q: k for k, q in enumerate(data)}
    n = len(data)

    def vec(s):
        v = np.zeros(n, dtype=np.uint8)
        for q in s:
            v[idx[q]] ^= 1
        return v
    S = np.array([vec(f) for f in x_stabs], dtype=np.uint8)
    L = vec(logical_x)
    # an X-type vector is a nontrivial logical iff it is in span(S, L) but not in span(S)
    R, piv = rref(S)
    R = R[:len(piv)]

    def reduce(v):
        v = v.copy()
        for r, c in zip(R, piv):
            if v[c]:
                v ^= r
        return v
    Lr = reduce(L)
    H = [vec(h) for h in hooks]
    singles = [vec([q]) for q in data] + [np.zeros(n, dtype=np.uint8)]
    bad = 0
    examples = []
    for combo in itertools.chain([()], itertools.combinations(range(len(H)), 1), itertools.combinations(range(len(H)), 2)):
        if len(combo) > max_terms:
            continue
        base = np.zeros(n, dtype=np.uint8)
        for i in combo:
            base ^= H[i]
        for s in singles:
            v = reduce(base ^ s)
            if v.any() and np.array_equal(v, Lr):
                bad += 1
                if len(examples) < 3:
                    examples.append(([sorted(hooks_list) for hooks_list in [list(hooks)[i] for i in combo]], s.nonzero()[0].tolist()))
    return bad, examples


def min_weight_logicals(data, x_stabs, logical_x):
    """all minimum-weight X logical supports (as frozensets) of the code"""
    idx = {q: k for k, q in enumerate(data)}
    gens = [sum(1 << idx[q] for q in f) for f in x_stabs]
    span = {0}
    for g in gens:
        span |= {s ^ g for s in span}
    L = sum(1 << idx[q] for q in logical_x)
    cos = [s ^ L for s in span]
    w = min(bin(v).count("1") for v in cos)
    return w, [frozenset(data[k] for k in range(len(data)) if (v >> k) & 1) for v in cos if bin(v).count("1") == w]


def fault_supports(check, parent, layers):
    """X supports on the data (before the final T layer) of single faults: X on a tree node at each fold/unfold
    step (hooks), XX on a fold/unfold CX pair, and X on one data qubit"""
    H = check.hooks(parent, layers)
    children = collections.defaultdict(list)
    for ch, par in parent.items():
        children[par].append(ch)
    # edge faults: X on parent and child right after their CX in the unfold = hook(parent after step) + hook(child)
    out = set(H)
    unfold = list(reversed(layers))
    when = {v: s for s, lay in enumerate(unfold) for p, v in lay}
    sub = {}

    def subtree(v):
        if v not in sub:
            s = {check.data_of[v]}
            for ch in children[v]:
                s |= subtree(ch)
            sub[v] = frozenset(s)
        return sub[v]
    for s, lay in enumerate(unfold):
        for p, v in lay:
            hp = {check.data_of[p]}
            for ch in children[p]:
                if when[ch] > s:
                    hp ^= subtree(ch)
            out.add(frozenset(set(hp) ^ set(subtree(v))))
    for q in {check.data_of[v] for v in check.reps} | set(check.data_of.values()):
        out.add(frozenset([q]))
    return [s for s in out if s]


def hook_distance(supports, logicals, kmax=4):
    """fewest supports XOR-ing exactly to a minimum-weight logical support (None if > kmax)"""
    allq = sorted({q for s in supports for q in s} | {q for l in logicals for q in l})
    idx = {q: k for k, q in enumerate(allq)}
    vec = lambda s: sum(1 << idx[q] for q in s)
    S = sorted({vec(s) for s in supports})
    targets = {vec(l) for l in logicals}
    level = {0: ()}
    reach = {0}
    frontier = {0}
    for k in range(1, kmax + 1):
        new = set()
        for a in frontier:
            for s in S:
                b = a ^ s
                if b in targets:
                    return k
                if b not in reach:
                    new.add(b)
        reach |= new
        frontier = new
    return None


def regenerate_detectors(c, shots=None, seed=3):
    """replace the circuit's DETECTORs by a basis of all record parities that are deterministic in the noiseless
    S-proxy circuit (the logical observable, a final MPP product in OBSERVABLE_INCLUDE, is kept and excluded)"""
    from gf2 import nullspace, rref
    clean = stim.Circuit()
    body = stim.Circuit()
    obs = []
    nrec = 0
    for inst in c.flattened():
        if inst.name == "DETECTOR":
            continue
        if inst.name == "OBSERVABLE_INCLUDE":
            obs.append([nrec + t.value for t in inst.targets_copy()])
            continue
        body.append(inst)
        if inst.name in ("M", "MX", "MY"):
            nrec += len(inst.targets_copy())
        elif inst.name == "MPP":
            nrec += len(inst.target_groups())
    obs_recs = set()
    for o in obs:
        obs_recs ^= set(o)
    s = body.without_noise().compile_sampler(seed=seed).sample(shots or (nrec + 256)).astype(np.uint8)
    A = np.hstack([s, np.ones((len(s), 1), dtype=np.uint8)])
    B = nullspace(A)[:, :nrec]
    # drop combinations involving the logical record(s) of the final MPP
    logical = [r for r in obs_recs if r >= nrec - 64]
    keep = [row for row in B if not any(row[r] for r in logical)]
    R, piv = rref(np.array(keep, dtype=np.uint8))
    R = R[:len(piv)]
    out = body.copy()
    for row in R:
        out.append("DETECTOR", [stim.target_rec(int(r) - nrec) for r in np.nonzero(row)[0]])
    for o in obs:
        out.append("OBSERVABLE_INCLUDE", [stim.target_rec(r - nrec) for r in o], [0])
    return out


class XScreen:
    """Exact real-T harm test for X-type configurations: X^u on the data before the final transversal U layer
    (U = T or T^dag per qubit, signs s_q) and X^v after it, followed by an ideal decoder.
    U X U^dag = phase * X * (I - i s_q Z)/sqrt2 ... so the logical action is sum_{w subset u} c_w Z^w X^{u+v};
    with trivial syndrome only w in <S_Z, Z_L> survive and the output is (alpha + beta Z_L) X_L^{[u+v ~ X_L]}|T_L>."""
    def __init__(self, data, faces, signs, target_dag=True):
        self.data = list(data)
        self.idx = {q: k for k, q in enumerate(self.data)}
        n = len(self.data)
        gens = [sum(1 << self.idx[q] for q in f) for f in faces]
        span = {0}
        for g in gens:
            span |= {s ^ g for s in span}
        self.SX = span                                   # X stabilizers (self-dual: same supports as Z)
        full = (1 << n) - 1
        self.ZL = {s ^ full for s in span}               # Z_L coset (transversal Z_L = all qubits)
        self.XLc = self.ZL                               # X_L coset, same supports
        self.sign = [signs[q] for q in self.data]
        # amplitude of Z^w in prod_q (I - i s_q Z_q)/sqrt2 (global factors dropped): prod_{q in w} (-i s_q)
        T = np.exp(1j * np.pi / 4)
        self.Tvec = np.array([1, np.conj(T) if target_dag else T]) / np.sqrt(2)

    def vec(self, s):
        return sum(1 << self.idx[q] for q in s)

    def harm(self, u, v):
        """(acceptance relative, bad) for X^u before and X^v after the layer, u, v bitmasks"""
        tot = u ^ v
        if tot in self.SX:
            xl = False
        elif tot in self.XLc:
            xl = True
        else:
            return 0.0, 0.0                              # nontrivial syndrome: rejected
        alpha = beta = 0j
        for w in self.SX:
            if w & ~u == 0:
                alpha += self._c(w) * (-1) ** bin(w & v).count("1")
        for w in self.ZL:
            if w & ~u == 0:
                beta += self._c(w) * (-1) ** bin(w & v).count("1")
        nu = bin(u).count("1")
        scale = 2.0 ** (-nu)
        op = alpha * np.eye(2) + beta * np.diag([1, -1])
        if xl:
            op = op @ np.array([[0, 1], [1, 0]])
        out = op @ self.Tvec
        acc = scale * float(np.vdot(out, out).real)
        good = scale * abs(np.vdot(self.Tvec, out)) ** 2
        return acc, max(acc - good, 0.0)

    def _c(self, w):
        c = 1 + 0j
        k = 0
        while w:
            if w & 1:
                c *= -1j * self.sign[k]
            w >>= 1
            k += 1
        return c


def screen_fd(check, parent, layers, xs, kmax=4):
    """smallest number of single faults (hooks, CX-pair hooks, data X before or after the layer) whose X parts
    give an accepted harmful output by XScreen; returns (k, count at k)"""
    pre = [xs.vec(s) for s in fault_supports(check, parent, layers)]
    pre = sorted(set(pre))
    singles = [1 << k for k in range(len(xs.data))]
    items = [(p, 0) for p in pre] + [(0, s) for s in singles]
    items = sorted(set(items))
    level = {(0, 0)}
    for k in range(1, kmax + 1):
        new = set()
        cnt = 0
        for (u, v) in level:
            for (a, b) in items:
                u2, v2 = u ^ a, v ^ b
                if (u2, v2) in new:
                    continue
                new.add((u2, v2))
        for (u2, v2) in new:
            acc, bad = xs.harm(u2, v2)
            if bad > 1e-12:
                cnt += 1
        if cnt:
            return k, cnt
        level = new
    return None, 0


def set_pairing(check, pairs):
    """use another nearest-neighbour (partner, data) matching for the pair / unpair layers"""
    check.pairs = [tuple(p) for p in pairs]
    check.data_of = {}
    for a, b in check.pairs:
        check.data_of[a] = b
        check.data_of[b] = b


def random_pairing(check, rng):
    partners = sorted({a for a, b in check.pairs0})
    data = sorted({b for a, b in check.pairs0})
    nb = lambda x, y: abs(check.coords[x][0] - check.coords[y][0]) + abs(check.coords[x][1] - check.coords[y][1]) == 1
    for _ in range(1000):
        used, out = set(), []
        ok = True
        for d in sorted(data, key=lambda d: (sum(nb(d, p) for p in partners), rng.random())):
            opts = [p for p in partners if nb(d, p) and p not in used]
            if not opts:
                ok = False
                break
            p = rng.choice(opts)
            used.add(p)
            out.append((p, d))
        if ok:
            return out
    return list(check.pairs0)


def merge_final_mpps(c):
    """merge the trailing run of MPP instructions (separated only by TICK / annotations) into one MPP"""
    insts = list(c.flattened())
    last_gate = max(i for i, x in enumerate(insts) if x.name not in ("MPP", "TICK", "DETECTOR", "OBSERVABLE_INCLUDE",
                                                                   "SHIFT_COORDS", "QUBIT_COORDS"))
    tail = insts[last_gate + 1:]
    mpps = [x for x in tail if x.name == "MPP"]
    ann = [x for x in tail if x.name in ("DETECTOR", "OBSERVABLE_INCLUDE")]
    out = stim.Circuit()
    for x in insts[:last_gate + 1]:
        out.append(x)
    out.append("TICK")
    targets = []
    for x in mpps:
        targets += x.targets_copy()
    out.append("MPP", targets)
    # annotations refer to records counted from the end; they were all after the MPPs in program order
    first_ann = min(i for i, x in enumerate(tail) if x.name in ("DETECTOR", "OBSERVABLE_INCLUDE"))
    assert all(x.name != "MPP" for x in tail[first_ann:]), "annotations interleaved with final MPPs"
    for x in ann:
        out.append(x)
    return out
