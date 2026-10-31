"""Circuit-level Clifford-proxy circuits for d=3 colour-code (Steane) cultivation, built with one shared gadget set.

Two protocols, identical building blocks:
  "T"  : one block.  Logical state |T>_L, check g = T X T^dag (Gidney's H_XY up to Clifford frame).
  "D"  : three blocks. Logical state D|+++>_L with D = CS12 CS13 (Clifford-equivalent to |CCZ>, Prop. 3 of our
         paper); checks g_b = D X_b D^dag, b = 1,2,3, all measured in one round.
Schedule:  noisy injection on bare carrier qubits + non-FT encoder  ->  stabiliser round  ->  check round
           ->  stabiliser round  ->  check round  ->  stabiliser round  ->  noiseless readout (MPP).
A check round = U^dag layer (transversal, phase polynomial) -> verified-cat X-parity of each block -> U layer.
Clifford proxy (as in Gidney-Shutty-Jones): every T is replaced by S (CS by CZ), so the whole circuit is Clifford
and the proxy logical state is |+i>_L (resp. the graph state CZ12CZ13|+++>_L).  Every detector is postselected.

Noise ("uniform"): DEPOLARIZE1(p) after 1q gates, DEPOLARIZE2(p) after CX, X/Z_ERROR(p) after resets and before
measurements, and (if idle=True) DEPOLARIZE1(p) on every live qubit idle in a moment.  The final MPP readout is
noiseless (the 'ungrown' convention: no escape stage).
"""
import stim

HAM = [[0, 2, 4, 6], [0, 1, 4, 5], [0, 1, 2, 3]]     # Steane X- and Z-stabiliser supports
PIVOT = [6, 5, 3]                                     # qubit unique to each generator
LOGX = [1, 2, 4]                                      # weight-3 logical X, carrier = qubit 1
CARRIER = 1


class Builder:
    def __init__(self, p, idle, model="uniform"):
        self.p, self.idle, self.model = p, idle, model
        self.ops = []              # (name, targets, arg) in program order
        self.nq = 0
        self.meas = []             # measurement keys in program order
        self.detectors = []        # lists of measurement keys
        self.observables = []      # list of lists of measurement keys

    def alloc(self, k):
        r = list(range(self.nq, self.nq + k)); self.nq += k; return r

    def g(self, name, *qs, key=None):
        self.ops.append((name, qs, key))

    # ---- scheduling into moments (ASAP, per-qubit program order) and noise insertion ----
    def build(self):
        free = {}
        moments = []
        for op in self.ops:
            name, qs, key = op
            if name in ("MPP",):
                t = max([max(free.values(), default=0)])      # barrier: noiseless readout at the very end
            else:
                t = max([free.get(q, 0) for q in qs] + [0])
            while len(moments) <= t: moments.append([])
            moments[t].append(op)
            for q in (qs if name != "MPP" else range(self.nq)):
                free[q] = t + 1
        c = stim.Circuit(); p = self.p
        # rates: uniform = every location p; si1000 = Gidney-Newman-McEwen superconducting-inspired model
        # (2q p, 1q and idle p/10, reset 2p, measurement 5p, idle during a measurement/reset moment 2p);
        # si1000_T0 is si1000 with noiseless (virtual) T gates, as on Google's hardware
        si = self.model.startswith("si1000")
        r_res, r_meas, r_1q, r_2q, r_idle, r_heavy = (2 * p, 5 * p, p / 10, p, p / 10, 2 * p) if si else (p,) * 6
        r_T = 0.0 if self.model == "si1000_T0" else r_1q
        live = set(); keys = []
        for mom in moments:
            touched = set()
            post1, postT, post2 = [], [], []
            heavy = any(op[0] in ("R", "RX", "M", "MX") for op in mom)
            for name, qs, key in mom:
                touched.update(qs if name != "MPP" else [])
                if name in ("R", "RX"):
                    c.append(name, qs); c.append("X_ERROR" if name == "R" else "Z_ERROR", qs, r_res); live.update(qs)
                elif name in ("M", "MX"):
                    c.append("X_ERROR" if name == "M" else "Z_ERROR", qs, r_meas); c.append(name, qs)
                    keys.append(key); live.difference_update(qs)
                elif name == "CX":
                    c.append("CX", qs); post2 += list(qs)
                elif name in ("S", "S_DAG"):
                    c.append(name, qs); postT += list(qs)
                elif name in ("Z", "H"):
                    c.append(name, qs); post1 += list(qs)
                elif name == "MPP":
                    c.append("MPP", qs[0][0]); keys.append(key)
            if post1: c.append("DEPOLARIZE1", post1, r_1q)
            if postT and r_T > 0: c.append("DEPOLARIZE1", postT, r_T)
            if post2: c.append("DEPOLARIZE2", post2, r_2q)
            if self.idle:
                idle = sorted(live - touched)
                if idle and not any(op[0] == "MPP" for op in mom):
                    c.append("DEPOLARIZE1", idle, r_heavy if heavy else r_idle)
            c.append("TICK")
        pos = {k: i for i, k in enumerate(keys)}
        n = len(keys)
        for det in self.detectors:
            c.append("DETECTOR", [stim.target_rec(pos[k] - n) for k in det])
        for i, obs in enumerate(self.observables):
            c.append("OBSERVABLE_INCLUDE", [stim.target_rec(pos[k] - n) for k in obs], i)
        return c


def make(protocol, p, idle=True, rounds=2):
    B = Builder(p, idle)
    r = 1 if protocol == "T" else 3
    data = [B.alloc(7) for _ in range(r)]
    mcount = [0]
    def key(tag):
        mcount[0] += 1; return (tag, mcount[0])

    # ---- U layers (phase polynomials), proxy-doubled: T -> S, CS -> CZ ----
    def U_layer(dag, qubits_per_pos):
        # qubits_per_pos: list over positions j of tuples of qubits (one per block)
        for qs in qubits_per_pos:
            if protocol == "T":
                seq = [("S", qs[0])]
            elif protocol == "D":
                q1, q2, q3 = qs
                # D = CS12 CS13 : phase 2x1 + x2 + x3 - (x1^x2) - (x1^x3)  (units pi/4); proxy doubles each T to S
                seq = [("Z", q1), ("S", q2), ("S", q3),
                       ("CX", q1, q2), ("S_DAG", q2), ("CX", q1, q2),
                       ("CX", q1, q3), ("S_DAG", q3), ("CX", q1, q3)]
            else:   # "C": CCZ, phase x1+x2+x3-(x12)-(x13)-(x23)+(x123); Gray-code network, 7 T and 6 CX
                q1, q2, q3 = qs
                seq = [("S", q1), ("S", q2), ("S", q3),
                       ("CX", q1, q2), ("S_DAG", q2), ("CX", q1, q2),
                       ("CX", q1, q3), ("S_DAG", q3), ("CX", q2, q3), ("S", q3),
                       ("CX", q1, q3), ("S_DAG", q3), ("CX", q2, q3)]
            if dag:
                inv = {"S": "S_DAG", "S_DAG": "S", "Z": "Z", "CX": "CX"}
                seq = [(inv[g[0]],) + tuple(g[1:]) for g in reversed(seq)]
            for g in seq: B.g(*g)
    positions = [tuple(data[b][j] for b in range(r)) for j in range(7)]

    # ---- injection: carriers in |+>, apply U (noisy, unencoded), then non-FT encoder ----
    for b in range(r):
        B.g("RX", data[b][CARRIER])
        for j in range(7):
            if j != CARRIER: B.g("RX" if j in PIVOT else "R", data[b][j])
    U_layer(False, [tuple(data[b][CARRIER] for b in range(r))])
    for b in range(r):
        d = data[b]
        for j in LOGX:
            if j != CARRIER: B.g("CX", d[CARRIER], d[j])
        for gen, pv in zip(HAM, PIVOT):
            for j in gen:
                if j != pv: B.g("CX", d[pv], d[j])

    # ---- gadgets ----
    def stabiliser_round():
        for b in range(r):
            d = data[b]
            anc = B.alloc(6); flg = B.alloc(6)
            # Chao-Reichardt flags: in the Steane code any weight-2 hook plus one more fault can be a logical,
            # so each weight-4 measurement carries a flag coupled after the first and before the last data CX.
            for i, gen in enumerate(HAM):          # X-type: ancilla |+>, X hooks caught by a Z-basis flag
                a, f = anc[i], flg[i]; B.g("RX", a); B.g("R", f)
                B.g("CX", a, d[gen[0]]); B.g("CX", a, f)
                B.g("CX", a, d[gen[1]]); B.g("CX", a, d[gen[2]])
                B.g("CX", a, f); B.g("CX", a, d[gen[3]])
                k = key("sx"); B.g("MX", a, key=k); B.detectors.append([k])
                k = key("fx"); B.g("M", f, key=k); B.detectors.append([k])
            for i, gen in enumerate(HAM):          # Z-type: ancilla |0>, Z hooks caught by an X-basis flag
                a, f = anc[3 + i], flg[3 + i]; B.g("R", a); B.g("RX", f)
                B.g("CX", d[gen[0]], a); B.g("CX", f, a)
                B.g("CX", d[gen[1]], a); B.g("CX", d[gen[2]], a)
                B.g("CX", f, a); B.g("CX", d[gen[3]], a)
                k = key("sz"); B.g("M", a, key=k); B.detectors.append([k])
                k = key("fz"); B.g("MX", f, key=k); B.detectors.append([k])

    def check_round():
        U_layer(True, positions)
        for b in range(r):
            d = data[b]
            cat = B.alloc(7); ver = B.alloc(6)
            B.g("RX", cat[0])
            for q in cat[1:]: B.g("R", q)
            for (c0, c1) in [(0, 1), (0, 2), (1, 3), (0, 4), (1, 5), (2, 6)]:
                B.g("CX", cat[c0], cat[c1])
            for i in range(6):                     # verify cat: Z_i Z_{i+1}
                B.g("R", ver[i]); B.g("CX", cat[i], ver[i]); B.g("CX", cat[i + 1], ver[i])
                k = key("ver"); B.g("M", ver[i], key=k); B.detectors.append([k])
            for j in range(7): B.g("CX", cat[j], d[j])
            ks = []
            for j in range(7):
                k = key("cat"); B.g("MX", cat[j], key=k); ks.append(k)
            B.detectors.append(ks)                 # check outcome
        U_layer(False, positions)

    stabiliser_round()
    for _ in range(rounds):
        check_round()
        stabiliser_round()

    # ---- noiseless readout: stabilisers (postselected) and the cultivated state's stabilisers (observables) ----
    for b in range(r):
        d = data[b]
        for gen in HAM:
            for P in "XZ":
                k = key("final"); B.g("MPP", [stim.target_combined_paulis([stim.target_x(d[j]) if P == "X" else stim.target_z(d[j]) for j in gen])], key=k)
                B.detectors.append([k])
    if protocol == "T":
        obs = [[stim.target_y(q) for q in data[0]]]                       # (S X S^dag)^{x7} = Y^{x7}
    elif protocol == "C":   # proxy CCZ -> identity: state |+++>_L, checks X_b^{x7}
        obs = [[stim.target_x(data[b][j]) for j in range(7)] for b in range(3)]
    else:   # (CZ12 CZ13 X_b CZ12 CZ13)^{x7}: X1 Z2 Z3, Z1 X2, Z1 X3 transversally
        obs = [[stim.target_x(data[0][j]) for j in range(7)] + [stim.target_z(data[1][j]) for j in range(7)] + [stim.target_z(data[2][j]) for j in range(7)],
               [stim.target_z(data[0][j]) for j in range(7)] + [stim.target_x(data[1][j]) for j in range(7)],
               [stim.target_z(data[0][j]) for j in range(7)] + [stim.target_x(data[2][j]) for j in range(7)]]
    for o in obs:
        k = key("obs"); B.g("MPP", [stim.target_combined_paulis(o)], key=k); B.observables.append([k])
    return B.build()


if __name__ == "__main__":
    for prot in ("T", "D", "C"):
        c = make(prot, 1e-3)
        print(prot, "qubits", c.num_qubits, "detectors", c.num_detectors, "observables", c.num_observables,
              "ticks", c.num_ticks)
        c.detector_error_model()      # raises if any detector/observable is non-deterministic
        print("  deterministic: ok")
