"""Fault distance and leading-order failure terms from a TDEM, by meet-in-the-middle on detector masks."""
import collections
import itertools


def fault_distance(m, kmax, mech=None, verbose=True, collect=True):
    """Smallest k <= kmax such that k mechanisms with zero total detector mask give a harmful output.
    Returns (k or None, list of (indices, o, acc, bad, prob)) for that k."""
    mech = mech if mech is not None else m.mechanisms()
    mech = [x for x in mech if x[1] or x[2]]
    n = len(mech)
    byD = collections.defaultdict(list)
    for i, (p, D, o) in enumerate(mech):
        byD[D].append(i)

    def record(idx, o, out):
        acc, bad = m.evaluate(o)
        if bad > 1e-9:
            prob = 1.0
            for i in idx:
                prob *= mech[i][0]
            out.append((idx, o, acc, bad, prob))
            return True
        return False

    for k in range(1, kmax + 1):
        bad_list = []
        if k == 1:
            for i in byD.get(0, []):
                record((i,), mech[i][2], bad_list)
        elif k == 2:
            for D, idx in byD.items():
                for i, j in itertools.combinations(idx, 2):
                    record((i, j), mech[i][2] ^ mech[j][2], bad_list)
        elif k == 3:
            for i in range(n):
                Di, oi = mech[i][1], mech[i][2]
                for j in range(i + 1, n):
                    Dij = Di ^ mech[j][1]
                    for l in byD.get(Dij, ()):
                        if l > j:
                            record((i, j, l), oi ^ mech[j][2] ^ mech[l][2], bad_list)
        elif k == 4:
            pairs = collections.defaultdict(list)
            for i in range(n):
                Di = mech[i][1]
                for j in range(i + 1, n):
                    pairs[Di ^ mech[j][1]].append((i, j))
            if verbose:
                print(f"  k=4: {sum(len(v) for v in pairs.values())} pairs, {len(pairs)} masks", flush=True)
            seen = set()
            for D, plist in pairs.items():
                if len(plist) < 2:
                    continue
                for (i, j), (a, b) in itertools.combinations(plist, 2):
                    s4 = tuple(sorted({i, j, a, b}))
                    if len(s4) == 4 and s4 not in seen:
                        seen.add(s4)
                        o = mech[i][2] ^ mech[j][2] ^ mech[a][2] ^ mech[b][2]
                        record(s4, o, bad_list)
        else:
            raise ValueError("kmax <= 4")
        if verbose:
            print(f"  k={k}: {len(bad_list)} malignant configurations", flush=True)
        if bad_list:
            return k, bad_list, mech
    return None, [], mech
