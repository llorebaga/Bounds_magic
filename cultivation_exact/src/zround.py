"""Real-T-native modification of a double check: a Z-stabilizer measurement round on the data immediately before
the final transversal T layer.  By the coherent-completion lemma, real-T failures need an undetected X error on
the data when the final T layer acts; this round detects every such X error with a nontrivial syndrome.

Each Z face f gets a fresh ancilla a_f: R a_f; CX q -> a_f for q in f (scheduled into layers, at most one CX per
qubit per layer); M a_f.  Hooks of this round are Z errors on data (Z on the target ancilla spreads to the
controls), which commute with T.  Noise: the file's uniform model (DEPOLARIZE2 after CX, idle DEPOLARIZE1 on every
other qubit in every inserted tick, reset and measurement flips).  Detectors are regenerated from the proxy."""
import collections
import stim

from foldtree import split_ticks, regenerate_detectors


def schedule_faces(faces):
    """greedy layering of the CX gates (q, face) so that no qubit and no ancilla is used twice per layer"""
    edges = [(q, k) for k, f in enumerate(faces) for q in f]
    layers = []
    for q, k in sorted(edges, key=lambda e: (-len(faces[e[1]]), e[1], e[0])):
        for lay in layers:
            if all(q != a and k != b for a, b in lay):
                lay.append((q, k))
                break
        else:
            layers.append([(q, k)])
    return layers


def insert_zround(c, after_tick, faces, p=1e-3, anc0=None):
    T = split_ticks(c)
    nq = c.num_qubits
    anc0 = anc0 if anc0 is not None else nq
    anc = [anc0 + k for k in range(len(faces))]
    allq = list(range(nq)) + anc
    layers = schedule_faces(faces)
    out = stim.Circuit()
    for t, tk in enumerate(T):
        for inst in tk:
            out.append(inst)
        if t < len(T) - 1:
            out.append("TICK")
        if t == after_tick:
            out.append("R", anc)
            out.append("X_ERROR", anc, [p])
            out.append("DEPOLARIZE1", list(range(nq)), [p])
            out.append("TICK")
            for lay in layers:
                tg = [x for q, k in lay for x in (q, anc[k])]
                out.append("CX", tg)
                out.append("DEPOLARIZE2", tg, [p])
                busy = set(tg)
                out.append("DEPOLARIZE1", [x for x in allq if x not in busy], [p])
                out.append("TICK")
            out.append("M", anc, [p])
            out.append("DEPOLARIZE1", list(range(nq)), [p])
            out.append("TICK")
    return regenerate_detectors(out), len(layers) + 2
