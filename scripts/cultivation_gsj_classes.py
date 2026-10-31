"""Group the harmful pairs of the published distance-three circuit (Proposition 5) into fault classes, as Chan et
al. count them: a fault class collects the single faults with the same detector pattern and the same Pauli operator
on the T inputs (the same fault-map key (D, U, E)), and a harmful configuration is a pair of classes.
Usage: python cultivation_gsj_classes.py   (prints the counts with and without idle noise)"""
import cultivation_gsj as g

for idle in (False, True):
    ops, w, ideal, data, nq = g.build("T", idle=idle)
    weights = g.all_faults(ops, ideal) if not idle else w
    A, faults = g.analysis(ops, weights, ideal, data, nq)
    key_of = {(x[1], x[2]): x[4] for x in faults}
    harm, npairs = g.harmful_pairs(A, faults)
    classes = {frozenset((key_of[(a[0], a[1])], key_of[(b[0], b[1])])) for a, b, k in harm}
    singles = {key_of[(a[0], a[1])] for a, b, k in harm} | {key_of[(b[0], b[1])] for a, b, k in harm}
    print(f"idle noise {idle}: {len(harm)} harmful pairs of faults, {len(classes)} harmful pairs of fault classes, "
          f"{len(singles)} fault classes involved")
