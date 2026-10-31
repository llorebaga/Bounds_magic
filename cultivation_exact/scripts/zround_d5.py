"""Gidney d=5 + Z-syndrome round before the final T layer: exact real-T fault distance (to weight 4) and S proxy."""
import sys, time, json
sys.path.insert(0, __file__.rsplit('scripts', 1)[0] + 'src')
import stim
from zround import insert_zround
from tdem import TDEM
from fd_fast import fault_distance_fast
from fd_list import harmful_k4

FACES = [[0, 3, 7, 11], [9, 13, 15, 19, 21, 27], [17, 25, 32, 36], [3, 7, 9, 13], [15, 21, 23, 29],
         [19, 25, 27, 32, 34, 38], [7, 11, 13, 17, 19, 25], [21, 27, 29, 34], [32, 36, 38, 40]]
if __name__ == "__main__":
    after = int(sys.argv[1]) if len(sys.argv) > 1 else 91
    c = stim.Circuit.from_file(__file__.rsplit('scripts', 1)[0] + "results/gidney_d5.stim")
    cz, nl = insert_zround(c, after, FACES)
    path = __file__.rsplit('scripts', 1)[0] + f"results/zround_after{after}.stim"
    open(path, "w").write(str(cz))
    print(f"inserted Z round after tick {after}: {nl} ticks, {len(FACES)} ancillas -> {path}", flush=True)
    for magic, kw in [("T", {"complement": True}), ("S", {})]:
        t0 = time.time()
        m = TDEM(cz, magic=magic, **kw)
        print(magic, "bloch", [round(x, 3) for x in m.output_bloch()], flush=True)
        mech = m.mechanisms()
        k, found = fault_distance_fast(m, 3, mech, verbose=False)
        if k is None:
            f4 = harmful_k4(m, mech, verbose=True)
            k, found = (4, f4) if f4 else (None, [])
        coef = sum(f[4] * f[3] for f in found) / 1e-3 ** k if k else 0
        print(f"{magic}: fault distance {k if k else '>=5'}, {len(found)} malignant, leading coefficient {coef:.4g} "
              f"({time.time()-t0:.0f}s)", flush=True)
