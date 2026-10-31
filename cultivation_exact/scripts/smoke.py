import sys, time; sys.path.insert(0, __file__.rsplit('scripts',1)[0] + 'src')
import stim
from tdem import TDEM
G = __file__.rsplit("scripts", 1)[0] + "external/gidney_circuits/"
d1 = sys.argv[1] if len(sys.argv) > 1 else "3"
f = G + ("c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=15,b=Y,r=4,d1=3.stim" if d1 == "3"
         else "c=inject[unitary]+cultivate,p=0.001,noise=uniform,g=css,q=42,b=Y,r=10,d1=5.stim")
for magic in ["S", "T"]:
    t0 = time.time()
    m = TDEM(stim.Circuit.from_file(f), magic=magic, verbose=True)
    print(magic, "t =", m.t, "bloch", [round(x, 4) for x in m.output_bloch()], "flips", sorted(m.flip_set), f"{time.time()-t0:.1f}s")
    mech = m.mechanisms()
    print(" mechanisms", len(mech), "with D=0:", sum(1 for p, D, o in mech if D == 0),
          "harmful singles:", sum(1 for p, D, o in mech if D == 0 and m.harmful(o)))
