"""Noise models for Gidney-Shutty-Jones style circuits (input: their uniform-noise .stim files).

uniform(c, p): every channel of the file set to p (the file's own model, rescaled).
si1000(c, p):  superconducting-inspired model of Gidney-Newman-McEwen, as used for Google's hardware
               projections: 2q gates DEPOLARIZE2(p); 1q gates (incl. T) DEPOLARIZE1(p/10); idle p/10;
               resets flip 2p; measurements flip 5p; idling during a measurement/reset layer 2p.
The idle sets are those of the file (every qubit not acted on in a tick), so the circuits stay comparable.
"""
import stim

NOISE = ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR")
ANNOT = ("DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS", "TICK")
MEAS = ("M", "MX", "MY")
RESET = ("R", "RX", "RY")


def _ticks(c):
    out = [[]]
    for inst in c.flattened():
        out[-1].append(inst)
        if inst.name == "TICK":
            out.append([])
    return out


def uniform(c, p):
    out = stim.Circuit()
    for inst in c.flattened():
        args = inst.gate_args_copy()
        if inst.name in NOISE or (inst.name in MEAS + ("MPP",) and args):
            out.append(inst.name, inst.targets_copy(), [p for _ in args])
        else:
            out.append(inst)
    return out


def si1000(c, p, t_noise=None):
    """t_noise: depolarizing strength after the magic (S/S_DAG) gates; default p/10 like other 1q gates"""
    t_noise = p / 10 if t_noise is None else t_noise
    out = stim.Circuit()
    for tick in _ticks(c):
        g1, g2, meas, reset, magic = set(), set(), set(), set(), set()
        for inst in tick:
            q = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if inst.name in MEAS:
                meas.update(q)
            elif inst.name in RESET:
                reset.update(q)
            elif inst.name in ("S", "S_DAG"):
                magic.update(q)
            elif inst.name in ("CX", "CZ", "CY", "SWAP", "XCX", "ZCZ"):
                g2.update(q)
            elif inst.name not in NOISE + ANNOT + ("MPP",):
                g1.update(q)
        heavy = bool(meas or reset)
        for inst in tick:
            nm = inst.name
            args = inst.gate_args_copy()
            tg = inst.targets_copy()
            if nm == "DEPOLARIZE2":
                out.append(nm, tg, [p])
            elif nm in ("X_ERROR", "Z_ERROR", "Y_ERROR"):
                qs = set(t.value for t in tg)
                if qs <= reset:
                    out.append(nm, tg, [2 * p])            # reset error
                elif qs <= meas:
                    out.append(nm, tg, [5 * p])            # measurement error written as a flip before M
                else:
                    raise AssertionError("flip channel neither after a reset nor before a measurement")
            elif nm == "DEPOLARIZE1":
                groups = {}
                for t in tg:
                    q = t.value
                    if q in meas:
                        continue
                    if q in magic:
                        r = t_noise
                    elif q in g1:
                        r = p / 10
                    elif q in reset:
                        continue
                    else:
                        r = 2 * p if heavy else p / 10
                    groups.setdefault(r, []).append(q)
                for r, qs in groups.items():
                    if r > 0:
                        out.append("DEPOLARIZE1", qs, [r])
            elif nm in MEAS and args:
                out.append(nm, tg, [5 * p])
            else:
                out.append(inst)
    return out
