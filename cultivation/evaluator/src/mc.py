"""Logical error rate and acceptance of a cultivation circuit with real magic gates, by sampling the TDEM.

Each sampled fault configuration F with D(F)=0 is weighted exactly: acceptance a(o) and bad weight b(o) of
the observable-flip class o = o(F), from the Choi-state contraction (TDEM.evaluate).  Then
    P_acc = E[1{D=0} a(o)],  P_bad = E[1{D=0} b(o)],  eps = P_bad / P_acc
(relative to the fault-free acceptance, which is 1 for the Gidney circuits)."""
import numpy as np
import stim


def scale_noise(circuit, factor):
    out = stim.Circuit()
    for inst in circuit.flattened():
        args = inst.gate_args_copy()
        if inst.name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR") or \
                (inst.name in ("M", "MX", "MY", "MPP", "MR", "MRX") and args):
            out.append(inst.name, inst.targets_copy(), [a * factor for a in args])
        else:
            out.append(inst)
    return out


def sample(m, shots, batch=2_000_000, seed=None, dem=None):
    dem = dem or m.dem()
    sampler = dem.compile_sampler(seed=seed)
    ng = len(m.gens)
    tot_acc = tot_bad = tot_bad2 = tot_acc2 = tot_ab = 0.0
    n = 0
    n_det0 = 0
    while n < shots:
        b = min(batch, shots - n)
        det, obs, _ = sampler.sample(b, bit_packed=True)
        keep = ~det.any(axis=1)
        n_det0 += int(keep.sum())
        ob = obs[keep]
        if len(ob):
            # integer key of each observable mask
            ob64 = np.zeros((len(ob), 8), dtype=np.uint8)
            ob64[:, :ob.shape[1]] = ob[:, :8] if ob.shape[1] >= 8 else ob
            if ob.shape[1] > 8:
                keys = [int.from_bytes(r.tobytes(), "little") for r in ob]
                uniq, inv = np.unique(np.array(keys, dtype=object), return_inverse=True)
                cnt = np.bincount(inv)
            else:
                k64 = ob64.view(np.uint64).ravel()
                uniq, cnt = np.unique(k64, return_counts=True)
            for key, c in zip(uniq, cnt):
                acc, bad = m.evaluate(int(key))
                tot_acc += c * acc
                tot_acc2 += c * acc * acc
                tot_bad += c * bad
                tot_bad2 += c * bad * bad
                tot_ab += c * acc * bad
        n += b
    return finish(dict(n=n, n_det0=n_det0, acc=tot_acc, acc2=tot_acc2, bad=tot_bad, bad2=tot_bad2, ab=tot_ab),
                  len(m._cache))


def merge(parts):
    out = {}
    for p in parts:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return out


def finish(S, classes=None):
    n = S["n"]
    tot_acc, tot_acc2, tot_bad, tot_bad2, tot_ab, n_det0 = S["acc"], S["acc2"], S["bad"], S["bad2"], S["ab"], S["n_det0"]
    Pa, Pb = tot_acc / n, tot_bad / n
    va = tot_acc2 / n - Pa ** 2
    vb = tot_bad2 / n - Pb ** 2
    cab = tot_ab / n - Pa * Pb
    eps = Pb / Pa if Pa else float("nan")
    # delta-method standard error of the ratio
    se_eps = np.sqrt(max(vb / Pa ** 2 - 2 * Pb * cab / Pa ** 3 + Pb ** 2 * va / Pa ** 4, 0) / n) if Pa else float("nan")
    return dict(shots=n, P_det0=n_det0 / n, P_acc=Pa, P_bad=Pb, se_P_bad=np.sqrt(max(vb, 0) / n),
                eps=eps, se_eps=se_eps, distinct_classes=classes, raw=S)


def _worker(args):
    circuit_text, magic, shots, seed, kw = args
    from tdem import TDEM
    m = TDEM(stim.Circuit(circuit_text), magic=magic, **kw)
    return sample(m, shots, seed=seed)["raw"]


def sample_parallel(circuit, magic, shots, workers=16, chunks=None, seed0=0, overrides=None, **tdem_kw):
    """split shots over processes (call under if __name__ == '__main__'); overrides / tdem_kw are passed to
    TDEM (reference-branch choice, complement pattern, ...)"""
    import multiprocessing as mp
    chunks = chunks or workers
    per = -(-shots // chunks)
    kw = dict(tdem_kw)
    if overrides:
        kw["overrides"] = overrides
    jobs = [(str(circuit), magic, per, seed0 + i, kw) for i in range(chunks)]
    with mp.Pool(workers) as pool:
        parts = pool.map(_worker, jobs)
    return finish(merge(parts))
