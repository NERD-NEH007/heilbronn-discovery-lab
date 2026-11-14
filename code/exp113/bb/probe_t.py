"""Knuth random-probe estimate of the bbt tree size (estimates only; copy of exp 110 probe2.py).
python probe_t.py n mode probes [seed] [--skip G]"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bbt
ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('mode'); ap.add_argument('probes', type=int)
ap.add_argument('seed', type=int, nargs='?', default=0); ap.add_argument('--skip', type=float, default=None)
a = ap.parse_args()
P, _ = bbt.make_problem(a.n, a.mode, skip_gap=a.skip)
roots = P.roots(); rng = np.random.default_rng(a.seed)
est = []; t0 = time.time(); evals = 0; st = bbt.new_stats(); fails = 0
for p in range(a.probes):
    lo, hi, c = roots[rng.integers(len(roots))]; lo, hi = lo.copy(), hi.copy(); pred = None
    tot, mult = 0.0, 1.0
    while True:
        tot += mult; evals += 1
        kind, ch = P.process(lo, hi, pred, st, c)
        if kind != 'split': fails += kind == 'fail'; break
        mult *= 2; (l2, h2, _), (l3, h3, _), pred = ch
        lo, hi = (l2, h2) if rng.integers(2) == 0 else (l3, h3)
    est.append(tot * len(roots))
E = np.array(est); sec = (time.time() - t0) / evals
print(json.dumps(dict(n=a.n, mode=a.mode, skip=a.skip, probes=a.probes, seed=a.seed, covers=len(P.covers), T=float(P.T),
                      mean=float(E.mean()), se=float(E.std() / np.sqrt(len(E))), median=float(np.median(E)), max=float(E.max()),
                      probe_fails=fails, sec_per_node=sec, est_core_hours=float(E.mean() * sec / 3600))))
