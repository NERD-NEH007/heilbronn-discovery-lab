"""Exp 78b: can the free points (rattlers) be nudged so every unloaded tie (active at 1e-9 but zero multiplier) grows?
LP over free-point directions d (|d_i| <= 1, wall-respecting): maximise s subject to sg_k grad_free A_k . d >= s for
each unloaded tie k. s > 0 -> a small nudge leaves exactly the loaded ties critical (all other gaps are positive),
so that optimal-value configuration has #critical = #loaded ties.
Also reports near-ties (1e-9 < gap < 1e-6 relative): counted as ties by a 1e-6 tolerance but not exact ties."""
import os, sys, json, itertools
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE)); import hlib
from kkt_check import setup, loaded_subset, det

LO = int(sys.argv[1]) if len(sys.argv) > 1 else 37   # n <= 36 needs the published records in records/
for n in range(LO, 65):
    e = hlib.current(n); P = [tuple(p) for p in e['points']]
    ties, sg, core, rat, walls, t = setup(P)
    if len(ties) + len(walls) > 2 * len(core) + 1: continue
    try: lt, lsg, lcore, lwalls = loaded_subset(P, ties, sg, core, walls, t)
    except Exception: continue
    free = [i for i in range(n) if i not in lcore]; un = [(tr, s) for tr, s in zip(ties, sg) if tr not in set(lt)]
    tri = list(itertools.combinations(range(n), 3)); A = np.array([abs(det(P[a], P[b], P[c])) / 2 for a, b, c in tri])
    near = int(((A > t * (1 + 1e-9)) & (A <= t * (1 + 1e-6))).sum())
    s_opt = None
    if un:
        fx = {i: c for c, i in enumerate(free)}; m = 2 * len(free); rows = []
        for (a, b, c), s in un:
            g = np.zeros(m + 1)
            for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
                if p in fx: g[2 * fx[p]] += s * 0.5 * (P[q][1] - P[r][1]); g[2 * fx[p] + 1] += s * 0.5 * (P[r][0] - P[q][0])
            g[-1] = -1; rows.append(-g)        # -(g.d) + s <= 0
        bounds = []
        for i in free:
            for k in range(2):
                v = P[i][k]; bounds.append((0 if v < 1e-11 else -1, 0 if v > 1 - 1e-11 else 1))
        res = linprog(np.r_[np.zeros(m), -1.], A_ub=np.array(rows), b_ub=np.zeros(len(rows)), bounds=bounds + [(None, 1)], method='highs')
        s_opt = float(-res.fun) if res.status == 0 else None
    print(f"n={n} active(1e-9) {len(ties)} loaded {len(lt)} unloaded {len(un)} near-ties(1e-9..1e-6) {near} free {len(free)} "
          f"nudge s={s_opt if s_opt is None else f'{s_opt:.3g}'} 2n-7={2*n-7}", flush=True)
