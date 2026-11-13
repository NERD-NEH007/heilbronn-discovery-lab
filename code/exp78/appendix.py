"""Appendix data for the conjecture note: every best (records n <= 36, ours above) whose exact critical count is < 2n - 7.
For each: published/library coordinates, t, counts (1e-6 tolerance, exact active, loaded), and where rattlers carry
unloaded ties a nudged configuration: rattlers moved by eps along the LP direction from nudge.py, with eps chosen so
that the minimum is unchanged and exactly the loaded ties stay within 1e-9 of it. Nudged configurations are checked
with exact rational arithmetic (Fraction): min area equals the original float configuration's min, other triangles larger.
Writes output/appendix_data.json (compare with certificates/appendix_data.json). Needs the published records
21, 22, 29, 35, 36 in records/ (see records/README.md)."""
import os, sys, json, itertools
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE)); import hlib
from kkt_check import setup, loaded_subset, det
OUT = os.path.join(hlib.REPO, 'output')
# Sizes whose exact or after-nudge count is below 2n - 7 (lint_conjecture.txt of 2026-11-12, plus the symmetric
# sizes 36 and 42, which the linter skips).
SIZES = [21, 22, 29, 35, 36, 39, 40, 41, 42, 45, 47, 48, 50, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64]
# Symmetric sizes: multipliers are not unique and loaded_subset keeps every tie. Use the rattlers of the exp 44
# certificate vertex (record 36: two C4 orbits, indices 20-23, 28-31 of the published file; our 42: points 7, 28).
# Checked 2026-11-12: the ties not through them number 60 / 76 (= exp 44's loaded ties), and the nudge LP has s > 0.
RATTLERS = {36: [20, 21, 22, 23, 28, 29, 30, 31], 42: [7, 28]}

def tri_areas(P):
    tri = list(itertools.combinations(range(len(P)), 3)); return tri, np.array([abs(det(P[a], P[b], P[c])) / 2 for a, b, c in tri])

def nudge_dir(P, free, un):
    fx = {i: c for c, i in enumerate(free)}; m = 2 * len(free); rows = []
    for (a, b, c), s in un:
        g = np.zeros(m + 1)
        for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
            if p in fx: g[2 * fx[p]] += s * 0.5 * (P[q][1] - P[r][1]); g[2 * fx[p] + 1] += s * 0.5 * (P[r][0] - P[q][0])
        g[-1] = -1; rows.append(-g)
    bounds = [((0 if P[i][k] < 1e-11 else -1), (0 if P[i][k] > 1 - 1e-11 else 1)) for i in free for k in range(2)]
    res = linprog(np.r_[np.zeros(m), -1.], A_ub=np.array(rows), b_ub=np.zeros(len(rows)), bounds=bounds + [(None, 1)], method='highs')
    return res.x[:m].reshape(-1, 2), -res.fun

def exact_counts(P, rel=Fr(1, 10**9)):
    Q = [(Fr(x), Fr(y)) for x, y in P]; A = [abs(det(Q[a], Q[b], Q[c])) / 2 for a, b, c in itertools.combinations(range(len(Q)), 3)]
    t = min(A); return t, sum(1 for a in A if a <= t * (1 + rel)), sum(1 for a in A if a <= t * (1 + Fr(1, 10**6)))

def main():
    data = []
    for n in SIZES:
        e = hlib.current(n)
        P = [tuple(p) for p in e['points']]
        ties, sg, core, rat, walls, t = setup(P)
        if n in RATTLERS:
            rs = set(RATTLERS[n]); lt = [tr for tr in ties if not rs & set(tr)]; lcore = sorted({i for tr in lt for i in tr})
            lwalls = [w for w in walls if w[0] in lcore]
        else: lt, lsg, lcore, lwalls = loaded_subset(P, ties, sg, core, walls, t)
        free = [i for i in range(n) if i not in lcore]; lset = set(lt); un = [(tr, s) for tr, s in zip(ties, sg) if tr not in lset]
        tri, A = tri_areas(P)
        row = dict(n=n, record=n <= 36, source=e['source'], t=float(t), tol1e6=int((A <= t * (1 + 1e-6)).sum()),
                   active=len(ties), loaded=len(lt), walls_loaded=len(lwalls), core=len(lcore), rattlers=free, bound=2 * n - 7, points=[list(p) for p in P])
        if un:
            fs = set(free); near = [(tri[j], 1 if det(*[P[i] for i in tri[j]]) > 0 else -1) for j in np.where((A > t * (1 + 1e-9)) & (A <= t * (1 + 1e-6)))[0] if fs & set(tri[j])]
            D, s = nudge_dir(P, free, un + near); best = None; row['near_ties_through_rattlers'] = len(near)
            for eps in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7):
                Q = [list(p) for p in P]
                for k, i in enumerate(free): Q[i] = [min(1., max(0., P[i][0] + eps * D[k][0])), min(1., max(0., P[i][1] + eps * D[k][1]))]
                _, B = tri_areas([tuple(q) for q in Q]); tq = B.min()
                if tq >= t * (1 - 1e-14) and int((B <= tq * (1 + 1e-9)).sum()) == len(lt): best = (eps, Q); break
            if best:
                eps, Q = best; te, ce, c6 = exact_counts(Q); te0, _, _ = exact_counts(P)
                row.update(nudge_eps=eps, nudged_points=Q, nudged_exact_count=ce, nudged_count_1e6=c6, nudged_min_equal=(te == te0) or float(abs(te - te0) / te0) < 1e-15,
                           nudged_min_rel_change=float((te - te0) / te0))
        data.append(row); print(n, {k: v for k, v in row.items() if 'points' not in k and k != 'source'}, flush=True)
    os.makedirs(OUT, exist_ok=True); json.dump(data, open(os.path.join(OUT, 'appendix_data.json'), 'w'), indent=1)

if __name__ == '__main__': main()
