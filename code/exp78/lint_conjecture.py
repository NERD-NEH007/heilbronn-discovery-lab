"""Conjecture linter (adopted from peer review 2, 2026-11-11): for every library best n = 20-64 (records as published
for n <= 36), report the exact critical-triangle count (relative 1e-9, the record site's rule), the loaded count
(= the critical count after a rattler nudge, when the nudge LP is feasible), and flag counts below 2n - 7
(Sudermann-Merx, arXiv 2603.11107, Conj. 1). Also checks Modir-Monji-Kocuk Conj. 1 (exactly 2 points on each edge).
python code/exp78/lint_conjecture.py [lo]  (writes output/lint_conjecture.txt; compare certificates/exp78/lint_conjecture.txt;
lo defaults to 20 if records/ holds the published records, else 37)."""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE)); import hlib
from kkt_check import setup, loaded_subset, det
from appendix import nudge_dir, tri_areas

def main():
    out = []; lo = int(sys.argv[1]) if len(sys.argv) > 1 else (20 if hlib.record_path(20) else 37)
    for n in range(lo, 65):
        e = hlib.current(n); P, src = [tuple(p) for p in e['points']], ('record' if n <= 36 else 'ours')
        ties, sg, core, rat, walls, t = setup(P); edges = [sum(1 for p in P if abs(p[k] - v) < 1e-9) for k, v in ((0, 0), (0, 1), (1, 0), (1, 1))]
        row = f"n={n:2} {src:6} critical {len(ties):3}  2n-7 {2*n-7:3}"
        if len(ties) + len(walls) > 2 * len(core) + 1:
            row += "  loaded -   (symmetric, multipliers not unique)"; nud = None
        else:
            lt, lsg, lcore, lw = loaded_subset(P, ties, sg, core, walls, t); free = [i for i in range(n) if i not in lcore]
            lset = set(lt); un = [(tr, s) for tr, s in zip(ties, sg) if tr not in lset]; nud = len(ties)
            if un:
                tri, A = tri_areas(P); fs = set(free)
                near = [(tri[j], 1 if det(*[P[i] for i in tri[j]]) > 0 else -1) for j in np.where((A > t * (1 + 1e-9)) & (A <= t * (1 + 1e-6)))[0] if fs & set(tri[j])]
                _, s = nudge_dir(P, free, un + near); nud = len(lt) if s > 1e-9 else len(ties)
            row += f"  after nudge {nud:3}"
        flag = []
        if len(ties) < 2 * n - 7: flag.append('BELOW 2n-7 as given')
        elif nud is not None and nud < 2 * n - 7: flag.append('below 2n-7 after nudge')
        if edges != [2, 2, 2, 2]: flag.append(f'edges {edges} (not 2 per edge)')
        out.append(row + ('   <- ' + '; '.join(flag) if flag else ''))
        print(out[-1], flush=True)
    os.makedirs(os.path.join(hlib.REPO, 'output'), exist_ok=True)
    open(os.path.join(hlib.REPO, 'output', 'lint_conjecture.txt'), 'w').write('\n'.join(out) + '\n')

if __name__ == '__main__': main()
