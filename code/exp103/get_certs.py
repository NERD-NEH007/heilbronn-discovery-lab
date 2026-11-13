"""Run the exp 44 local certifier (unchanged) on the n = 5-10 optima of exp 102 and store, per n, the exact
polished centre x~ (rationals), the existence radius r, t enclosure and (first-order mode) the proven basin radius.
Writes output/exp103/local_certs.json (compare certificates/exp103/local_certs.json).  The global B&B (bb.py) uses: every box inside the sup-ball B(g.x~, basin - r - 2^-60)
for some square symmetry g and relabelling is covered by the local certificate."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(C, 'exp44')); sys.path.insert(0, C)
import hlib
import certify_interval as CI
S = hlib.small_n_optima()
out = {}
for n in range(5, 11):
    for a, v in enumerate(S[str(n)]):
        P = np.clip(np.array(v['points'], float), 0, 1)
        if n == 6 and a > 0: continue          # shear-family members are non-isolated; not certifiable
        try: o = CI.certify(n, dict(points=P.tolist(), family='rebuilt', source='exp102', value=v['value']), log=lambda *a, **k: None)
        except Exception as ex: o = dict(certified=False, reason=repr(ex))
        pts = o.pop('_points', None)
        keep = {k: o.get(k) for k in ('certified', 'reason', 'mode', 'existence_radius', 'box_rho', 'basin_radius',
                                      't_lower', 't_upper', 'symmetry', 'n_core', 'rattlers', 'loaded_ties', 'loaded_walls', 'y_min_lower')}
        keep['alt'] = a; keep['points_exact'] = pts['points'] if pts else None
        print(n, a, {k: keep[k] for k in ('certified', 'mode', 'basin_radius', 'existence_radius', 'box_rho', 't_lower')}, flush=True)
        out.setdefault(str(n), []).append(keep)
od = os.path.join(hlib.REPO, 'output', 'exp103'); os.makedirs(od, exist_ok=True)
json.dump(out, open(os.path.join(od, 'local_certs.json'), 'w'), indent=1)
