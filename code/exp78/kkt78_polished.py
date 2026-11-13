"""Exp 108: run the exp 78 independent checker (kkt_check.certify, unchanged) on the exp 44 polished centres
(44-interval-certificate/results/polished/n{n}.json, rounded to floats) instead of the stored elites coordinates.
Needed at the second-order sizes, where the stored coordinates lie 1e-7..5e-6 from the KKT point along the flat
direction, so exp 78's 1e-9 tie selection misses ties there.
    python code/exp78/kkt78_polished.py 65,76,81,89,100 output/exp78/kkt78_polished.jsonl
(polished centres: configurations/polished/n{n}.json)"""
import os, sys, json, time
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import hlib
import kkt_check as K
PD = os.path.join(hlib.CONF, 'polished')
sizes = list(map(int, sys.argv[1].split(','))); out = sys.argv[2]
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, 'a') as fo:
    for n in sizes:
        d = json.load(open(os.path.join(PD, f'n{n}.json')))
        P = [(float(Fr(x)), float(Fr(y))) for x, y in d['points_exact']]
        cur = hlib.current_value(n)
        tp = float(Fr(d['t_centre_exact'])); t0 = time.time()
        try: info = K.certify(n, P)
        except Exception as ex: info = dict(n=n, certified=False, reason=f'{type(ex).__name__}: {ex}')
        info.update(input='exp44 polished centre', t_polished_vs_stored_rel=tp / cur - 1, secs=round(time.time() - t0, 1))
        fo.write(json.dumps(info) + '\n'); fo.flush()
        print({k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in info.items() if k != 'second_order_eigs'},
              'eigs', info.get('second_order_eigs'), flush=True)
