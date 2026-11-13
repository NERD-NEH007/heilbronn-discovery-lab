"""Serial driver.  python run_small2.py n mode [--eps E] [--nodiag] [--skip G --rho R] [--audit A] [--record FILE]
mode eps / exact / neg (as bb2.make_problem)."""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bb2
ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('mode')
ap.add_argument('--eps', type=float, default=1e-3); ap.add_argument('--nodiag', action='store_true')
ap.add_argument('--skip', type=float, default=None); ap.add_argument('--rho', type=float, default=0.0)
ap.add_argument('--audit', type=float, default=0.0); ap.add_argument('--record', default=None)
ap.add_argument('--maxfail', type=int, default=50)
a = ap.parse_args()
rec = [] if a.record else None
P, tl = bb2.make_problem(a.n, a.mode, a.eps, diag=not a.nodiag, skip_gap=a.skip, skip_rho=a.rho, record=rec)
P.audit = a.audit
st = bb2.new_stats(); fails = []; t0 = time.time()
for lo, hi, cid in P.roots():
    P.solve2(lo, hi, st, fails=fails, max_fail=a.maxfail)
    if len(fails) >= a.maxfail: print('ABORT: too many failures'); break
print(json.dumps(dict(n=a.n, mode=a.mode, eps=a.eps, diag=not a.nodiag, skip=a.skip, rho=a.rho, T=float(P.T),
                      sec=round(time.time() - t0, 1), audited=P.audit_count, lp_fail=P.lp_fail, **st)))
if fails: print('FAILS', len(fails), fails[:2])
if rec is not None: json.dump(rec, open(a.record, 'w'))
