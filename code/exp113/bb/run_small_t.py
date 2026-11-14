"""Serial driver.  python run_small_t.py n mode [--eps E] [--skip G] [--audit A] [--maxfail K]   (mode eps / neg / exact)"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bbt
ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('mode')
ap.add_argument('--eps', type=float, default=1e-3); ap.add_argument('--skip', type=float, default=None)
ap.add_argument('--audit', type=float, default=0.0); ap.add_argument('--maxfail', type=int, default=50)
a = ap.parse_args()
P, (flo, fhi) = bbt.make_problem(a.n, a.mode, a.eps, skip_gap=a.skip)
P.audit = a.audit
st = bbt.new_stats(); fails = []; t0 = time.time(); roots = P.roots()
for lo, hi, c in roots:
    P.solve2(lo, hi, c, st, fails=fails, max_fail=a.maxfail, weight=1.0 / len(roots))
    if len(fails) >= a.maxfail: print('ABORT: too many failures'); break
print(json.dumps(dict(n=a.n, mode=a.mode, eps=a.eps, skip=a.skip, T=float(P.T), T_exact=str(P.T), f_upper=float(fhi),
                      roots=len(roots), covers=len(P.covers), sec=round(time.time() - t0, 1), audited=P.audit_count,
                      lp_fail=P.lp_fail, infeas=P.n_infeas, farkas=P.n_farkas, colsum_max_over_D=P.colsum_max / bbt.D, colcount_max=P.colcount_max, **st)))
if fails: print('FAILS', len(fails), fails[:2])
