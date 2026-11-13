"""Knuth random-probe estimate of the B&B tree size (exploratory; for run-time estimates only).
python probe.py n mode probes [dom]"""
import sys, os, json, time
import numpy as np
from fractions import Fraction as Fr
os.environ.setdefault('DOM', sys.argv[4] if len(sys.argv) > 4 else 'basic')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_par, bb
n, mode, K = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
P, tl = run_par.make_problem(n, mode, 1e-3)
roots = P.roots(); rng = np.random.default_rng(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
def expand(lo, hi, cid):
    P.cur_cons = P.cons_table[cid]; T2 = P.T2f
    if not P.propagate(lo, hi): return None
    A2lo, A2hi = P.triple_ranges(lo, hi)
    if (np.maximum(A2hi, -A2lo) <= T2).any(): return None
    sig = np.where(A2lo >= -T2, 1, np.where(A2hi <= T2, -1, 0)).astype(np.int64)
    wmax = int((hi - lo).max())
    if P.covers and wmax <= P.cover_width and P.covered(lo, hi): return None
    ok, val, info = P.lp_bound(lo, hi, sig)
    if ok: return None
    kq = P.choose(lo, hi, sig, A2lo, A2hi)
    if kq is None: return None
    k, mid = kq
    l2, h2 = lo.copy(), hi.copy(); h2[k] = mid; l3, h3 = lo.copy(), hi.copy(); l3[k] = mid
    return [(l2, h2), (l3, h3)]
est = []; t0 = time.time(); evals = 0
strat = os.environ.get('STRAT') == '1'
per = {}
paths = []
for p in range(K):
    ri = (p % len(roots)) if strat else rng.integers(len(roots))
    r = roots[ri]; lo, hi, cid = r[0].copy(), r[1].copy(), r[2]
    tot, mult = 0.0, 1.0
    while True:
        tot += mult; evals += 1
        ch = expand(lo, hi, cid)
        if ch is None: break
        mult *= 2; lo, hi = ch[rng.integers(2)]
    est.append(tot * len(roots)); per.setdefault(int(ri), []).append(tot)
    paths.append((tot, lo.tolist(), hi.tolist()))
E = np.array(est)
sec = (time.time() - t0) / evals
if strat:
    tab = sorted(((float(np.mean(v)), k) for k, v in per.items()), reverse=True)
    tot_ = sum(m for m, k in tab)
    for m, k in tab[:12]:
        lo_, hi_, c_ = roots[k]; print(f'root {k} cid {c_} fixed-y-bottom {[i for i in range(n) if hi_[n+i]==0]} top {[i for i in range(n) if lo_[n+i]==bb.D]}: mean {m:.3g} ({m/tot_:.1%})')
paths.sort(key=lambda z: -z[0])
_od = os.path.join(run_par.REPO, 'output', 'exp103'); os.makedirs(_od, exist_ok=True)
json.dump(paths[:20], open(os.path.join(_od, f'probe_top_n{n}.json'), 'w'))
print(json.dumps(dict(n=n, mode=mode, dom=P.domain, probes=K, mean=float(E.mean()), se=float(E.std() / np.sqrt(K)), median=float(np.median(E)),
                      sec_per_node=sec, est_core_hours=float(E.mean() * sec / 3600))))
