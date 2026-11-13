"""Dump P2 certificates in an implementation-neutral format, for the clean-room checker (peer review, 2026-11-13).
bb2.py is NOT modified: the dump hooks the existing audit path (audit2 is called with (lo, hi, sig, mu) on a
random fraction --frac of all successful P2 discards; the original audit2 still runs and must agree).

Usage:  python dump_p2.py n --frac F --out FILE [--roots i,j,...] [--minutes M]
Output: JSON lines.  Line 1 (header):
  {"format": "p2cert-v1", "n": n, "D": 16777216, "T": "p/q"}
Each further line is one discarded box:
  {"lo": [...2n ints], "hi": [...2n ints], "mu": [[row, value], ...]}
  Box: x_i in [lo[i]/D, hi[i]/D], y_i in [lo[n+i]/D, hi[n+i]/D]  (the box as P2 saw it, after P0).
  mu: non-negative integer multipliers, nonzero entries only.  Each multiplies the ROW IN NATURAL UNITS listed
  here (tau = a free variable standing for min over oriented triples of sigma_t * 2*Delta_t; w_ab = x_a*y_b):
    ["area", i, j, k, s]   i<j<k, s=+1/-1:   tau - s * 2*Delta_ijk <= 0,
                           2*Delta_ijk = w_ij - w_ik + w_jk - w_ji + w_ki - w_kj
    ["mc", a, b, "LL"]     (x_a - xl)(y_b - yl) >= 0  as  -w_ab + xl*y_b + yl*x_a <= xl*yl
    ["mc", a, b, "HH"]     (xh - x_a)(yh - y_b) >= 0  as  -w_ab + xh*y_b + yh*x_a <= xh*yh
    ["mc", a, b, "HL"]     (xh - x_a)(y_b - yl) >= 0  as   w_ab - xh*y_b - yl*x_a <= -xh*yl
    ["mc", a, b, "LH"]     (x_a - xl)(yh - y_b) >= 0  as   w_ab - xl*y_b - yh*x_a <= -xl*yh
                           (xl, xh = box bounds of x_a; yl, yh = box bounds of y_b)
    ["ord", i]             x_i - x_{i+1} <= 0
    ["mean"]               sum_i x_i <= n/2
  (The integer model of bb2 multiplies every natural row by D^2; this does not change the certificate.)"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb2

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int)
ap.add_argument('--frac', type=float, required=True); ap.add_argument('--out', required=True)
ap.add_argument('--roots', default=None); ap.add_argument('--minutes', type=float, default=None)
a = ap.parse_args()
P, tl = bb2.make_problem(a.n, 'exact', diag=True, skip_gap=1.2)
n, m = P.n, P.m
KIND = ('LL', 'HH', 'HL', 'LH')
out = open(a.out, 'w')
out.write(json.dumps({"format": "p2cert-v1", "n": n, "D": bb2.D, "T": f"{P.T.numerator}/{P.T.denominator}"}) + '\n')
orig = P.audit2
def hook(lo, hi, sig, mu):
    ok = orig(lo, hi, sig, mu)                      # the original exact re-derivation must still pass
    if not ok: raise RuntimeError('AUDIT MISMATCH')
    rows = []
    for r in map(int, mu.nonzero()[0]):
        v = int(mu[r])
        if r < 2 * m:
            i, j, k = (int(q) for q in P.tri[r // 2]); rows.append([["area", i, j, k, 1 if r % 2 == 0 else -1], v])
        elif r < 2 * m + 4 * len(P.pairs):
            blk, p = divmod(r - 2 * m, len(P.pairs)); x, y = P.pairs[p]
            rows.append([["mc", int(x), int(y), KIND[blk]], v])
        elif r < 2 * m + 4 * len(P.pairs) + n - 1:
            rows.append([["ord", r - 2 * m - 4 * len(P.pairs)], v])
        else:
            rows.append([["mean"], v])
    out.write(json.dumps({"lo": [int(q) for q in lo], "hi": [int(q) for q in hi], "mu": rows}) + '\n')
    return True
P.audit2 = hook; P.audit = a.frac
st = bb2.new_stats(); fails = []; t0 = time.time()
roots = P.roots()
sel = range(len(roots)) if a.roots is None else [int(q) for q in a.roots.split(',')]
for idx in sel:
    lo, hi, cid = roots[idx]
    P.solve2(lo, hi, st, fails=fails, node_budget=None)
    if a.minutes and time.time() - t0 > 60 * a.minutes: break
out.close()
print(json.dumps(dict(n=n, roots=list(sel), sec=round(time.time() - t0, 1), dumped=P.audit_count, fails=len(fails), **st)))
