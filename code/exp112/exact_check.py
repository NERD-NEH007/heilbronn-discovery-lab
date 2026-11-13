"""Exact check of a repair: original float coordinates as exact rationals, t = exact min.
Each moved point is re-placed exactly at the intersection of two of its tie lines
(or a tie line and a wall); verify exactly: new min == t, and count exact ties."""
import json, sys, itertools
from fractions import Fraction as F

def area2(p, q, r):
    return abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))   # = 2*area

def exact_min_and_ties(P):
    vals = [area2(P[i], P[j], P[k]) for i, j, k in itertools.combinations(range(len(P)), 3)]
    m = min(vals); return m, sum(v == m for v in vals)

def solve(l1, l2):
    (a1, b1, c1), (a2, b2, c2) = l1, l2
    D = a1*b2 - a2*b1
    if D == 0: return None
    return ((c1*b2 - c2*b1)/D, (a1*c2 - a2*c1)/D)

def lines_through(p, Q, T2, tol=1e-9):
    """lines (u, c) with u.x = c on which some area2(x,a,b) = T2, active near p (float), plus walls."""
    L = []
    for a, b in itertools.combinations(Q, 2):
        d = (b[0]-a[0], b[1]-a[1]); u = (-d[1], d[0]); c0 = u[0]*a[0] + u[1]*a[1]
        s = u[0]*p[0] + u[1]*p[1] - c0
        if abs(abs(float(s)) - float(T2)) <= tol*float(T2):
            L.append((u[0], u[1], c0 + (T2 if s > 0 else -T2)))
    for v in (0, 1):
        if abs(float(p[0]) - v) < 1e-12: L.append((F(1), F(0), F(v)))
        if abs(float(p[1]) - v) < 1e-12: L.append((F(0), F(1), F(v)))
    return L

if __name__ == '__main__':
    src, res, n = sys.argv[1], sys.argv[2], sys.argv[3]
    d = json.load(open(src)); v = d[n] if n in d else d
    P0 = [(F(x), F(y)) for x, y in v['points']]
    T2, k0 = exact_min_and_ties(P0)
    R = json.load(open(res))
    P = list(P0)
    for k in sorted({m[0] for m in R['moves']}):
        pf = R['points_after'][k]
        Q = [P[i] for i in range(len(P)) if i != k]
        L = lines_through(pf, Q, T2)
        ok = False
        for l1, l2 in itertools.combinations(L, 2):
            x = solve(l1, l2)
            if x is None or not (0 <= x[0] <= 1 and 0 <= x[1] <= 1): continue
            if all(area2(x, a, b) >= T2 for a, b in itertools.combinations(Q, 2)):
                P[k] = x; ok = True; break
        print('point', k, 'lines', len(L), 'placed exactly' if ok else 'FAILED')
    T2b, k1 = exact_min_and_ties(P)
    print(f'n={n}: exact min unchanged: {T2b == T2}; exact ties before {k0}, after {k1}')
    vals = sorted(float(area2(*tri)/T2) for tri in itertools.combinations(P, 3))
    print('ties within rel 1e-12 after:', sum(v <= 1 + 1e-12 for v in vals), ' 2n-7 =', 2*int(n) - 7)
