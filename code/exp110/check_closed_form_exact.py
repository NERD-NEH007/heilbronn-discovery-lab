"""Exact identification of the certified n = 9 optimum with a configuration C in Q(sqrt 65).

Replaces the numerical steps of check_closed_form.py (peer review, 2026-11-13). sympy's nsimplify is used
only to PROPOSE C; every claim below is then checked in exact arithmetic in Q(sqrt 65), with numbers stored as
pairs (a, b) of Fractions meaning a + b*sqrt(65), and signs decided exactly
(sign(a + b r) for r = sqrt 65: compare a^2 with 65 b^2).

Checks (all exact; the script stops with AssertionError on any failure):
  1. t_lower < v < t_upper, where v = (9 sqrt 65 - 55)/320 and [t_lower, t_upper] is the certified enclosure.
  2. Every coordinate of C lies in [0, 1], and the 8 wall contacts of the centre (coordinates exactly 0 or 1)
     are the same coordinates of C.
  3. Each coordinate of C is within r_cert = (1 - 1e-6) * r of the corresponding coordinate of the polished centre
     P (r = stored existence radius, a double; the margin covers its rounding).
  4. For each of the 84 triangles, |area(C)| is either EXACTLY v or STRICTLY greater than v.
  5. The tied triangles of C are exactly the 11 triangles that are within 1e-20 (relative) of minimal at P,
     with the same orientation signs; C is symmetric under the diagonal mirror (x, y) -> (y, x).
Hence C solves the certificate's tie system exactly, lies in its existence box, and so C equals the certified
optimum; its minimum area is exactly v.
Usage:  python check_closed_form_exact.py [path/to/local_certs.json]"""
import os, sys, json, ast, itertools
from fractions import Fraction as Fr
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.join(HERE, '..', '..', '103-global-proof', 'results', 'local_certs.json'),   # private repo
              os.path.join(HERE, '..', '..', 'certificates', 'exp103', 'local_certs.json')]        # public repo
path = sys.argv[1] if len(sys.argv) > 1 else next(p for p in CANDIDATES if os.path.exists(p))
CERT = json.load(open(path))['9'][0]

class Q65:
    """a + b*sqrt(65), a, b rational."""
    __slots__ = ('a', 'b')
    def __init__(s, a, b=0): s.a, s.b = Fr(a), Fr(b)
    def __add__(s, o): o = o if isinstance(o, Q65) else Q65(o); return Q65(s.a + o.a, s.b + o.b)
    def __sub__(s, o): o = o if isinstance(o, Q65) else Q65(o); return Q65(s.a - o.a, s.b - o.b)
    def __neg__(s): return Q65(-s.a, -s.b)
    def __mul__(s, o):
        o = o if isinstance(o, Q65) else Q65(o)
        return Q65(s.a * o.a + 65 * s.b * o.b, s.a * o.b + s.b * o.a)
    def __eq__(s, o): o = o if isinstance(o, Q65) else Q65(o); return s.a == o.a and s.b == o.b
    def sign(s):
        a, b = s.a, s.b
        if b == 0: return (a > 0) - (a < 0)
        if a == 0: return (b > 0) - (b < 0)
        if (a > 0) == (b > 0): return 1 if a > 0 else -1
        # opposite signs: compare a^2 with 65 b^2
        c = a * a - 65 * b * b
        assert c != 0                     # sqrt 65 is irrational
        return (1 if a > 0 else -1) if c > 0 else (1 if b > 0 else -1)
    def __float__(s): return float(s.a) + float(s.b) * 65 ** 0.5

def to_q65(expr, s65):
    """exact conversion of a sympy expression a + b*sqrt(65) (a, b rational) to Q65."""
    e = sp.expand(expr)
    b = e.coeff(s65); a = sp.expand(e - b * s65)
    assert a.is_Rational and b.is_Rational, expr
    return Q65(Fr(int(a.p), int(a.q)), Fr(int(b.p), int(b.q)))

s65 = sp.sqrt(65)
V = Q65(Fr(-55, 320), Fr(9, 320))                          # v = (9 sqrt 65 - 55)/320
P = CERT['points_exact']; P = ast.literal_eval(P) if isinstance(P, str) else P
P = [(Fr(x), Fr(y)) for x, y in P]
assert len(P) == 9

# 1. enclosure
tl = Fr(int(CERT['t_lower'].split('e')[0]), 10 ** 32); tu = Fr(int(CERT['t_upper'].split('e')[0]), 10 ** 32)
assert CERT['t_lower'].endswith('e-32') and CERT['t_upper'].endswith('e-32')
assert (V - tl).sign() > 0 and (Q65(tu) - V).sign() > 0
print('1. t_lower < v < t_upper: exact OK')

# propose C (discovery only), then convert exactly
def propose(q):
    if q in (0, 1): return Q65(q)
    e = sp.nsimplify(sp.N(sp.Rational(q.numerator, q.denominator), 60), [s65], tolerance=1e-40)
    return to_q65(e, s65)
C = [(propose(x), propose(y)) for x, y in P]

# 2. inside the square; same wall contacts
for (cx, cy), (px, py) in zip(C, P):
    for c, p in ((cx, px), (cy, py)):
        assert c.sign() >= 0 and (Q65(1) - c).sign() >= 0
        if p in (0, 1): assert c == Q65(p)
print('2. C in [0,1]^2, wall contacts identical: exact OK')

# 3. within the existence radius of the polished centre
r = Fr(CERT['existence_radius']) * (1 - Fr(1, 10 ** 6))
dmax = 0.0
for (cx, cy), (px, py) in zip(C, P):
    for c, p in ((cx, px), (cy, py)):
        d = c - p
        assert (Q65(r) - d).sign() > 0 and (Q65(r) + d).sign() > 0
        dmax = max(dmax, abs(float(sp.N(sp.Rational(d.a.numerator, d.a.denominator)
                                         + sp.Rational(d.b.numerator, d.b.denominator) * s65, 50))))
print(f'3. |C - P|_inf < r(1 - 1e-6), r = {float(CERT["existence_radius"]):.4g}: exact OK (float estimate {dmax:.3g})')

# 4. every triangle: exactly v or strictly larger
def area2(p, q, r_):  # signed double area
    return (q[0] - p[0]) * (r_[1] - p[1]) - (r_[0] - p[0]) * (q[1] - p[1])
V2 = V * 2
tied, margin = {}, None
for t in itertools.combinations(range(9), 3):
    A = area2(*(C[i] for i in t)); sg = A.sign(); assert sg != 0
    absA = A if sg > 0 else -A
    diff = absA - V2
    if diff == Q65(0): tied[t] = sg
    else:
        assert diff.sign() > 0, t                           # strictly greater
        m = float(diff) / 2; margin = m if margin is None else min(margin, m)
print(f'4. {len(tied)} triangles with area exactly v; the other {84 - len(tied)} strictly larger '
      f'(smallest excess {margin:.4g}): exact OK')

# 5. tie set and orientation match the polished centre's near-minimal set; diagonal symmetry
def area2f(p, q, r_): return (q[0] - p[0]) * (r_[1] - p[1]) - (r_[0] - p[0]) * (q[1] - p[1])
AP = {t: area2f(*(P[i] for i in t)) for t in itertools.combinations(range(9), 3)}
mP = min(abs(a) for a in AP.values())
near = {t: (1 if a > 0 else -1) for t, a in AP.items() if abs(a) <= mP * (1 + Fr(1, 10 ** 20))}
assert near == tied and len(tied) == 11, (sorted(near), sorted(tied))
key = lambda q: (q[0].a, q[0].b, q[1].a, q[1].b)
assert sorted(C, key=key) == sorted([(y, x) for x, y in C], key=key)
print('5. tie set and orientations equal the certificate centre\'s 11 near-ties; diagonal symmetric: exact OK')
print('C =', [(f'{x.a}+{x.b}r', f'{y.a}+{y.b}r') for x, y in C], '(r = sqrt 65)')
print('ALL EXACT CHECKS PASSED')
