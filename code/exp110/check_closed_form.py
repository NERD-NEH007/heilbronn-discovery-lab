"""paper3: identify the certified n = 9 optimum x* (exp 44 certificate, stored by exp 103) with an exact
configuration C in Q(sqrt 65) and check the closed form t* = (9 sqrt 65 - 55)/320.  Run from the repo root:
    python code/exp110/check_closed_form.py
Checks (sympy, exact in Q(sqrt 65) except the margin of non-minimal triangles, printed to 30 digits):
  1. the closed form lies in the certified enclosure [t_lower, t_upper];
  2. C (recognised from the exact polished centre) is within sup distance < r (existence radius) of that centre;
  3. exactly 11 triangles of C have area exactly (9 sqrt 65 - 55)/320, all others are larger."""
import os, json, ast, itertools
from fractions import Fraction as Fr
import sympy as sp
HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, '..', '..', 'certificates', 'exp103', 'local_certs.json')))['9'][0]
P = C['points_exact']; P = ast.literal_eval(P) if isinstance(P, str) else P
s = sp.sqrt(65); v = (9 * s - 55) / 320
tl = sp.Rational(C['t_lower'].split('e')[0]) / sp.Integer(10) ** 32
tu = sp.Rational(C['t_upper'].split('e')[0]) / sp.Integer(10) ** 32
print('closed form in enclosure:', bool(sp.N(v - tl, 50) > 0 and sp.N(tu - v, 50) > 0), sp.N(v, 40))
def R(q): q = Fr(q); return sp.Rational(q.numerator, q.denominator)
def ident(q):
    q = R(q)
    return q if q in (0, 1) else sp.nsimplify(sp.nsimplify(sp.N(q, 50), [s], tolerance=1e-35))
Q = [(ident(x), ident(y)) for x, y in P]
print('C =', Q)
dist = max(abs(sp.N(a - R(x), 60)) for (a, b), (x, y) in zip(Q, P) for a, x in ((a, x), (b, y)))
print('sup distance to polished centre:', sp.N(dist, 5), ' existence radius r =', C['existence_radius'])
tied, others = [], []
for t in itertools.combinations(range(9), 3):
    (x1, y1), (x2, y2), (x3, y3) = (Q[i] for i in t)
    A = sp.expand(sp.Rational(1, 2) * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)))
    if sp.simplify(abs(A) - v) == 0: tied.append(t)
    else: others.append(sp.N(abs(A) - v, 30))
print('triangles with area exactly v:', len(tied), tied)
print('min margin of the other', len(others), 'triangles:', min(others))
print('diagonal symmetric:', sorted(Q, key=str) == sorted([(b, a) for a, b in Q], key=str))
