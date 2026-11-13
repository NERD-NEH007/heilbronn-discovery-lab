"""Shared helpers for the scripts in code/: repository paths, loading configurations, exact minimum area.

Configurations
  n >= 37   : configurations/best/n{n}.json (ours; coordinates are exact decimal expansions of IEEE doubles)
  n <= 36   : published records of https://math.tejstead.com/heilbronn. They are NOT redistributed here.
              Download them yourself into records/n{n}.txt (see records/README.md); load_record() parses them.
  n = 5..18 : configurations/small_n/optima_small.json (rebuilt by our own search)
"""
import os
import re
import json
import itertools
from fractions import Fraction

CODE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(CODE)
CONF = os.path.join(REPO, 'configurations')
CERT = os.path.join(REPO, 'certificates')
RECORDS_DIR = os.environ.get('HEILBRONN_RECORDS', os.path.join(REPO, 'records'))


def jl(path):
    """Read a .jsonl file."""
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------------------------------------------------- exact arithmetic
def to_fraction(v):
    """Exact rational value of a decimal string, a 'p/q' string, an int or a float (floats exactly, as binary)."""
    return v if isinstance(v, Fraction) else Fraction(v)


def exact_areas2(P):
    """Twice the absolute areas of all C(n,3) triangles, exactly: returns (list of ints, common denominator D2) with
    area_T = ints[T] / (2 * D2). Coordinates are put on a common denominator so all arithmetic is integer."""
    Q = [(to_fraction(x), to_fraction(y)) for x, y in P]
    den = 1
    for x, y in Q:
        for c in (x, y):
            d = c.denominator
            den = den * d // _gcd(den, d)
    X = [int(x * den) for x, _ in Q]
    Y = [int(y * den) for _, y in Q]
    out = []
    for a, b, c in itertools.combinations(range(len(Q)), 3):
        out.append(abs((X[b] - X[a]) * (Y[c] - Y[a]) - (X[c] - X[a]) * (Y[b] - Y[a])))
    return out, den * den


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def exact_min_area(P, rels=(Fraction(1, 10 ** 9), Fraction(1, 10 ** 6))):
    """Exact minimum triangle area of P (Fraction) and tie counts: 'equal' (area exactly = min) and, for each relative
    tolerance r in rels, the number of triangles with area <= min * (1 + r)."""
    A, D2 = exact_areas2(P)
    m = min(A)
    counts = {'equal': sum(1 for a in A if a == m)}
    for r in rels:
        lim = m * (1 + r)                      # compare a <= m (1 + r) exactly
        counts[f'rel_{float(r):.0e}'] = sum(1 for a in A if a <= lim)
    return Fraction(m, 2 * D2), counts


def inside_unit_square(P):
    return all(0 <= to_fraction(x) <= 1 and 0 <= to_fraction(y) <= 1 for x, y in P)


# ---------------------------------------------------------------- loading
def load_best(n):
    """Our best configuration for 37 <= n <= 100 as an elites-style dict: n, value (float), points (floats),
    points_exact (Fractions), family, source."""
    d = json.load(open(os.path.join(CONF, 'best', f'n{n}.json')))
    pts = [(to_fraction(x), to_fraction(y)) for x, y in d['points']]
    return dict(n=n, value=d['value_float'], points=[[float(x), float(y)] for x, y in pts], points_exact=pts,
                family=d.get('family', 'own'), source=f'configurations/best/n{n}.json')


def parse_record_file(path):
    """Parse a record-site coordinate file: '#' comment lines, then rows whose last two fields are x y
    (tab/space/comma separated). Decimal literals are read exactly. Returns (points as Fractions, header lines)."""
    pts, notes = [], []
    for line in open(path, encoding='utf-8'):
        s = line.strip()
        if not s:
            continue
        if s.startswith('#'):
            notes.append(s[1:].strip())
            continue
        parts = [x for x in re.split(r'[\s,;]+', s) if x]
        if len(parts) < 2:
            continue
        if not all(re.fullmatch(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', x) for x in parts[-2:]):
            continue                                            # header row
        pts.append((Fraction(parts[-2]), Fraction(parts[-1])))
    return pts, notes


def record_path(n):
    for name in (f'n{n}.txt', f'heilbronn_n{n}_points.txt', f'square-n{n}.txt'):
        p = os.path.join(RECORDS_DIR, name)
        if os.path.exists(p):
            return p
    return None


def load_record(n):
    """Published record for 19 <= n <= 36 from a file you downloaded into records/ (not redistributed here)."""
    p = record_path(n)
    if p is None:
        raise FileNotFoundError(
            f'record n={n} not found in {RECORDS_DIR}. The record coordinates are not redistributed in this repository; '
            f'download the n={n} square configuration from https://math.tejstead.com/heilbronn and save it as '
            f'records/n{n}.txt (see records/README.md).')
    pts, notes = parse_record_file(p)
    return dict(n=n, value=float(exact_min_area(pts, rels=())[0]), points=[[float(x), float(y)] for x, y in pts],
                points_exact=pts, family='record', source=f'records/{os.path.basename(p)} (published record)')


def load(n):
    """Elites-compatible loader used by the certificate scripts: a one-element list with the current best
    (records for n <= 36 if downloaded, ours for n >= 37)."""
    if n <= 36:
        try:
            return [load_record(n)]
        except FileNotFoundError:
            return []
    p = os.path.join(CONF, 'best', f'n{n}.json')
    return [load_best(n)] if os.path.exists(p) else []


def current(n):
    """Current best for 19 <= n <= 100: records (must be downloaded) for n <= 36, ours above."""
    return load_record(n) if n <= 36 else load_best(n)


def record_values():
    """Published record values 19 <= n <= 36 (exact minimum of the published coordinates, as float)."""
    d = json.load(open(os.path.join(CONF, 'record_values.json')))
    return {int(k): v['min_area_exact_float'] for k, v in d['records'].items()}


def current_value(n):
    if n <= 36:
        return record_values()[n]
    return json.load(open(os.path.join(CONF, 'best', f'n{n}.json')))['value_float']


def small_n_optima():
    return json.load(open(os.path.join(CONF, 'small_n', 'optima_small.json')))
