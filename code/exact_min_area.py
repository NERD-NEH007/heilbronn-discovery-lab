"""Exact minimum triangle area and tie counts, in rational arithmetic (no floating point).

  python code/exact_min_area.py                     # check every configurations/best/n*.json against its stored fields
  python code/exact_min_area.py 40 51 97            # selected sizes
  python code/exact_min_area.py --file F.json       # any JSON with "points": [[x, y], ...] (decimal or p/q strings, or numbers)
  python code/exact_min_area.py --file F.txt        # a whitespace/comma separated x y text file ('#' comments)

Decimal strings are read exactly (Fraction('0.1') = 1/10); JSON numbers are read as IEEE doubles and then converted
exactly. Counts: 'equal' = triangles of area exactly t; 'rel_1e-09' / 'rel_1e-06' = triangles with area <= t (1 + r).
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlib


def check_file(path):
    if path.endswith('.json'):
        d = json.load(open(path)); P = d['points'] if isinstance(d, dict) else d[0]['points']
    else:
        P = hlib.parse_record_file(path)[0]
    m, c = hlib.exact_min_area(P)
    n = len(P)
    print(f'{os.path.basename(path)}: n={n} inside={hlib.inside_unit_square(P)} t={float(m):.17g} n^2 t={float(m) * n * n:.6f} '
          f'counts={c}')
    return m, c


def check_best(n):
    p = os.path.join(hlib.CONF, 'best', f'n{n}.json'); d = json.load(open(p))
    m, c = hlib.exact_min_area(d['points'])
    ok = (f'{m.numerator}/{m.denominator}' == d['min_area_exact'] and c['equal'] == d['ties_exact_equal']
          and c['rel_1e-09'] == d['count_rel_1e9'] and c['rel_1e-06'] == d['count_rel_1e6'] and hlib.inside_unit_square(d['points'])
          and len(d['points']) == n)
    print(f'n={n:3d} t={float(m):.15e} n^2t={float(m) * n * n:.5f} ties(=t) {c["equal"]:3d} (1e-9) {c["rel_1e-09"]:3d} '
          f'(1e-6) {c["rel_1e-06"]:3d}  {"OK" if ok else "MISMATCH"}', flush=True)
    return ok


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == '--file':
        for f in a[1:]: check_file(f)
        sys.exit(0)
    sizes = [int(x) for x in a] or sorted(int(os.path.basename(f)[1:-5]) for f in glob.glob(os.path.join(hlib.CONF, 'best', 'n*.json')))
    bad = [n for n in sizes if not check_best(n)]
    print('all OK' if not bad else f'MISMATCH at {bad}'); sys.exit(1 if bad else 0)
