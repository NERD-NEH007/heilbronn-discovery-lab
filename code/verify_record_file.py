"""Verify a record configuration that you downloaded yourself from https://math.tejstead.com/heilbronn.

The record coordinates (19 <= n <= 36) are not redistributed in this repository. Save the site's coordinate file for
the unit square as records/n{n}.txt (any text file whose data rows end with 'x y' works; '#' lines are comments) and run

  python code/verify_record_file.py records/n22.txt [more files]
  python code/verify_record_file.py --all          # every records/n*.txt

The file is read exactly (decimal literals -> rationals). Reported: number of points, inside the unit square, exact
minimum area, tie counts, and agreement with the value used in the papers (configurations/record_values.json,
the exact minimum of the coordinates published at the time of writing). A record improved on the site since then
will show a different value: that is reported, not an error in the file.
"""
import os, sys, re, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlib


def verify(path, n=None):
    P, notes = hlib.parse_record_file(path)
    n = n or len(P)
    m, c = hlib.exact_min_area(P, rels=(hlib.Fraction(1, 10 ** 12), hlib.Fraction(1, 10 ** 9), hlib.Fraction(1, 10 ** 6)))
    RV = hlib.record_values(); ref = RV.get(n)
    rel = float(m) / ref - 1 if ref else float('nan')
    print(f'{os.path.basename(path)}: points {len(P)} inside {hlib.inside_unit_square(P)} distinct {len(set(P)) == len(P)} '
          f't = {float(m):.16g}  ties {c}')
    if ref:
        print(f'   value used in the papers {ref:.16g}: relative difference {rel:+.2e} '
              f'{"(same configuration value)" if abs(rel) < 1e-12 else "(DIFFERENT: record changed or wrong file)"}')
    return abs(rel) < 1e-12 if ref else None


if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(hlib.RECORDS_DIR, 'n*.txt'))) if sys.argv[1:] == ['--all'] else sys.argv[1:]
    if not files: print(__doc__); sys.exit(0)
    res = [verify(f) for f in files]
    sys.exit(0 if all(r is not False for r in res) else 1)
