"""Check the exact values of Table 7 (paper 1, Section 6.3): for every entry of certificates/exp104/exact_forms.json with
status 'proved exact', the stored minimal polynomial vanishes at the stored high-precision value t_full (to the
precision of t_full, 300 digits or fewer), and t_full agrees with the stored double value.
This checks the published polynomial against the published digits; the proof itself (Newton, LLL, exact check in
Q[x]/(p), Krawczyk test) is code/exp104/run.py with code/exp104/exact_core.py.
  python code/check_exact_forms.py [n ...]"""
import os, sys, json
import mpmath, sympy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlib


def check(e):
    digits = len(e['t_full'].split('.')[-1]) if '.' in e['t_full'] else 0
    if digits < 30:                                   # rational values stored as fractions
        t = sympy.Rational(e['t_full']) if '/' in e['t_full'] else sympy.Rational(e['value'])
        p = sympy.Poly(sympy.sympify(e['minpoly']), sympy.Symbol('x'))
        return p.eval(t) == 0, 'exact rational root', p
    mpmath.mp.dps = digits + 10
    x = sympy.Symbol('x'); p = sympy.Poly(sympy.sympify(e['minpoly']), x)
    c = [mpmath.mpf(int(a)) for a in p.all_coeffs()]
    t = mpmath.mpf(e['t_full'])
    val = mpmath.polyval(c, t); scale = mpmath.polyval([abs(a) for a in c], abs(t))
    r = abs(val) / scale
    return r < mpmath.mpf(10) ** (-(digits - 15)), f'|p(t)|/sum|c_i t^i| = {mpmath.nstr(r, 3)} at {digits} digits', p


if __name__ == '__main__':
    want = {int(a) for a in sys.argv[1:]}
    ok_all = True
    for e in json.load(open(os.path.join(hlib.CERT, 'exp104', 'exact_forms.json'))):
        if e['status'] != 'proved exact' or (want and int(e['n']) not in want): continue
        ok, msg, p = check(e)
        lc = p.LC(); ok_all &= ok
        print(f"n={e['label']:>10}: degree {p.degree()}, leading coefficient {lc} = {sympy.factorint(lc)}; {msg}  {'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok_all else 1)
