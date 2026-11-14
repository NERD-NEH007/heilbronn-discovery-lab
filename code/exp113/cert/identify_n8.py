"""Exp 113 / cert: exact identification of the certified triangle n = 8 optimum with the Chen-Zeng-Zhou septic.

Septic (math.tejstead.com/heilbronn/triangle/8/, for the unit right triangle, value A = 2t; t = min area):
    p(t) = 1216512 t^7 + 1022976 t^6 - 2745456 t^5 + 2218730 t^4 + 32206304 t^3 - 19148615 t^2 + 3156103 t - 86233

Discovery (not proof): Newton polish of the certificate's reduced tie system F(u, t) = 0 to ~1e-400 in exact rationals,
then mpmath PSLQ expresses every parameter u_l as a rational polynomial c_l(theta) of degree <= 6 in theta = t.
Proof (exact, sympy QQ polynomials and Fractions only):
  1. p is irreducible over Q and has exactly one root theta* in the certified enclosure [t_lower, t_upper]
     (exact sign change + Sturm count), and exactly one real root overall.
  2. Every reduced tie equation  sigma_k A_{T_k}(p(c(theta))) - theta  is 0 modulo p(theta)  (exact remainder),
     so w_alg = (c(theta*), theta*) is an exact zero of F.
  3. |w_alg - w~|_inf < 1e-6 (rigorous bound from the enclosure of theta*), and the Krawczyk test in certify_tri
     holds for the radius 1e-6 (uniqueness_radius_1e6), so F has exactly one zero there: w_alg = w*.
  Hence the certified optimum's minimum area t* is EXACTLY theta*, the real root of p (site value 2 theta*).
"""
import os, sys, json
from fractions import Fraction as Fr
import sympy as sp
import mpmath as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import certify_tri as CT

COEF = [1216512, 1022976, -2745456, 2218730, 32206304, -19148615, 3156103, -86233]   # t^7 ... t^0
th = sp.symbols('theta')
p = sp.Poly(COEF, th, domain='QQ')

def ev(poly_coeffs_low, x):                # exact evaluation, coefficients low -> high
    s = Fr(0)
    for c in reversed(poly_coeffs_low): s = s * x + c
    return s

def solve_exact(A, b):                     # Gaussian elimination in Fractions
    n = len(A); M = [list(r) + [bb] for r, bb in zip(A, b)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c])); M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c] / M[c][c]; M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]

def main():
    o = CT.certify(8, CT.load_input(8)['points'], keep_ctx=True)
    assert o['certified'] and o['uniqueness_radius_1e6']
    cx = o['_ctx']; par, eqs, du = cx['par'], cx['eqs'], cx['du']
    tl = Fr(int(o['t_lower'].split('e')[0]), 10 ** 32); tu = Fr(int(o['t_upper'].split('e')[0]), 10 ** 32)
    out = {}
    # ---- 1. septic facts
    fl = sp.factor_list(p.as_expr()); out['irreducible'] = len(fl[1]) == 1 and fl[1][0][1] == 1
    pl = [Fr(int(c.p), int(c.q)) for c in reversed(p.all_coeffs())]
    sl, su = ev(pl, tl), ev(pl, tu)
    out['sign_change_on_enclosure'] = (sl < 0 < su) or (su < 0 < sl)
    out['roots_in_enclosure'] = int(p.count_roots(sp.Rational(tl.numerator, tl.denominator), sp.Rational(tu.numerator, tu.denominator)))
    out['real_roots_total'] = int(p.count_roots())
    print('septic:', out, flush=True)
    # ---- discovery: high-precision polish (exact Newton, grid 2^-720)
    u, tv = list(cx['u']), cx['tv']; K = 1400
    for it in range(8):
        P = CT.positions(par, u)
        Fv = [s * CT.area2(P, T) / 2 - tv for T, s in eqs]
        res = max(abs(x) for x in Fv)
        print(f'  polish it {it}: residual {float(res):.2e}', flush=True)
        if res < Fr(1, 10 ** 400): break
        J = CT.dense(CT.jac_rows(par, P, eqs, du), du + 1, conv=Fr)
        st = solve_exact(J, Fv)
        u = [CT.grid(x - s, K) for x, s in zip(u, st[:du])]; tv = CT.grid(tv - st[du], K)
    mp.mp.dps = 420
    tm = mp.mpf(tv.numerator) / tv.denominator
    basis = [tm ** k for k in range(7)]
    cpolys = []
    for l, x in enumerate(u):
        xm = mp.mpf(x.numerator) / x.denominator
        rel = mp.pslq([xm] + basis, maxcoeff=10 ** 80, maxsteps=10 ** 6)
        if rel is None or rel[0] == 0:
            print(f'  u_{l}: PSLQ found no relation of degree <= 6 in t'); out['recognised'] = False; break
        cpolys.append([Fr(-int(c), int(rel[0])) for c in rel[1:]])          # low -> high
        print(f'  u_{l} = ' + ' + '.join(f'({c})t^{k}' for k, c in enumerate(cpolys[-1]) if c), flush=True)
    else:
        out['recognised'] = True
    if not out['recognised']:
        json.dump(out, open(os.path.join(HERE, 'identify_n8.json'), 'w'), indent=1); return
    # ---- 2. exact: tie equations vanish mod p
    U = [sp.Poly([sp.Rational(c.numerator, c.denominator) for c in reversed(cp)], th, domain='QQ') for cp in cpolys]
    R = lambda f: sp.Rational(Fr(f).numerator, Fr(f).denominator)
    PP = []                                           # affine in u with rational coefficients -> polys in theta
    for a, Bl in par:
        x = sp.Poly(R(a[0]), th, domain='QQ'); y = sp.Poly(R(a[1]), th, domain='QQ')
        for l, b in Bl: x = x + U[l] * R(b[0]); y = y + U[l] * R(b[1])
        PP.append((x, y))
    zero = True
    for T, s in eqs:
        a, b, c = PP[T[0]], PP[T[1]], PP[T[2]]
        e = ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) * sp.Rational(s, 2) - sp.Poly(th, th, domain='QQ')
        if not e.rem(p).is_zero: zero = False
    out['tie_equations_vanish_mod_p'] = zero
    # ---- 3. rigorous distance |w_alg - w~| over theta* in [tl, tu]
    B = max(abs(tl), abs(tu)); width = tu - tl; dist = abs(Fr((tl + tu) / 2) - cx['tv']) + width
    for cp, uc in zip(cpolys, cx['u']):
        lip = sum(k * abs(c) * B ** (k - 1) for k, c in enumerate(cp) if k)
        dist = max(dist, abs(ev(cp, tl) - uc) + lip * width)
    out['dist_alg_to_centre_upper'] = float(dist)
    out['within_uniqueness_ball_1e6'] = dist < Fr(1, 10 ** 6)
    out['coords_in_Q_theta'] = [[str(c) for c in cp] for cp in cpolys]
    out['proved_t_equals_septic_root'] = bool(out['roots_in_enclosure'] == 1 and zero
                                              and out['within_uniqueness_ball_1e6'])
    print({k: v for k, v in out.items() if k != 'coords_in_Q_theta'})
    json.dump(out, open(os.path.join(HERE, 'identify_n8.json'), 'w'), indent=1)
    cf = os.path.join(HERE, 'local_certs_tri.json')
    if out['proved_t_equals_septic_root'] and os.path.exists(cf):
        d = json.load(open(cf))
        d['8'][0]['t_exact'] = ('unique real root of 1216512 t^7 + 1022976 t^6 - 2745456 t^5 + 2218730 t^4 + 32206304 t^3 '
                                '- 19148615 t^2 + 3156103 t - 86233 (irreducible over Q); proved by identify_n8.py')
        json.dump(d, open(cf, 'w'), indent=1)

if __name__ == '__main__':
    main()
