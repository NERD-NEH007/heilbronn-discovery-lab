"""Exp 104 driver: exact forms of best-known min areas.  python run.py --sizes 5,6,7 [--digits 250] [--maxdeg 24]
Writes results/n{n}.json (one per size; for n=6 several configurations)."""
import os, sys, json, time, argparse, itertools, random, math
from fractions import Fraction as Fr
import numpy as np, scipy.linalg, scipy.optimize
import flint
from flint import arb, arb_mat, fmpz, fmpq, fmpq_poly, fmpz_poly, ctx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_core import *

DEGS = None
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); import hlib


def load(n):
    """list of (label, points)"""
    if n <= 18:
        d = hlib.small_n_optima()[str(n)]
        if n == 6:  # primary (4 corners + 2) and one shear-family member
            return [('6 primary', d[0]['points']), ('6 family', d[3]['points'])]
        return [(str(n), d[0]['points'])]
    e = hlib.current(n)          # n = 19..36: published record from records/ (see records/README.md)
    return [(str(n), e['points'])]


def build(P, act, log):
    ties, sg, core, walls = act['ties'], act['sg'], act['core'], act['walls']
    eqs = [('tie', tr, s) for tr, s in zip(ties, sg)] + [('wall', i, e, v) for i, e, v, s in walls]
    S0 = System(core, eqs); N = S0.N
    z0 = np.zeros(N)
    for i, c in S0.ix.items(): z0[2 * c], z0[2 * c + 1] = P[i]
    z0[S0.it] = act['t']
    J = dense(S0.J(list(z0)), N)
    sv = np.linalg.svd(J, compute_uv=False); rank = int((sv > 1e-8 * sv[0]).sum())
    info = dict(eqs=len(eqs), unknowns=N, rank=rank)
    if rank == N:
        q, r, piv = scipy.linalg.qr(J.T, pivoting=True); sel = sorted(piv[:N])
        info['mode'] = 'primal'
        return System(core, [eqs[k] for k in sel]), z0, info, S0
    u, s_, vt = np.linalg.svd(J); V = vt[rank:].T  # null space
    tcomp = float(np.abs(V[S0.it]).max()); info['t_null'] = tcomp
    family = False
    if tcomp < 1e-6:
        # t is stationary along the null directions: a flat family (t constant) or a second-order maximum (t drops).
        # Test: move along each null direction by 1e-3, re-solve the tie system with that gauge, compare t.
        q, r, piv = scipy.linalg.qr(V.T, pivoting=True); fixv = list(piv[:N - rank])
        q2, r2, piv2 = scipy.linalg.qr(J.T, pivoting=True); sel = sorted(piv2[:rank])
        drops = []
        for k in range(N - rank):
            z1 = z0 + 1e-3 * V[:, k]
            fx = [('fix', int(v), Fr(z1[v]).limit_denominator(10**6)) for v in fixv]
            St = System(core, [eqs[q_] for q_ in sel] + fx)
            for _ in range(40): z1 = z1 - np.linalg.lstsq(dense(St.J(list(z1)), N), np.array(St.F(list(z1)), float), rcond=None)[0]
            drops.append(float(z1[S0.it] - z0[S0.it]))
        info['null_t_drop'] = drops
        family = max(abs(x) for x in drops) < 1e-11
    if family:
        # flat family: gauge-fix null directions at nearby simple rationals
        q, r, piv = scipy.linalg.qr(V.T, pivoting=True); fixv = list(piv[:N - rank])
        q2, r2, piv2 = scipy.linalg.qr(J.T, pivoting=True); sel = sorted(piv2[:rank])
        fixes = [('fix', int(v), Fr(z0[v]).limit_denominator(1000)) for v in fixv]
        info['mode'] = 'primal+gauge'; info['gauge'] = [(f[1], str(f[2])) for f in fixes]
        z1 = z0.copy()
        for f in fixes: z1[f[1]] = float(f[2])
        # re-project the float start onto the system with the gauge
        S = System(core, [eqs[k] for k in sel] + fixes)
        for _ in range(30): z1 = z1 - np.linalg.lstsq(dense(S.J(list(z1)), N), np.array(S.F(list(z1)), float), rcond=None)[0]
        return S, z1, info, S0
    # KKT: vertex multipliers by LP (support has independent columns)
    nt, nw = len(ties), len(walls)
    G = np.zeros((S0.nx + 1, nt + nw))
    for k in range(nt):
        for m, gv in S0.area_grad(list(z0), ties[k], sg[k]).items(): G[m, k] += gv
        G[S0.nx, k] = 1
    for w, (i, e, v, sw) in enumerate(walls): G[2 * S0.ix[i] + e, nt + w] = sw
    b = np.zeros(S0.nx + 1); b[S0.nx] = 1
    rng = np.random.default_rng(1)
    lp = scipy.optimize.linprog(rng.uniform(1, 2, nt + nw), A_eq=G, b_eq=b, bounds=[(0, None)] * (nt + nw), method='highs-ds')
    if lp.x is not None: m = lp.x
    else: m, rn = scipy.optimize.nnls(G, b); info['nnls_res'] = float(rn)
    supp = [j for j in range(nt + nw) if m[j] > 1e-9]
    Gs = G[:, supp]; info['support_rank'] = int(np.linalg.matrix_rank(Gs, 1e-9)); info['support_size'] = len(supp)
    T = [ties[j] for j in supp if j < nt]; SG = [sg[j] for j in supp if j < nt]; W = [walls[j - nt] for j in supp if j >= nt]
    lam = [m[j] for j in supp if j < nt]; mu = [m[j] for j in supp if j >= nt]
    info['mode'] = 'kkt'; info['support'] = [len(T), len(W)]; info['d'] = N - len(T) - len(W)
    eqsS = [('tie', tr, s) for tr, s in zip(T, SG)] + [('wall', i, e, v) for i, e, v, s in W]
    S = System(core, eqsS, kkt=(T, SG, W))
    z1 = np.concatenate([z0, lam, mu])
    return S, z1, info, S0


def check_all(P, act, S, S0, z, digits):
    """residuals of all active equations (S0) at z, min of non-active triangles minus t, coordinate shift."""
    zc = z[:S0.N]
    res = max((abs(f) for f in S0.F(zc)), default=arb(0))
    t = zc[S0.it]; X = {i: S0.X(zc, i) for i in S0.core}
    for i in act['rat']: X[i] = (arb(P[i][0]), arb(P[i][1]))
    tieset = set(act['ties']); gap = None
    for tr in itertools.combinations(range(len(P)), 3):
        if tr in tieset: continue
        a = abs(det(X[tr[0]], X[tr[1]], X[tr[2]])) / 2 - t
        gap = a if gap is None or a < gap else gap
    shift = max(abs(float(X[i][e]) - P[i][e]) for i in S0.core for e in (0, 1))
    return float(arb(res.mid())) if res != 0 else 0.0, float(gap / t), shift


def polymod_eval(expr_polys, m):
    pass


def exact_verify(S, z, gens_poly, mpoly):
    """gens_poly: list of fmpq_poly (one per variable) in a primitive element; mpoly: its minimal polynomial (fmpq_poly).
    Evaluate every equation of S in Q[x]/(mpoly) exactly. Returns True if all are 0."""
    zz = [g % mpoly for g in gens_poly]
    class R:  # residue ring element
        __slots__ = ('p',)
        def __init__(s, p): s.p = p
        def __add__(s, o): return R((s.p + (o.p if isinstance(o, R) else fmpq_poly([o]))) % mpoly)
        __radd__ = __add__
        def __sub__(s, o): return R((s.p - (o.p if isinstance(o, R) else fmpq_poly([o]))) % mpoly)
        def __rsub__(s, o): return R((fmpq_poly([o]) - s.p) % mpoly)
        def __mul__(s, o):
            if isinstance(o, R): return R((s.p * o.p) % mpoly)
            if isinstance(o, float): o = Fr(o)
            if isinstance(o, Fr): return R(s.p * fmpq(o.numerator, o.denominator))
            return R(s.p * o)
        __rmul__ = __mul__
        def __truediv__(s, o): return R(s.p / o)
        def __neg__(s): return R(-s.p)
    vals = [R(g) for g in zz]
    out = S.F(vals)
    bad = [k for k, f in enumerate(out) if not (f.p if isinstance(f, R) else fmpq_poly([f])).is_zero()]
    return len(bad) == 0, bad


def radical_info(poly):
    import sympy as sp
    x = sp.Symbol('x'); c = [int(a) for a in poly.coeffs()]
    P = sum(sp.Integer(ci) * x ** i for i, ci in enumerate(c))
    info = dict(discriminant=str(sp.discriminant(P, x)))
    D = sp.discriminant(P, x)
    sf = sp.sqrt(D).simplify()
    info['sqrt_disc'] = str(sf)
    if poly.degree() <= 6:
        try:
            G, alt = sp.polys.numberfields.galoisgroups.galois_group(sp.Poly(P, x), by_name=True)
            info['galois'] = str(G)
        except Exception as ex: info['galois'] = f'n/a ({type(ex).__name__})'
    return info, P, x


def radical_form(poly, tval):
    import sympy as sp
    _, P, x = radical_info(poly)
    if poly.degree() > 4: return None
    best = None
    for r in sp.roots(sp.Poly(P, x), cubics=True, quartics=True, multiple=True):
        try: v = complex(sp.N(r, 40))
        except Exception: continue
        if abs(v - tval) < 1e-12 * max(1, abs(tval)):
            best = r; break
    if best is None: return None
    s = sp.nsimplify(best) if poly.degree() == 1 else sp.simplify(best)
    if poly.degree() == 3 and 'I' in str(s):  # casus irreducibilis: trigonometric form
        c3, c2, c1, c0 = [sp.Rational(int(a)) for a in reversed(poly.coeffs())]
        a, b, c = c2 / c3, c1 / c3, c0 / c3  # x^3 + a x^2 + b x + c, x = y - a/3
        p = b - a ** 2 / 3; q = 2 * a ** 3 / 27 - a * b / 3 + c
        for k in range(3):
            y = 2 * sp.sqrt(-p / 3) * sp.cos(sp.acos(3 * q / (2 * p) * sp.sqrt(-3 / p)) / 3 - 2 * sp.pi * k / 3)
            xr = sp.nsimplify(y - a / 3) if False else sp.simplify(y - a / 3)
            if abs(float(sp.N(xr, 30)) - tval) < 1e-12: return 'trig: ' + str(xr)
    return str(s)


def solve_one(label, P, digits, maxdeg, log):
    n = len(P); t0 = time.time()
    act = active_set(P); out = dict(label=label, n=n, ties=len(act['ties']), walls=len(act['walls']), core=len(act['core']), rattlers=len(act['rat']))
    Q = [(Fr(x).limit_denominator(1000), Fr(y).limit_denominator(1000)) for x, y in P]
    if max(abs(float(Q[i][e]) - P[i][e]) for i in range(n) for e in (0, 1)) < 1e-12:
        tq = min(abs(det(Q[i], Q[j], Q[k])) / 2 for i, j, k in itertools.combinations(range(n), 3))
        if abs(float(tq) - act['t']) < 1e-14:
            out.update(mode='rational', status='proved exact', minpoly=f'{tq.denominator}*x - {tq.numerator}', degree=1,
                       radical=str(tq), field_degree=1, t_digits=str(tq), coords=[[str(a), str(b)] for a, b in Q],
                       note='coordinates are rationals (denominators <= 1000); min area computed exactly with Fractions')
            log(f'[{label}] rational configuration, min area exactly {tq}'); return out
    S, z0, info, S0 = build(P, act, log); out.update(info)
    log(f'[{label}] ties {out["ties"]} walls {out["walls"]} core {out["core"]} rat {out["rattlers"]} mode {info["mode"]} N {S.N} rank {info["rank"]}')
    z = newton_arb(S, list(z0), digits, log=None)
    res_sq = max(abs(f) for f in S.F(z))
    res_all, gap, shift = check_all(P, act, S, S0, z, digits)
    out.update(residual_square=float(arb(res_sq.mid())) if res_sq != 0 else 0.0, residual_all=res_all, gap_rel=gap, shift=shift)
    t = z[S.it]; out['t_digits'] = t.str(digits, radius=False)
    log(f'[{label}] t = {t.str(40, radius=False)}  res_all {res_all:.1e} gap {gap:.3g} shift {shift:.1e}')
    if res_all > 10.0 ** (-(digits - 20)):
        out['status'] = 'active system inconsistent'; return out
    if gap <= 0: out['status'] = 'another triangle below t'; return out
    if info['mode'] == 'primal+gauge':  # t must not depend on the gauge: re-solve at shifted gauge
        S2 = System(S.core, [e if e[0] != 'fix' else ('fix', e[1], e[2] + Fr(1, 997)) for e in S.eqs])
        z2 = newton_arb(S2, [float(v) for v in z], digits)
        out['gauge_t_change'] = float(abs(z2[S.it] - t).mid())
    ok, kw = krawczyk(S, z, rad_digits=digits // 3)
    out['krawczyk'] = ok; out['krawczyk_radius'] = f'1e-{digits // 3}'
    log(f'[{label}] Krawczyk unique zero in ball 1e-{digits // 3}: {ok}')
    mp = find_minpoly(t, digits, maxdeg=maxdeg, degs=DEGS, log=log)
    if mp is None:
        import exact_core
        out['status'] = f'not found up to degree {max(DEGS) if DEGS else maxdeg}'; out['lll_last'] = dict(exact_core.LAST)
        out['secs'] = round(time.time() - t0, 1); return out
    p = mp['poly']; D = p.degree()
    out.update(degree=D, minpoly=str(p), minpoly_coeffs=[int(a) for a in p.coeffs()], lll_digits=mp['lll_digits'], height_digits=mp['height_digits'])
    log(f'[{label}] minpoly deg {D}: {p}')
    # coordinates (and multipliers) in Q(t)?
    gens = None; prim = None
    basis = powers(t, D - 1)[1:] if D > 1 else []
    def express_all(theta, Dth, mth):
        bas = powers(theta, Dth - 1)
        polys = []
        for i, zi in enumerate(z):
            if i == S.it and theta is t: polys.append(fmpq_poly([0, 1])); continue
            q = express_in_basis(zi, bas, digits)
            if q is None: return None, i
            polys.append(fmpq_poly([fmpq(c.numerator, c.denominator) for c in q]))
        return polys, None
    polys, fail = express_all(t, D, p)
    if polys is not None: prim = dict(element='t', poly=p)
    else:
        log(f'[{label}] variable {fail} not in Q(t); trying a primitive element')
        rng = random.Random(7)
        for attempt in range(3):
            cs = [rng.randint(-3, 3) for _ in range(S.nx)]
            theta = sum((c * z[i] for i, c in enumerate(cs) if c), arb(0)) + t
            mt = find_minpoly(theta, digits, maxdeg=maxdeg)
            if mt is None: continue
            Dth = mt['poly'].degree(); polys, fail = express_all(theta, Dth, mt['poly'])
            if polys is not None:
                prim = dict(element=f'theta = t + sum c_i z_i, c={cs}', poly=mt['poly']); break
        if prim is None:
            out['status'] = f'guessed (minpoly of t stable to {digits} digits); coordinate field not found'
            out['secs'] = round(time.time() - t0, 1); return out
    mpq = fmpq_poly([fmpq(int(a)) for a in prim['poly'].coeffs()])
    out['field_generator'] = prim['element']; out['field_degree'] = prim['poly'].degree(); out['field_minpoly'] = str(prim['poly'])
    okS, bad = exact_verify(S, z, polys, mpq)
    okA, badA = exact_verify(S0, z[:S0.N], polys[:S0.N], mpq)
    # t's minpoly holds exactly in the field
    tp = polys[S.it]; acc = fmpq_poly([0])
    for c in reversed([int(a) for a in p.coeffs()]): acc = (acc * tp + fmpq(c)) % mpq
    okp = acc.is_zero()
    irreducible = len(p.factor()[1]) == 1 and p.factor()[1][0][1] == 1
    # rigorous embedding: real root of the field polynomial near theta; evaluate polys there; inside the Krawczyk ball?
    theta_num = t if prim['element'] == 't' else None
    inball = None
    try:
        roots = prim['poly'].complex_roots()
        th_val = polys_eval = None
        thv = t if prim['element'] == 't' else None
        if thv is None:
            cs = eval(prim['element'].split('c=')[1]); thv = sum((c * z[i] for i, c in enumerate(cs) if c), arb(0)) + t
        best = min(roots, key=lambda r: abs(r[0].real - thv).mid() if hasattr(r[0], 'real') else 1)
        rt = best[0].real
        r = arb(10) ** (-(digits // 3)); inball = True
        for i, g in enumerate(polys):
            v = sum((arb(fmpq(g.coeffs()[j])) * rt ** j for j in range(len(g.coeffs()))), arb(0)) if g.coeffs() else arb(0)
            if not (abs(v - z[i]) < r): inball = False; break
    except Exception as ex:
        out['embed_error'] = repr(ex)
    names = [f'{S.core[k // 2]}.{"xy"[k % 2]}' for k in range(S.nx)] + ['t'] + [f'mult{k}' for k in range(S.N - S.nx - 1)]
    out['exact_values'] = {nm: [str(c) for c in g.coeffs()] for nm, g in zip(names, polys)}
    out['exact_values_note'] = 'each value = sum_j c_j g^j, g = field generator (coefficient lists, low degree first)'
    out['rattler_coords'] = {str(i): list(P[i]) for i in act['rat']}
    try:
        import sympy as sp
        if prim['poly'].degree() <= 2 and prim['element'] == 't':
            rad = sp.sympify(radical_form(p, float(t)))
            out['coords_radical'] = {nm: str(sp.nsimplify(sp.expand(sum(sp.Rational(str(c)) * rad ** j for j, c in enumerate(g.coeffs()))))) for nm, g in zip(names[:S.nx], polys)}
    except Exception as ex: out['coords_radical_error'] = repr(ex)
    out.update(exact_square=okS, exact_all_active=okA, exact_bad_rows=badA[:10], exact_minpoly_t=okp, minpoly_irreducible=irreducible, embedding_in_ball=inball)
    proved = okS and okA and okp and irreducible and ok and bool(inball) and gap > 0
    if info['mode'] == 'primal+gauge':
        out['note'] = 'flat family: exact point at a rational gauge; t constant along the family (checked numerically)'
    out['status'] = 'proved exact' if proved else f'guessed (PSLQ/LLL stable to {digits} digits)'
    try:
        ri, _, _ = radical_info(p); out.update(ri)
        rf = radical_form(p, float(t)); out['radical'] = rf
    except Exception as ex: out['radical_error'] = repr(ex)
    out['secs'] = round(time.time() - t0, 1)
    log(f'[{label}] status {out["status"]}  field deg {out["field_degree"]}  radical {out.get("radical")}  ({out["secs"]} s)')
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sizes', default='5'); ap.add_argument('--digits', type=int, default=250)
    ap.add_argument('--maxdeg', type=int, default=24); ap.add_argument('--outdir', default=''); ap.add_argument('--degs', default=''); a = ap.parse_args()
    global DEGS; DEGS = [int(x) for x in a.degs.split(',')] if a.degs else None
    RES = os.path.join(hlib.REPO, 'output', 'exp104'); os.makedirs(os.path.join(RES, a.outdir), exist_ok=True)
    def log(s): print(s, flush=True)
    for n in [int(x) for x in a.sizes.split(',')]:
        outs = []
        for label, P in load(n):
            P = [tuple(p) for p in P]
            try: outs.append(solve_one(label, P, a.digits, a.maxdeg, log))
            except Exception as ex:
                import traceback; traceback.print_exc(); outs.append(dict(label=label, n=n, status=f'error {ex!r}'))
        json.dump(outs, open(os.path.join(RES, a.outdir, f'n{n}.json'), 'w'), indent=1, default=str)


if __name__ == '__main__': main()
