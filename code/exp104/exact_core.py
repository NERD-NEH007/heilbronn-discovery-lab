"""Exp 104 core: active system, high-precision Newton (arb), Krawczyk uniqueness, LLL guessing, exact verification.

Pipeline for one configuration P (floats):
 1. active set: ties (rel 1e-9 of min area), wall contacts (1e-11), core = points in ties, rattlers = the rest.
 2. system choice
    primal:  unknowns (core coords, t); equations  sg*det/2 - t = 0 (ties), coord - wall = 0.
             Jacobian full column rank -> pick a square, well-conditioned subset of rows (pivoted QR).
             rank-deficient with no t-component in the null space -> flat family: gauge-fix the null directions
             at nearby simple rationals (t does not change along them; checked by re-solving at another gauge).
    kkt:     t moves along the null space (d >= 1): maximise t; square KKT system on a vertex multiplier support S
             (unknowns coords, t, lambda_S, mu_S; equations ties_S, walls_S, stationarity, sum lambda = 1).
 3. Newton in arb at `digits` decimal digits (midpoints only), then residual of ALL active equations.
 4. Krawczyk test on the square system: a ball of radius rad around the solution contains exactly one zero.
 5. LLL (flint fmpz_mat.lll) for the minimal polynomial of t; stability: relation found from `digits_lll` digits must
    hold to the full precision. Coordinates expressed in Q(t) (or a primitive element) by LLL.
 6. exact check in Q[x]/(p): every active equation of the square system AND the dropped ties hold identically.
"""
import itertools, math, random
from fractions import Fraction as Fr
import numpy as np, scipy.linalg, scipy.optimize
import flint
from flint import arb, arb_mat, fmpz, fmpz_mat, fmpq, fmpq_poly, fmpz_poly, ctx


def det(a, b, c): return (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])


def active_set(P, rtol=1e-9, wtol=1e-11):
    n = len(P); tri = list(itertools.combinations(range(n), 3))
    A = np.array([abs(det(P[i], P[j], P[k])) / 2 for i, j, k in tri]); t = A.min()
    ties = [tri[q] for q in np.where(A <= t * (1 + rtol))[0]]
    sg = [1 if det(P[i], P[j], P[k]) > 0 else -1 for i, j, k in ties]
    core = sorted({i for tr in ties for i in tr}); rat = [i for i in range(n) if i not in core]
    walls = [(i, e, v, s) for i in core for e, v, s in ((0, 0, 1), (1, 0, 1), (0, 1, -1), (1, 1, -1)) if abs(P[i][e] - v) < wtol]
    return dict(ties=ties, sg=sg, core=core, rat=rat, walls=walls, t=float(t))


class System:
    """equations: list of ('tie', (i,j,k), sg) | ('wall', i, e, v) | ('fix', var, value) | stationarity (kkt).
    variables: 2 per core point, then t, then (kkt) multipliers."""
    def __init__(s, core, eqs, kkt=None):
        s.core = core; s.ix = {i: c for c, i in enumerate(core)}; s.nx = 2 * len(core); s.it = s.nx
        s.eqs = eqs; s.kkt = kkt  # kkt = (tie_list, sg_list, wall_list) of the support S
        s.N = s.nx + 1 + (len(kkt[0]) + len(kkt[2]) if kkt else 0)

    def X(s, z, i): c = s.ix[i]; return (z[2 * c], z[2 * c + 1])

    def area_grad(s, z, tr, sg):
        """dict var -> d(sg*det/2)/dvar"""
        g = {}
        for p, q, r in ((tr[0], tr[1], tr[2]), (tr[1], tr[2], tr[0]), (tr[2], tr[0], tr[1])):
            Q, R = s.X(z, q), s.X(z, r); c = s.ix[p]
            g[2 * c] = g.get(2 * c, 0) + (Q[1] - R[1]) * sg / 2
            g[2 * c + 1] = g.get(2 * c + 1, 0) + (R[0] - Q[0]) * sg / 2
        return g

    def area_hess(s, tr, sg):
        H = []
        for p, q, r in ((tr[0], tr[1], tr[2]), (tr[1], tr[2], tr[0]), (tr[2], tr[0], tr[1])):
            px, py = 2 * s.ix[p], 2 * s.ix[p] + 1; qx, qy, rx, ry = 2 * s.ix[q], 2 * s.ix[q] + 1, 2 * s.ix[r], 2 * s.ix[r] + 1
            h = Fr(sg, 2); H += [(px, qy, h), (px, ry, -h), (py, rx, h), (py, qx, -h)]
        return H  # list of (a, b, v): d2/da db contributions (each pair once, as d/da of the b-gradient... symmetric use)

    def F(s, z):
        out = []
        for e in s.eqs:
            if e[0] == 'tie':
                tr, sg = e[1], e[2]; out.append(det(s.X(z, tr[0]), s.X(z, tr[1]), s.X(z, tr[2])) * sg / 2 - z[s.it])
            elif e[0] == 'wall': out.append(s.X(z, e[1])[e[2]] - e[3])
            elif e[0] == 'fix': out.append(z[e[1]] * e[2].denominator - e[2].numerator)
        if s.kkt:
            T, SG, W = s.kkt; nl = len(T); st = [0] * s.nx
            for k, (tr, sg) in enumerate(zip(T, SG)):
                lam = z[s.nx + 1 + k]
                for m, gv in s.area_grad(z, tr, sg).items(): st[m] = st[m] + lam * gv
            for w, (i, e, v, sw) in enumerate(W): st[2 * s.ix[i] + e] = st[2 * s.ix[i] + e] + sw * z[s.nx + 1 + nl + w]
            out += st; out.append(1 - sum(z[s.nx + 1 + k] for k in range(nl)))
        return out

    def J(s, z, zero=0):
        rows = []
        for e in s.eqs:
            r = {}
            if e[0] == 'tie': r = s.area_grad(z, e[1], e[2]); r[s.it] = -1
            elif e[0] == 'wall': r = {2 * s.ix[e[1]] + e[2]: 1}
            elif e[0] == 'fix': r = {e[1]: e[2].denominator}
            rows.append(r)
        if s.kkt:
            T, SG, W = s.kkt; nl = len(T); st = [dict() for _ in range(s.nx)]
            for k, (tr, sg) in enumerate(zip(T, SG)):
                lam = z[s.nx + 1 + k]
                for m, gv in s.area_grad(z, tr, sg).items(): st[m][s.nx + 1 + k] = st[m].get(s.nx + 1 + k, 0) + gv
                for a, b, v in s.area_hess(tr, sg):
                    st[a][b] = st[a].get(b, 0) + lam * v.numerator / v.denominator
            for w, (i, e, v, sw) in enumerate(W): st[2 * s.ix[i] + e][s.nx + 1 + nl + w] = sw
            rows += st; rows.append({s.nx + 1 + k: -1 for k in range(nl)})
        return rows


def dense(rows, N):
    M = np.zeros((len(rows), N))
    for a, r in enumerate(rows):
        for b, v in r.items(): M[a, b] = float(v)
    return M


def to_arb_mat(rows, N):
    M = [[arb(0)] * N for _ in rows]
    for a, r in enumerate(rows):
        M[a] = [arb(0)] * N
        for b, v in r.items(): M[a][b] = arb(v) if not isinstance(v, arb) else v
    return arb_mat(M)


def mid(x): return arb(x.mid())


def newton_arb(S, z0, digits, iters=60, log=None):
    """Newton on the square system S at `digits` decimal digits; z0 floats. Returns list of arb midpoints."""
    ctx.dps = digits + 30
    z = [arb(repr(float(v))) for v in z0]
    # float warm-up
    zf = np.array(z0, float)
    for _ in range(6):
        zf = zf - np.linalg.solve(dense(S.J(list(zf)), S.N), np.array(S.F(list(zf)), float))
    z = [arb(repr(float(v))) for v in zf]
    tgt = arb(10) ** (-(digits + 5)); last = None
    for it in range(iters):
        Fz = S.F(z); res = max(abs(f.mid()) for f in Fz) if Fz else arb(0)
        if log: log(f'   newton it {it} |F| {float(arb(res)):.2e}' if float(arb(res)) > 0 else f'   newton it {it} |F| < 1e-300')
        if arb(res) < tgt: break
        Jm = to_arb_mat(S.J(z), S.N); b = arb_mat([[f] for f in Fz])
        h = Jm.solve(b)
        z = [mid(z[i] - h[i, 0]) for i in range(S.N)]
    return z


def krawczyk(S, z, rad_digits):
    """K(X) = m - Y F(m) + (I - Y J(X)) (X - m) subset of interior of X => unique zero in X. X = m +- 10^-rad_digits."""
    N = S.N; r = arb(10) ** (-rad_digits)
    Xb = [arb(zi.mid(), r) for zi in z]
    Jm = dense(S.J([float(v) for v in z]), N); Y = np.linalg.inv(Jm)
    # refine Y in arb from the high-precision mid Jacobian
    JmA = to_arb_mat(S.J(z), N); Ya = JmA.inv(); Ya = arb_mat([[mid(Ya[i, j]) for j in range(N)] for i in range(N)])
    Fm = arb_mat([[f] for f in S.F(z)])
    JX = to_arb_mat(S.J(Xb), N)
    I = arb_mat([[arb(1) if i == j else arb(0) for j in range(N)] for i in range(N)])
    dX = arb_mat([[arb(0, r)] for _ in range(N)])
    K = Ya * Fm * (-1) + (I - Ya * JX) * dX
    worst = max(abs(K[i, 0]).upper() for i in range(N))
    ok = all(abs(K[i, 0]) < r for i in range(N))
    return bool(ok), float(arb(worst)) if worst != 0 else 0.0


# ---------------------------------------------------------------- integer relations

LAST = {}


def lll_relation(vals, digits):
    """vals: list of arb (high precision). Find integer c with sum c_i vals_i ~ 0 using `digits` digits. Returns list of ints."""
    m = len(vals); C = fmpz(10) ** digits
    rows = []
    for i, v in enumerate(vals):
        row = [0] * m; row[i] = 1
        x = (v * arb(C)).mid()
        row.append(int(arb(x).floor().unique_fmpz()))
        rows.append(row)
    M = fmpz_mat(rows).lll()
    b1 = sum(int(M[0, j]) ** 2 for j in range(m + 1))
    LAST['b1_log10'] = 0.5 * math.log10(b1) if b1 else 0.0; LAST['dim'] = m; LAST['lll_digits'] = digits
    LAST['no_relation_below_log10'] = LAST['b1_log10'] - (m - 1) / 2 * math.log10(2)
    best = None
    for i in range(m):
        c = [int(M[i, j]) for j in range(m)]
        if any(c): best = c; break
    return best


def eval_rel(c, vals):
    s = arb(0)
    for ci, v in zip(c, vals):
        if ci: s += ci * v
    return s


def powers(x, D):
    p = [arb(1)]
    for _ in range(D): p.append(p[-1] * x)
    return p


def find_minpoly(x, digits_full, maxdeg=24, frac=0.6, log=None, degs=None):
    """minimal polynomial guess of x (arb with ~digits_full correct digits). LLL uses frac*digits_full digits;
    accept when |p(x)| < 10^-(digits_full - 15) * height (relation holds to all digits, not only to the LLL digits)."""
    dl = int(frac * digits_full)
    for D in (degs or range(1, maxdeg + 1)):
        pw = powers(x, D); c = lll_relation(pw, dl)
        if log: log(f'   LLL degree {D} at {dl} digits')
        if c is None: continue
        val = abs(eval_rel(c, pw)); H = max(abs(v) for v in c)
        okfull = val < arb(10) ** (-(digits_full - 15)) * H
        if okfull:
            p = fmpz_poly(c)
            # minimal factor vanishing at x
            fac = p.factor()[1]; bestf = None; bestv = None
            for f, e in fac:
                fv = abs(eval_rel([int(a) for a in f.coeffs()], powers(x, f.degree())))
                if bestv is None or fv < bestv: bestf, bestv = f, fv
            if bestf.leading_coefficient() < 0: bestf = -bestf
            return dict(poly=bestf, degree=bestf.degree(), lll_digits=dl, height_digits=len(str(max(abs(int(a)) for a in bestf.coeffs()))))
    return None


def express_in_basis(x, basis, digits_full, frac=0.6):
    """find rational q_j with x = sum q_j basis_j (basis[0]=1). Returns list of Fractions or None."""
    dl = int(frac * digits_full)
    vals = [x] + basis; c = lll_relation(vals, dl)
    if c is None or c[0] == 0: return None
    if not abs(eval_rel(c, vals)) < arb(10) ** (-(digits_full - 15)) * max(abs(v) for v in c): return None
    return [Fr(-cj, c[0]) for cj in c[1:]]
