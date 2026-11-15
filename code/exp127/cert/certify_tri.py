"""Exp 113 / cert: the exp 44 local-optimality certifier ported from the unit square to the right triangle
Omega = {x >= 0, y >= 0, x + y <= 1} (area 1/2).  t = min |area| (NOT normalised; the site's value is 2t).
Copied from experiments/44-interval-certificate/certify_interval.py; see NOTES.md for every change.
Exact rational / integer arithmetic only in the proof; floats only choose the active set, the preconditioners and
the vertex stress.

    python certify_tri.py            -> local_certs_tri.json (n = 6, 7, 8)

Walls are written as affine functions g_w(p) = a_w . p + b_w >= 0 on Omega:
    L: x >= 0   (a = (1, 0),   b = 0)
    B: y >= 0   (a = (0, 1),   b = 0)
    H: x+y <= 1 (a = (-1, -1), b = 1)
A contact (j, w) means g_w(p_j) = 0.  In the stress matrix a loaded contact is the row [a_w at point j, 0].
"""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import json, math, time, itertools, argparse
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import qr

HERE = os.path.dirname(os.path.abspath(__file__))
E105 = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'configurations', 'triangle'))

KGRID = 110
WALLS = (('L', (1, 0), 0), ('B', (0, 1), 0), ('H', (-1, -1), 1))      # (name, a_w, b_w): g_w = a_w . p + b_w >= 0
# the 6 affine symmetries of Omega = vertex permutations of O=(0,0), X=(1,0), Y=(0,1);  g(p) = M p + c
_V = ((0, 0), (1, 0), (0, 1))
def _vmap(s):
    o, x, y = (_V[k] for k in s)
    return (((x[0] - o[0], y[0] - o[0]), (x[1] - o[1], y[1] - o[1])), o)
SYMS = [(_vmap(s), nm) for s, nm in (((0, 1, 2), 'id'), ((1, 2, 0), 'rot'), ((2, 0, 1), 'rot2'),
                                       ((0, 2, 1), 'mirror_O'), ((2, 1, 0), 'mirror_X'), ((1, 0, 2), 'mirror_Y'))]

# ----------------------------------------------------------------------------------------------- float stage
def sareas(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))

def tri_grad(P, tri, n):
    i, j, l = tri; g = np.zeros(2 * n)
    for (u, v, w) in ((i, j, l), (j, l, i), (l, i, j)):
        g[2 * u] += 0.5 * (P[v, 1] - P[w, 1]); g[2 * u + 1] += 0.5 * (P[w, 0] - P[v, 0])
    return g

def stress_lp(P, ties, sig, walls):
    n = len(P); cols = [s * tri_grad(P, T, n) for T, s in zip(ties, sig)]
    for (i, w) in walls:
        g = np.zeros(2 * n); g[2 * i] = WALLS[w][1][0]; g[2 * i + 1] = WALLS[w][1][1]; cols.append(g)
    m = len(ties); A = np.array(cols).T
    A = np.vstack([A, np.r_[np.ones(m), np.zeros(len(walls))]]); b = np.r_[np.zeros(2 * n), 1.0]
    nv = A.shape[1]; sup = np.zeros(nv, bool); acc = np.zeros(nv); cnt = 0
    for k in range(nv):
        if sup[k]: continue
        c = np.zeros(nv); c[k] = -1
        r = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * nv, method='highs')
        if r.status == 0 and r.x[k] > 1e-10: sup |= r.x > 1e-10; acc += r.x; cnt += 1
    y = acc / cnt if cnt else None
    return y, cnt, (float(np.abs(A @ y - b).max()) if cnt else None)

def float_stage(P, tie_rel=1e-6, wall_tol=1e-8):
    n = len(P); T = np.array(list(itertools.combinations(range(n), 3))); S = sareas(P, T); A = np.abs(S); t = A.min()
    act = np.where(A <= t * (1 + tie_rel))[0]
    rest = A[A > t * (1 + tie_rel)]; next_gap = float(rest.min() / t - 1)
    walls = [(int(i), w) for w, (_, a, b) in enumerate(WALLS)
             for i in np.where(np.abs(P[:, 0] * a[0] + P[:, 1] * a[1] + b) < wall_tol)[0]]
    ties = [tuple(int(v) for v in T[k]) for k in act]; sig = [int(np.sign(S[k])) for k in act]
    y, cnt, res = stress_lp(P, ties, sig, walls)
    if y is None: return dict(ok=False, reason='stress LP infeasible')
    m = len(ties); lt = [k for k in range(m) if y[k] > 1e-10]; lw = [k for k in range(len(walls)) if y[m + k] > 1e-10]
    touched = set()
    for k in lt: touched |= set(ties[k])
    for k in lw: touched.add(walls[k][0])
    return dict(ok=True, t=float(t), ties=ties, sig=sig, walls=walls, y=y, loaded_ties=lt, loaded_walls=lw,
                core=sorted(touched), rattlers=sorted(set(range(n)) - touched), next_gap=next_gap, lp_count=cnt,
                lp_resid=res, tie_spread=float(A[act].max() / t - 1))

# ----------------------------------------------------------------------------------------------- symmetry + parametrisation
def detect_sym(Pc, tol=1e-7):
    G = []
    for (M, c), name in SYMS:
        Q = Pc @ np.array(M, float).T + np.array(c, float)
        D = np.hypot(Q[:, None, 0] - Pc[None, :, 0], Q[:, None, 1] - Pc[None, :, 1]); nn = D.argmin(1)
        if D.min(1).max() < tol and len(set(nn.tolist())) == len(Pc): G.append(((M, c), nn.tolist(), name))
    return G

def mapply(g, p):                       # M p + c, exact
    M, c = g
    return (c[0] + M[0][0] * p[0] + M[0][1] * p[1], c[1] + M[1][0] * p[0] + M[1][1] * p[1])

def mlin(g, w):
    M = g[0]; return (M[0][0] * w[0] + M[0][1] * w[1], M[1][0] * w[0] + M[1][1] * w[1])

def build_param(Pc, G, wallset):
    """Exact affine parametrisation of G-symmetric core configurations with every contact in wallset (set of
    (core index, wall index)) held exactly on its wall.  par[j] = (a_j, [(l, b_jl)])."""
    nc = len(Pc); par = [None] * nc; u0 = []
    for i in range(nc):
        if par[i] is not None: continue
        orbit = {}
        for (g, perm, _) in G: orbit.setdefault(perm[i], g)
        rows = []                                                      # k . p_i = h
        for (g, perm, _) in G:
            if perm[i] != i: continue
            M, c = g
            for r in range(2):
                k = (M[r][0] - (r == 0), M[r][1] - (r == 1))
                if k != (0, 0) or c[r] != 0: rows.append((Fr(k[0]), Fr(k[1]), Fr(-c[r])))
        for j, g in orbit.items():
            M, c = g
            for (jj, w) in wallset:
                if jj == j:                                            # a_w . (M p_i + c) = -b_w
                    a, b = WALLS[w][1], WALLS[w][2]
                    k = (a[0] * M[0][0] + a[1] * M[1][0], a[0] * M[0][1] + a[1] * M[1][1])
                    rows.append((Fr(k[0]), Fr(k[1]), Fr(-b - a[0] * c[0] - a[1] * c[1])))
        for r1 in rows:
            if r1[0] == 0 and r1[1] == 0 and r1[2] != 0: raise ValueError(f'inconsistent constraints at point {i}')
        rows = [r1 for r1 in rows if r1[0] != 0 or r1[1] != 0]
        rank, sol = 0, None
        for r1 in rows:
            if rank == 0: base = r1; rank = 1; continue
            det = base[0] * r1[1] - base[1] * r1[0]
            if det != 0:
                x = (base[2] * r1[1] - base[1] * r1[2]) / det; y = (base[0] * r1[2] - base[2] * r1[0]) / det
                sol = (x, y); rank = 2; break
        if rank == 2:
            for r1 in rows:
                if r1[0] * sol[0] + r1[1] * sol[1] != r1[2]: raise ValueError(f'inconsistent constraints at point {i}')
            a, dirs = sol, []
        elif rank == 1:
            k0, k1, h = base
            for r1 in rows:
                if k0 * r1[1] - k1 * r1[0] != 0 or (r1[2] * (k0 if k0 else k1) != h * (r1[0] if k0 else r1[1])):
                    raise ValueError(f'inconsistent constraints at point {i}')
            a = (h / k0, Fr(0)) if k0 != 0 else (Fr(0), h / k1); w = (-k1, k0); dirs = [w]
        else:
            a, dirs = (Fr(0), Fr(0)), [(Fr(1), Fr(0)), (Fr(0), Fr(1))]
        idx = []
        for w in dirs:
            ww = float(w[0] ** 2 + w[1] ** 2)
            u0.append(float(w[0] * (Pc[i, 0] - float(a[0])) + w[1] * (Pc[i, 1] - float(a[1]))) / ww)
            idx.append(len(u0) - 1)
        for j, g in orbit.items():
            par[j] = (mapply(g, a), [(l, mlin(g, w)) for l, w in zip(idx, dirs)])
    for (g, perm, _) in G:                                             # exact equivariance of the parametrisation
        for j in range(nc):
            a, B = par[j]; a2, B2 = par[perm[j]]
            if mapply(g, a) != a2 or sorted((l, mlin(g, b)) for l, b in B) != sorted(B2):
                raise ValueError('orbit parametrisation inconsistent')
    return par, u0

def positions(par, u):
    out = []
    for a, B in par:
        x, y = a
        for l, b in B: x += u[l] * b[0]; y += u[l] * b[1]
        out.append((x, y))
    return out

def area2(p, T):
    a, b, c = p[T[0]], p[T[1]], p[T[2]]
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

def pgrads(p, T):
    i, j, k = T; pi, pj, pk = p[i], p[j], p[k]
    return {i: ((pj[1] - pk[1]) / 2, (pk[0] - pj[0]) / 2), j: ((pk[1] - pi[1]) / 2, (pi[0] - pk[0]) / 2),
            k: ((pi[1] - pj[1]) / 2, (pj[0] - pi[0]) / 2)}

def jac_rows(par, p, eqs, du):
    rows = []
    for T, s in eqs:
        row = {du: -1}
        for j, g in pgrads(p, T).items():
            for l, b in par[j][1]: row[l] = row.get(l, 0) + s * (g[0] * b[0] + g[1] * b[1])
        rows.append(row)
    return rows

def hess_entries(par, T):
    i, j, k = T; D1, D2 = {}, {}
    for l, b in par[j][1]: D1[l] = (D1.get(l, (0, 0))[0] + b[0], D1.get(l, (0, 0))[1] + b[1])
    for l, b in par[i][1]: D1[l] = (D1.get(l, (0, 0))[0] - b[0], D1.get(l, (0, 0))[1] - b[1])
    for l, b in par[k][1]: D2[l] = (D2.get(l, (0, 0))[0] + b[0], D2.get(l, (0, 0))[1] + b[1])
    for l, b in par[i][1]: D2[l] = (D2.get(l, (0, 0))[0] - b[0], D2.get(l, (0, 0))[1] - b[1])
    ks = set(D1) | set(D2); z = (0, 0); H = {}
    cr = lambda u, v: u[0] * v[1] - u[1] * v[0]
    for l in ks:
        for m in ks:
            v = Fr(cr(D1.get(l, z), D2.get(m, z)) + cr(D1.get(m, z), D2.get(l, z)), 2)
            if v: H[(l, m)] = v
    return H

def hess_l1(par, T): return sum((abs(v) for v in hess_entries(par, T).values()), Fr(0))

def dense(rows, d, conv=float):
    J = np.zeros((len(rows), d)) if conv is float else [[Fr(0)] * d for _ in rows]
    for r, row in enumerate(rows):
        for c, v in row.items(): J[r][c] = conv(v)
    return J

def grid(x, k=KGRID): return Fr(round(Fr(x) * 2 ** k), 2 ** k)

# ----------------------------------------------------------------------------------------------- exact integer linear algebra
def to_int(Fmat):
    flat = [v for row in Fmat for v in row] if isinstance(Fmat[0], list) else list(Fmat)
    D = 1
    for v in flat: D = math.lcm(D, Fr(v).denominator)
    arr = np.array([[int(Fr(v) * D) for v in row] for row in Fmat] if isinstance(Fmat[0], list)
                   else [int(Fr(v) * D) for v in Fmat], dtype=object)
    return arr, D

def float_to_int(R, bits=50):
    e = math.frexp(float(np.abs(R).max()))[1]; k = bits - e
    return np.array(np.rint(np.ldexp(R, k)).astype(np.int64), dtype=object), 2 ** k if k >= 0 else Fr(1, 2 ** -k)

def inf_norm_I_minus(Pint, scale):
    best = Fr(0); d = Pint.shape[0]
    for i in range(d):
        s = sum(abs(scale * (1 if i == j else 0) - Pint[i, j]) for j in range(d))
        best = max(best, Fr(s) / scale)
    return best

# ----------------------------------------------------------------------------------------------- the certificate
def contraction(Fv, DFrows, D, L, r_try=None):
    """Krawczyk / radii-polynomial test (unchanged from exp 44).  r_try: also test this given radius (for a large
    uniqueness ball); returns o['r_big'] = r_try if it passes."""
    Jx = dense(DFrows, D, conv=Fr); Jf = np.array([[float(v) for v in row] for row in Jx]); Rf = np.linalg.inv(Jf)
    Rint, sR = float_to_int(Rf); Jint, sJ = to_int(Jx); Gint, sG = to_int(Fv) if any(Fv) else (np.zeros(D, dtype=object), 1)
    Lint, sL = to_int(L)
    Z0 = inf_norm_I_minus(Rint.dot(Jint), sR * sJ)
    Y = Fr(max(abs(v) for v in Rint.dot(Gint))) / (sR * sG)
    Z2 = Fr(max(abs(v) for v in np.abs(Rint).dot(Lint))) / (sR * sL)
    o = dict(Y=Y, Z0=Z0, Z2=Z2, cond=float(np.linalg.cond(Jf)), r=None, r_big=None)
    if Z0 >= 1: return o
    r = max(2 * Y / (1 - Z0), Fr(1, 10 ** 40))
    for _ in range(80):
        if Y + Z0 * r + Z2 * r * r / 2 <= r and Z0 + Z2 * r < 1: o['r'] = r; break
        r *= 4
    if r_try is not None and Y + Z0 * r_try + Z2 * r_try * r_try / 2 <= r_try and Z0 + Z2 * r_try < 1: o['r_big'] = r_try
    return o

def poly_of(par, T):
    i, j, k = T; cr = lambda u, v: u[0] * v[1] - u[1] * v[0]
    def diff(x, y):
        D = {}
        for l, b in par[x][1]: D[l] = (D.get(l, (0, 0))[0] + b[0], D.get(l, (0, 0))[1] + b[1])
        for l, b in par[y][1]: D[l] = (D.get(l, (0, 0))[0] - b[0], D.get(l, (0, 0))[1] - b[1])
        return (par[x][0][0] - par[y][0][0], par[x][0][1] - par[y][0][1]), D
    al, D1 = diff(j, i); be, D2 = diff(k, i); P = {(): Fr(cr(al, be), 2)}
    def add(key, v):
        if v: P[key] = P.get(key, 0) + Fr(v, 2)
    for l, b in D1.items(): add((l,), cr(b, be))
    for l, b in D2.items(): add((l,), cr(al, b))
    for l, b1 in D1.items():
        for m, b2 in D2.items(): add((min(l, m), max(l, m)), cr(b1, b2))
    return {k_: v for k_, v in P.items() if v}

def poly_identity(target, basis):
    for P, v in basis:
        if P == target: return Fr(v)
        if all(-P.get(k_, 0) == x for k_, x in target.items()) and len(P) == len(target): return -Fr(v)
    mons = set(target); idx = set()
    for _ in range(4):
        new = {k_ for k_, (P, v) in enumerate(basis) if mons & set(P)}
        if new == idx: break
        idx = new
        for k_ in idx: mons |= set(basis[k_][0])
    idx = sorted(idx)
    if not idx: return None
    ml = sorted(mons, key=str); mi = {m_: r for r, m_ in enumerate(ml)}
    A = np.zeros((len(ml), len(idx))); b = np.zeros(len(ml))
    for c_, k_ in enumerate(idx):
        for m_, x in basis[k_][0].items(): A[mi[m_], c_] = float(x)
    for m_, x in target.items(): b[mi[m_]] = float(x)
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    if np.abs(A @ c - b).max() > 1e-9: return None
    cq = [Fr(float(x)).limit_denominator(64) for x in c]
    acc = {}
    for x, k_ in zip(cq, idx):
        if x:
            for m_, y in basis[k_][0].items(): acc[m_] = acc.get(m_, 0) + x * y
    acc = {k_: v for k_, v in acc.items() if v}
    if acc != target: return None
    return sum((x * basis[k_][1] for x, k_ in zip(cq, idx)), Fr(0))

def close_exact(seed, p, G, par=None, cands=()):
    """Unchanged from exp 44.  The symmetries of Omega are affine with det = +-1, so |A_{g(T)}| = |A_T| identically
    (the square maps were isometries; only |det| = 1 is used)."""
    sg = lambda T: 1 if area2(p, T) > 0 else -1
    val = {T: sg(T) for T in seed}; changed = True
    tv = abs(area2(p, next(iter(seed)))) / 2
    while changed:
        changed = False
        for T in list(val):
            for (g, perm, _) in G:
                U = tuple(sorted(perm[v] for v in T))
                if U not in val: val[U] = abs(val[T]) * sg(U); changed = True
        by_pair = {}
        for T in val:
            for pr in itertools.combinations(T, 2): by_pair.setdefault(pr, []).append(T)
        for Ts in by_pair.values():
            for T1, T2 in itertools.combinations(Ts, 2):
                a, b, c, d = sorted(set(T1) | set(T2)); tri = [(b, c, d), (a, c, d), (a, b, d), (a, b, c)]; co = [1, -1, 1, -1]
                kn = [x in val for x in tri]
                if sum(kn) == 4:
                    assert sum(co[k] * val[tri[k]] for k in range(4)) == 0, '4-point identity violated: bug'
                elif sum(kn) == 3:
                    i = kn.index(False); s = -co[i] * sum(co[k] * val[tri[k]] for k in range(4) if k != i)
                    if s == 0 or abs(s) > 12: continue
                    if abs(abs(area2(p, tri[i])) / 2 - abs(s) * tv) > tv / 1000:
                        raise AssertionError('implied area value inconsistent with the point: bug')
                    val[tri[i]] = s; changed = True
        if not changed and par is not None:
            basis = [(poly_of(par, T), v) for T, v in val.items()]
            for T in cands:
                if T in val: continue
                imp = poly_identity(poly_of(par, T), basis)
                if imp is not None and imp != 0 and imp.denominator == 1 and abs(imp) <= 12:
                    if abs(abs(area2(p, T)) / 2 - abs(imp) * tv) > tv / 1000:
                        raise AssertionError('implied area value inconsistent with the point: bug')
                    val[T] = int(imp); changed = True
    return {T: v for T, v in val.items() if abs(v) == 1}

def wall_aff(par, j, w):
    """g_w(p_j(u)) as an exact affine function of u: (constant, {l: coeff})."""
    a, b = WALLS[w][1], WALLS[w][2]; (x0, y0), B = par[j]
    lin = {}
    for l, v in B:
        cf = a[0] * v[0] + a[1] * v[1]
        if cf: lin[l] = lin.get(l, 0) + cf
    return a[0] * x0 + a[1] * y0 + b, {l: v for l, v in lin.items() if v}

def param_walls(par):
    """Contacts held exactly by the parametrisation: g_w(p_j(u)) == 0 identically."""
    return [(j, w) for j in range(len(par)) for w in range(3) if wall_aff(par, j, w) == (0, {})]

def precond_inverse(Mx, rho, nnz_change):
    Mf = np.array([[float(v) for v in row] for row in Mx]); Cf = np.linalg.inv(Mf)
    Cint, sC = float_to_int(Cf); Mint, sM = to_int(Mx); N = len(Mx)
    beta0 = inf_norm_I_minus(Cint.dot(Mint), sC * sM)
    Cnorm = Fr(max(sum(abs(v) for v in Cint[i]) for i in range(N))) / sC
    return Cint, sC, Cnorm, beta0 + Cnorm * nnz_change * rho, float(np.linalg.cond(Mf))

def tie_row(p, Tl, nc):
    s = 1 if area2(p, Tl) > 0 else -1; row = {2 * nc: Fr(-1)}
    for j, g in pgrads(p, Tl).items(): row[2 * j] = s * g[0]; row[2 * j + 1] = s * g[1]
    return row

def wall_row(j, w):
    a = WALLS[w][1]; row = {}
    if a[0]: row[2 * j] = Fr(a[0])
    if a[1]: row[2 * j + 1] = Fr(a[1])
    return row

def dec(x, digits=32, up=False):
    v = x.numerator * 10 ** digits
    q = -((-v) // x.denominator) if up else v // x.denominator
    return f'{q}e-{digits}'

def f_down(x):
    f = float(x)
    while Fr(f) > x: f = math.nextafter(f, -math.inf)
    return f

def f_up(x):
    f = float(x)
    while Fr(f) < x: f = math.nextafter(f, math.inf)
    return f

def try_rational(par, eqs, u, tv, du, maxden=10 ** 6):
    """If the polished point rounds to small rationals that solve the tie system EXACTLY, return them (the centre
    then has Y = 0).  Choice only: the Krawczyk test is still run around whatever centre is used."""
    uq = [Fr(x).limit_denominator(maxden) for x in u]; tq = Fr(tv).limit_denominator(maxden)
    if max([abs(a - b) for a, b in zip(uq, u)] + [abs(tq - tv)]) > Fr(1, 10 ** 25): return None
    p = positions(par, uq)
    if all(s * area2(p, T) / 2 - tq == 0 for T, s in eqs): return uq, tq
    return None

def certify(n, P, log=print, keep_ctx=False):
    t0 = time.time(); P = np.array(P, float)
    out = dict(n=n)
    fs = float_stage(P)
    if not fs['ok']: out.update(certified=False, reason=fs['reason']); return out
    core = fs['core']; ci = {g: k for k, g in enumerate(core)}; nc = len(core); Pc = P[core]
    lties = [(tuple(ci[v] for v in fs['ties'][k]), fs['sig'][k]) for k in fs['loaded_ties']]
    lwalls = [(ci[fs['walls'][k][0]], fs['walls'][k][1]) for k in fs['loaded_walls']]
    out.update(n_core=nc, rattlers=fs['rattlers'], active_ties=len(fs['ties']), loaded_ties=len(lties),
               loaded_walls=len(lwalls), wall_contacts=len(fs['walls']), tie_spread_input=fs['tie_spread'],
               next_gap_input=fs['next_gap'], dim_full=2 * nc + 1)
    unl = [(ci[i], w) for (i, w) in fs['walls'] if i in ci and (ci[i], w) not in lwalls]
    G = detect_sym(Pc); out['symmetry'] = [g[2] for g in G]
    wallset = set(lwalls + unl)
    par, u0 = build_param(Pc, G, wallset)
    du = len(u0); d = du + 1
    seen, eqs = set(), []
    for T, s in lties:
        key = tuple(sorted(T))
        if key in seen: continue
        seen |= {tuple(sorted(perm[v] for v in key)) for (g, perm, _) in G}
        eqs.append((key, 1 if area2(Pc, key) > 0 else -1))
    out.update(reduced_unknowns=d, tie_orbits=len(eqs), unloaded_wall_contacts=len(unl))
    u = [grid(x) for x in u0]; tv = grid(np.mean([abs(area2(Pc, T)) / 2 for T, s in eqs]))
    pf = positions(par, [float(x) for x in u]); J = dense(jac_rows(par, pf, eqs, du), d)
    sv = np.linalg.svd(J, compute_uv=False); rank = int((sv > 1e-8 * sv[0]).sum())
    if rank < len(eqs):
        _, _, piv = qr(J.T, pivoting=True); keep = sorted(piv[:rank].tolist()); eqs = [eqs[k] for k in keep]
    out.update(reduced_rank=rank, dropped_tie_orbits=out['tie_orbits'] - len(eqs))
    k_left = d - rank
    mode = 'first-order' if k_left == 0 else 'second-order' if k_left == 1 else None
    out['mode'] = mode
    log(f"n={n}: core {nc} (rattlers {len(fs['rattlers'])}), loaded {len(lties)} ties + {len(lwalls)} walls vs 2nc+1={2*nc+1}; "
        f"sym {out['symmetry']}; reduced {out['tie_orbits']} orbits, rank {rank}, unknowns {d} -> {mode}")
    if mode is None:
        out.update(certified=False, reason=f'{k_left} leftover directions in the reduced system (only 0 or 1 handled)'); return out
    Hs = [hess_entries(par, T) for T, s in eqs]
    def kkt_F(u, tv, lam, exact=True):
        p = positions(par, u) if exact else positions(par, [float(x) for x in u])
        Gv = [s * area2(p, T) / 2 - tv for T, s in eqs]; rows = jac_rows(par, p, eqs, du)
        if mode == 'first-order': return p, Gv, rows
        st = [Fr(0) if exact else 0.0] * d; st[du] += 1
        for k_, row in enumerate(rows):
            for c, v in row.items(): st[c] += lam[k_] * v
        DF = [dict(r_) for r_ in rows]
        Hl = [dict() for _ in range(d)]
        for k_, (H, (T, s)) in enumerate(zip(Hs, eqs)):
            for (l, mm), v in H.items(): Hl[l][mm] = Hl[l].get(mm, 0) + lam[k_] * s * (v if exact else float(v))
        for c in range(d):
            row = dict(Hl[c])
            for k_, r_ in enumerate(rows):
                if c in r_: row[d + k_] = r_[c]
            DF.append(row)
        return p, Gv + st, DF
    lam = None
    if mode == 'second-order':
        Jk = dense(jac_rows(par, pf, eqs, du), d); lam0 = np.linalg.lstsq(Jk.T, -np.eye(d)[du], rcond=None)[0]
        lam = [grid(float(x)) for x in lam0]
    hist = []
    for it in range(20):
        p, Fv, rows = kkt_F(u, tv, lam)
        res = max(abs(g) for g in Fv); hist.append(float(res))
        if res < Fr(1, 10 ** 31): break
        _, _, rf = kkt_F(u, tv, [float(x) for x in lam] if lam else None, exact=False) if mode == 'second-order' else (None, None, None)
        Jn = dense(rf if mode == 'second-order' else jac_rows(par, [(float(x), float(y)) for x, y in p], eqs, du), len(Fv))
        step = np.linalg.solve(Jn, np.array([float(g) for g in Fv]))
        u = [grid(x - Fr(float(s))) for x, s in zip(u, step[:du])]; tv = grid(tv - Fr(float(step[du])))
        if lam: lam = [grid(x - Fr(float(s))) for x, s in zip(lam, step[d:])]
    out['polish_residuals'] = [f'{h:.1e}' for h in hist]
    out['exact_rational_centre'] = False
    if mode == 'first-order':
        rq = try_rational(par, eqs, u, tv, du)
        if rq is not None: u, tv = rq; out['exact_rational_centre'] = True
    p, Fv, DFrows = kkt_F(u, tv, lam)
    if max(abs(g) for g in Fv) > Fr(1, 10 ** 20):
        out.update(certified=False, reason=f'Newton polish did not converge (residual {float(max(abs(g) for g in Fv)):.1e})'); return out
    out['t_polished'] = dec(tv)
    L = [hess_l1(par, T) for T, s in eqs]
    if mode == 'second-order':
        Ls = [Fr(0)] * d
        for H in Hs:
            for (l, mm), v in H.items(): Ls[l] += 2 * abs(v)
        L = L + Ls
    C = contraction(Fv, DFrows, len(Fv), L, r_try=Fr(1, 10 ** 6))
    out.update(Y=float(C['Y']), Z0=float(C['Z0']), Z2=float(C['Z2']), cond_reduced=C['cond'],
               uniqueness_radius_1e6=C['r_big'] is not None)
    if C['r'] is None: out.update(certified=False, reason='contraction inequalities fail (Z0 >= 1 or no radius)'); return out
    r = C['r']
    if C['Y'] == 0: out['t_exact'] = str(tv)          # F(centre) = 0 exactly: the unique zero IS the centre
    bmax = max(sum(abs(b[c]) for l, b in B) for a, B in par for c in (0, 1))
    rho = bmax * r
    out.update(existence_radius=f_up(r), box_rho=f_up(rho), existence_radius_exact=str(r), box_rho_exact=str(rho))
    assert rho < Fr(1, 10 ** 9)
    # ---------------- (a) every triangle not exactly tied stays strictly above t (ALL triangles checked exactly)
    Tc = np.array(list(itertools.combinations(range(nc), 3))); pcf = np.array([[float(x), float(y)] for x, y in p])
    Ac = np.abs(sareas(pcf, Tc)); cands = [tuple(int(v) for v in Tc[k_]) for k_ in np.where(np.abs(Ac / float(tv) - 1) < 1e-9)[0]]
    sign = close_exact({T for T, s in eqs}, p, G, par, cands)
    exact_set = {tuple(sorted(core[v] for v in T)) for T in sign}
    out['exact_ties'] = len(exact_set)
    if fs['rattlers']:
        out.update(certified=False, reason='rattlers present (rattler handling not ported to the triangle)'); return out
    Pfull = [None] * n
    for k_, g in enumerate(core): Pfull[g] = p[k_]
    tup = tv + r; slack = 4 * rho + 4 * rho * rho
    bad, worst = [], None
    for tri in itertools.combinations(range(n), 3):
        if tri in exact_set: continue
        mg = abs(area2(Pfull, tri)) / 2 - slack - tup
        worst = mg if worst is None or mg < worst else worst
        if mg <= 0: bad.append(tri)
    out['min_margin_other'] = float(worst) if worst is not None else None
    out['_points'] = dict(points=[[str(x), str(y)] for x, y in Pfull], core=core, t=str(tv),
                          exact_ties=sorted([list(T) for T in exact_set]))
    if bad:
        out.update(certified=False, reason=f'{len(bad)} triangles not provably above t', bad=[list(b) for b in bad[:12]]); return out
    # every point stays in Omega over the box: each wall function is either identically 0 on the parametrisation or
    # > ||a_w||_1 rho at the centre (each coordinate moves by <= rho)
    for j in range(nc):
        for w in range(3):
            c0, lin = wall_aff(par, j, w)
            if (c0, lin) == (0, {}): continue
            gv = WALLS[w][1][0] * p[j][0] + WALLS[w][1][1] * p[j][1] + WALLS[w][2]
            if not gv > (abs(WALLS[w][1][0]) + abs(WALLS[w][1][1])) * rho:
                out.update(certified=False, reason=f'core point {core[j]} not provably inside wall {WALLS[w][0]}'); return out
    walls = param_walls(par); N = 2 * nc + 1
    out['exact_wall_contacts'] = [[core[j], WALLS[w][0]] for j, w in walls]
    wrows = [wall_row(j, w) for j, w in walls]
    if mode == 'first-order':
        cand = [tie_row(p, T, nc) for T in sorted(sign)] + wrows
        out['exact_constraints'] = len(cand)
        Mall = dense(cand, N); chosen = None
        if len(cand) == N: chosen = list(range(N))
        elif len(cand) > N:
            rng = np.random.default_rng(0); be = np.zeros(N); be[-1] = -1
            for trial in range(40):
                cc = rng.random(len(cand)) if trial else np.ones(len(cand))
                res = linprog(cc, A_eq=Mall.T, b_eq=be, bounds=[(0, None)] * len(cand), method='highs-ds')
                if res.status != 0: break
                supp = np.where(res.x > 1e-12)[0]
                if len(supp) == N and np.linalg.matrix_rank(Mall[supp]) == N: chosen = supp.tolist(); break
            out['vertex_trials'] = trial + 1
        if chosen is None:
            out.update(certified=False, reason=f'no nondegenerate vertex stress with {N} rows among {len(cand)} exact constraints'); return out
        Mx = dense([cand[k_] for k_ in chosen], N, conv=Fr)
        Cint, sC, Cnorm, beta, condf = precond_inverse(Mx, rho, 6)    # wall rows are constant; tie rows: 6 entries
        out['cond_full'] = condf
        if beta >= 1: out.update(certified=False, reason=f'stationarity matrix not proven nonsingular (beta={float(beta):.3g})'); return out
        err = Cnorm * beta / (1 - beta)
        ylo = [-Fr(Cint[N - 1, k_]) / sC - err for k_ in range(N)]
        ymin = min(ylo); Minv = Cnorm / (1 - beta)
        nties = sum(1 for k_ in chosen if k_ < len(sign))
        out.update(stress_rows=N, stress_ties=nties, stress_walls=N - nties, y_min_lower=f_down(ymin), Minv_norm_upper=f_up(Minv))
        if ymin <= 0: out.update(certified=False, reason=f'multiplier not proven positive (lower bound {float(ymin):.3g})'); return out
        radius = min(ymin / Minv / 4, tv / 10)
        out.update(certified=True, basin_radius=f_down(radius), basin_radius_exact=str(radius),
                   reason='proved (first order): unique exact tie solution in box, all other triangles > t, '
                          f'{N} independent loaded constraints with multipliers > 0')
    else:
        lam_lo = [x - r for x in lam]
        if min(lam_lo) <= 0: out.update(certified=False, reason=f'reduced multiplier not proven positive ({float(min(lam_lo)):.3g})'); return out
        orb = [{tuple(sorted(perm[v] for v in T)) for (g, perm, _) in G} for T, s in eqs]
        yT = {}
        for k_, O in enumerate(orb):
            for U in O: yT[U] = (lam[k_], Fr(len(O)))
        # wall multipliers: at each point j with exact contacts W_j, solve sum_w mu_w a_w = -gsum_j (gsum_j = sum_T
        # y_T sigma_T grad_j A_T); centre value exact, componentwise error e_j bound over the box, propagated by
        # ||A_j^-1||_inf (A_j = matrix of the contact normals at j; 1 or 2 contacts)
        mus = []; musc = {}
        for j in sorted({j for j, w in walls}):
            Wj = [w for jj, w in walls if jj == j]
            gs, er = [Fr(0), Fr(0)], Fr(0)
            for U, (lk, no) in yT.items():
                if j not in U: continue
                gr = pgrads(p, U)[j]; s = 1 if area2(p, U) > 0 else -1
                for c in (0, 1): gs[c] += lk / no * s * gr[c]
                er += r / no * (max(abs(gr[0]), abs(gr[1])) + rho) + (lk + r) / no * rho
            if len(Wj) == 1:
                a = WALLS[Wj[0]][1]; aa = a[0] ** 2 + a[1] ** 2
                mu = -(a[0] * gs[0] + a[1] * gs[1]) / aa; e = Fr(abs(a[0]) + abs(a[1]), aa) * er
                musc[(j, Wj[0])] = mu; mus.append(mu - e)
            else:
                a1, a2 = WALLS[Wj[0]][1], WALLS[Wj[1]][1]; det = Fr(a1[0] * a2[1] - a2[0] * a1[1])
                inv = [[a2[1] / det, -a2[0] / det], [-a1[1] / det, a1[0] / det]]
                ninv = max(abs(inv[0][0]) + abs(inv[0][1]), abs(inv[1][0]) + abs(inv[1][1]))
                for q, w in enumerate(Wj):
                    mu = -(inv[q][0] * gs[0] + inv[q][1] * gs[1]); musc[(j, w)] = mu; mus.append(mu - ninv * er)
        out['wall_mu_min_lower'] = float(min(mus)) if mus else None
        if mus and min(mus) <= 0: out.update(certified=False, reason='wall multiplier not proven positive'); return out
        g = [Fr(0)] * N
        for U, (lk, no) in yT.items():
            for c_, v in tie_row(p, U, nc).items(): g[c_] += lk / no * v
        for (j, w), mu in musc.items():
            for c_, v in wall_row(j, w).items(): g[c_] += mu * v
        g[N - 1] += 1
        out['full_stationarity_residual_centre'] = float(max(abs(v) for v in g))
        if out['full_stationarity_residual_centre'] > 1e-20:
            out.update(certified=False, reason='full-space stationarity check failed (lifting from reduced KKT)'); return out
        Mp = [tie_row(p, U, nc) for U in sorted(yT)] + wrows
        Mpf = dense(Mp, N); _, _, piv = qr(Mpf.T, pivoting=True); sel = sorted(piv[:N - 1].tolist())
        if np.linalg.matrix_rank(Mpf[sel]) != N - 1:
            out.update(certified=False, reason='loaded rows do not have rank 2nc'); return out
        w_ = np.linalg.svd(Mpf[sel])[2][-1]; j0 = int(np.abs(w_).argmax())
        cols = [c_ for c_ in range(N) if c_ != j0]
        Mx = dense([Mp[k_] for k_ in sel], N, conv=Fr); MB = [[row[c_] for c_ in cols] for row in Mx]; b = [-row[j0] for row in Mx]
        Cint, sC, Cnorm, beta, condf = precond_inverse(MB, rho, 6)
        out['cond_full'] = condf
        if beta >= 1: out.update(certified=False, reason='null-space system not proven nonsingular'); return out
        zf = np.linalg.solve(np.array([[float(v) for v in row] for row in MB]), np.array([float(v) for v in b]))
        z = [grid(float(v)) for v in zf]
        res0 = max(abs(bi - sum(mij * zj for mij, zj in zip(row, z))) for row, bi in zip(MB, b))
        zn = max(abs(v) for v in z)
        ez = Cnorm * (res0 + 6 * rho * zn + rho) / (1 - beta)
        wv = [Fr(0)] * N
        for c_, v in zip(cols, z): wv[c_] = v
        wv[j0] = Fr(1)
        q, dq = Fr(0), Fr(0)
        for U, (lk, no) in yT.items():
            i_, j_, k2 = U; s = 1 if area2(p, U) > 0 else -1
            a_ = (wv[2 * j_] - wv[2 * i_], wv[2 * j_ + 1] - wv[2 * i_ + 1]); b_ = (wv[2 * k2] - wv[2 * i_], wv[2 * k2 + 1] - wv[2 * i_ + 1])
            cr = a_[0] * b_[1] - a_[1] * b_[0]; q += lk / no * s * cr
            A_, B_ = max(abs(a_[0]), abs(a_[1])), max(abs(b_[0]), abs(b_[1])); e2 = 2 * ez
            dcr = 2 * (e2 * B_ + A_ * e2 + e2 * e2)
            dq += r / no * (2 * (A_ + e2) * (B_ + e2)) + (lk + r) / no * dcr
        out.update(sosc_q=float(q), sosc_q_err=float(dq), null_err=float(ez))
        if q + dq >= 0: out.update(certified=False, reason=f'second-order condition not proven (q={float(q):.3g}, err {float(dq):.2g})'); return out
        out.update(certified=True, basin_radius=None, y_min_lower=float(min(min(lam_lo[k_] / len(O) for k_, O in enumerate(orb)), min(mus) if mus else 1)),
                   reason='proved (second order): unique exact KKT point in box, all other triangles > t, multipliers > 0, '
                          'Lagrangian Hessian negative on the 1-dim null space of the loaded rows')
    t_lo, t_hi = tv - r, tv + r
    out.update(strict='strict local max', t_lower=dec(t_lo), t_upper=dec(t_hi, up=True), seconds=round(time.time() - t0, 1))
    if keep_ctx: out['_ctx'] = dict(par=par, eqs=eqs, u=u, tv=tv, r=r, core=core, du=du, C=C)
    return out

def load_input(n):
    return json.load(open(os.path.join(E105, f'pool_tri_n{n}.json')))[0]

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--sizes', default='6,7,8')
    ap.add_argument('--out', default=os.path.join(HERE, 'local_certs_tri.json'))
    a = ap.parse_args()
    res = {}
    for n in [int(s) for s in a.sizes.split(',')]:
        e = load_input(n)
        try: o = certify(n, e['points'])
        except Exception as ex: o = dict(n=n, certified=False, reason=f'error: {type(ex).__name__}: {ex}')
        pts = o.pop('_points', None)
        keep = {k: o.get(k) for k in ('certified', 'reason', 'mode', 'existence_radius', 'box_rho', 'basin_radius',
                                      't_lower', 't_upper', 'symmetry', 'n_core', 'rattlers', 'loaded_ties', 'loaded_walls',
                                      'y_min_lower', 't_exact')}
        keep['alt'] = 0
        keep['points_exact'] = pts['points'] if pts else None
        keep['frame'] = 'right triangle {x>=0, y>=0, x+y<=1}, area 1/2; t = min |area|; site normalised value = 2t'
        keep['extra'] = {k: v for k, v in o.items() if k not in keep and k != 'n'}
        keep['extra']['input'] = dict(source='105-shape-universality/results/pool_tri_n%d.json[0]' % n, value_normalised=e['value'])
        if pts: keep['extra']['exact_ties'] = pts['exact_ties']
        res[str(n)] = [keep]
        print(f"  -> n={n}: certified={o.get('certified')} ({o.get('reason')}); mode={o.get('mode')} r={o.get('existence_radius')}, "
              f"basin={o.get('basin_radius')}, ymin={o.get('y_min_lower')}, t in [{o.get('t_lower')}, {o.get('t_upper')}]", flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)
