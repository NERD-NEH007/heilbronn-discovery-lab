"""Rigorous certificate that a Heilbronn configuration is (modulo rattlers) a strict local maximum of the minimum
triangle area in the unit square.  Exact rational / integer arithmetic only in the proof; floats only to choose the
approximate point and the preconditioners.  See NOTES.md for the mathematics.

    python code/exp44/certify_interval.py --sizes 40-40  -> output/exp44/certs.jsonl (one line per n)
    (n <= 36 needs the published record in records/n{n}.txt; see records/README.md)

Steps per n
  1. float stage: ties (rel 1e-6), wall contacts (1e-8), KKT stress LPs -> loaded constraints, rattlers, core.
  2. symmetry of the core (8 square maps, tol 1e-7) -> exact affine orbit parametrisation p_j = a_j + B_j u
     (stabiliser fixed lines and loaded wall contacts are built in exactly).
  3. reduced tie system G(u,t) = (sigma_T A_T(p(u)) - t) over tie orbits; dependent orbits dropped (pivoted QR).
     rank = #unknowns -> 'first-order' mode: Newton on G; rank = #unknowns - 1 -> 'second-order' mode: Newton on the
     square KKT system (G = 0, J^T lam = -e_t).  Polish to ~1e-31 with exact residuals on a 2^-110 grid.
  4. existence + uniqueness of an exact zero in the sup-ball of radius r around the polished point (contraction /
     Krawczyk-type radii inequality  Y + Z0 r + Z2 r^2/2 <= r,  Z0 + Z2 r < 1, all computed exactly).
  5. exact ties beyond the solved equations: closure under the symmetry group, the 4-point identity and exact
     polynomial identities on the parametrised variety; every other triangle (all C(n,3), rattlers included) is
     proved > t over the ball; free coordinates proved inside (0,1).
  6. first-order mode: vertex stress of 2*n_core+1 exact constraints, M nonsingular, multipliers
     y = -(M^-1)[last,:] > 0, proven basin radius c/4 with c = min(y)/||M^-1||_inf.
     second-order mode: reduced multipliers > 0, wall multipliers > 0, and the Lagrangian Hessian is negative on the
     (1-dim) null space of the loaded rows (SOSC).  No basin radius in this mode.
"""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import json, math, time, itertools, argparse
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import qr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import hlib
import hlib as elites

KGRID = 110                                   # polished coordinates live on the grid 2^-KGRID
HALF = Fr(1, 2)
EDGES = ((0, 0, +1), (1, 0, +1), (0, 1, -1), (1, 1, -1))   # (axis, value, inward sign): x=0, y=0, x=1, y=1
MAPS = [((1, 0), (0, 1)), ((0, -1), (1, 0)), ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
        ((-1, 0), (0, 1)), ((1, 0), (0, -1)), ((0, 1), (1, 0)), ((0, -1), (-1, 0))]
MAP_NAMES = ['id', 'rot90', 'rot180', 'rot270', 'mirror_x', 'mirror_y', 'diag', 'antidiag']

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
    """Relative interior of the KKT stress polytope (as exp 40 kkt.py): returns y (lam for ties, mu for walls)."""
    n = len(P); cols = [s * tri_grad(P, T, n) for T, s in zip(ties, sig)]
    for (i, e) in walls:
        g = np.zeros(2 * n); g[2 * i + EDGES[e][0]] = EDGES[e][2]; cols.append(g)
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
    walls = [(int(i), e) for e, (ax, val, sg) in enumerate(EDGES) for i in np.where(np.abs(P[:, ax] - val) < wall_tol)[0]]
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
    for M, name in zip(MAPS, MAP_NAMES):
        Mm = np.array(M, float); Q = 0.5 + (Pc - 0.5) @ Mm.T
        D = np.hypot(Q[:, None, 0] - Pc[None, :, 0], Q[:, None, 1] - Pc[None, :, 1]); nn = D.argmin(1)
        if D.min(1).max() < tol and len(set(nn.tolist())) == len(Pc): G.append((M, nn.tolist(), name))
    return G

def mapply(M, p):                       # c + M (p - c), exact
    q = (p[0] - HALF, p[1] - HALF)
    return (HALF + M[0][0] * q[0] + M[0][1] * q[1], HALF + M[1][0] * q[0] + M[1][1] * q[1])

def mlin(M, w): return (M[0][0] * w[0] + M[0][1] * w[1], M[1][0] * w[0] + M[1][1] * w[1])

def build_param(Pc, G, wallset):
    """Exact affine parametrisation of symmetric core configurations: par[j] = (a_j, [(l, b_jl)]).  wallset: set of
    (core index, axis, value) loaded contacts.  Returns par, u0 (float start), fixed coordinate mask."""
    nc = len(Pc); par = [None] * nc; u0 = []
    for i in range(nc):
        if par[i] is not None: continue
        orbit = {}
        for (M, perm, _) in G: orbit.setdefault(perm[i], M)
        rows = []
        for (M, perm, _) in G:
            if perm[i] != i: continue
            for r in range(2):
                k = (M[r][0] - (r == 0), M[r][1] - (r == 1))
                if k != (0, 0): rows.append((Fr(k[0]), Fr(k[1]), (k[0] + k[1]) * HALF))
        for j, M in orbit.items():
            for (jj, ax, val) in wallset:
                if jj == j:
                    k = M[ax]; rows.append((Fr(k[0]), Fr(k[1]), Fr(val) - HALF + (k[0] + k[1]) * HALF))
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
        for j, M in orbit.items():
            par[j] = (mapply(M, a), [(l, mlin(M, w)) for l, w in zip(idx, dirs)])
    # consistency: p_{perm_g(j)} == c + M_g (p_j - c) as affine maps
    for (M, perm, _) in G:
        for j in range(nc):
            a, B = par[j]; a2, B2 = par[perm[j]]
            if mapply(M, a) != a2 or sorted((l, mlin(M, b)) for l, b in B) != sorted(B2):
                raise ValueError('orbit parametrisation inconsistent')
    return par, u0

def positions(par, u):
    out = []
    for a, B in par:
        x, y = a
        for l, b in B: x += u[l] * b[0]; y += u[l] * b[1]
        out.append((x, y))
    return out

def area2(p, T):                    # 2 * signed area, exact or float
    a, b, c = p[T[0]], p[T[1]], p[T[2]]
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

def pgrads(p, T):                   # d A / d p for the three points (A = signed area)
    i, j, k = T; pi, pj, pk = p[i], p[j], p[k]
    return {i: ((pj[1] - pk[1]) / 2, (pk[0] - pj[0]) / 2), j: ((pk[1] - pi[1]) / 2, (pi[0] - pk[0]) / 2),
            k: ((pi[1] - pj[1]) / 2, (pj[0] - pi[0]) / 2)}

def jac_rows(par, p, eqs, du):
    """Rows of DG (dict col -> value), col du = t."""
    rows = []
    for T, s in eqs:
        row = {du: -1}
        for j, g in pgrads(p, T).items():
            for l, b in par[j][1]: row[l] = row.get(l, 0) + s * (g[0] * b[0] + g[1] * b[1])
        rows.append(row)
    return rows

def hess_entries(par, T):
    """The (constant) Hessian of the signed area A_T(p(u)) in u, as dict (l, m) -> Fraction.
    A = 1/2 cross(D1 u, D2 u) + affine, D1 = B_j - B_i, D2 = B_k - B_i  =>  H_lm = 1/2 [cross(D1_l, D2_m) + cross(D1_m, D2_l)]."""
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

def grid(x): return Fr(round(Fr(x) * 2 ** KGRID), 2 ** KGRID)

# ----------------------------------------------------------------------------------------------- exact integer linear algebra
def to_int(Fmat):
    """Matrix (list of lists or list) of Fractions -> (numpy object int array, common denominator)."""
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
    """|| I - P/scale ||_inf exactly, P integer object matrix, scale int or Fraction."""
    best = Fr(0); d = Pint.shape[0]
    for i in range(d):
        s = sum(abs(scale * (1 if i == j else 0) - Pint[i, j]) for j in range(d))
        best = max(best, Fr(s) / scale)
    return best

# ----------------------------------------------------------------------------------------------- the certificate
def contraction(Fv, DFrows, D, L):
    """Existence + uniqueness of a zero of the quadratic map F near the point where F = Fv (exact), DF = DFrows (exact),
    with row Lipschitz constants L (||DF_i(w) - DF_i(w~)||_1 <= L_i |w - w~|_inf).  Newton-type operator
    T(w) = w - R F(w), R ~ DF^-1 (float, rounded to dyadics).  Returns dict with r (sup-norm radius) or None."""
    Jx = dense(DFrows, D, conv=Fr); Jf = np.array([[float(v) for v in row] for row in Jx]); Rf = np.linalg.inv(Jf)
    Rint, sR = float_to_int(Rf); Jint, sJ = to_int(Jx); Gint, sG = to_int(Fv); Lint, sL = to_int(L)
    Z0 = inf_norm_I_minus(Rint.dot(Jint), sR * sJ)
    Y = Fr(max(abs(v) for v in Rint.dot(Gint))) / (sR * sG)
    Z2 = Fr(max(abs(v) for v in np.abs(Rint).dot(Lint))) / (sR * sL)
    o = dict(Y=float(Y), Z0=float(Z0), Z2=float(Z2), cond=float(np.linalg.cond(Jf)), r=None)
    if Z0 >= 1: return o
    r = max(2 * Y / (1 - Z0), Fr(1, 10 ** 40))
    for _ in range(80):
        if Y + Z0 * r + Z2 * r * r / 2 <= r and Z0 + Z2 * r < 1: o['r'] = r; return o
        r *= 4
    return o

def poly_of(par, T):
    """Signed area A_T(p(u)) as an exact quadratic polynomial in u: dict monomial -> Fraction, monomials (), (l,), (l, m) l<=m."""
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
    """Exact certificate that polynomial `target` = sum_k c_k basis[k] (basis: list of (poly, value-in-units-of-t)).
    Finds c by float least squares on the basis polys sharing a monomial with the target (closure), rounds c to small
    rationals and verifies the identity exactly.  Returns the implied value sum c_k v_k (exact) or None."""
    for P, v in basis:                                            # fast path: +- equality
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
    """Triangles (sorted local triples) whose |area| equals t exactly at the exact point, given that the seed ties do:
    closure under the symmetry group and under the 4-point identity [bcd] - [acd] + [abd] - [abc] = 0 (signed areas).
    Returns dict triple -> sign (signed area = sign * t)."""
    # val[T] = integer k with signed area A_T = k * t exactly (|k| = 1: a tie; other k propagate through identities)
    sg = lambda T: 1 if area2(p, T) > 0 else -1
    val = {T: sg(T) for T in seed}; changed = True
    tv = abs(area2(p, next(iter(seed)))) / 2
    while changed:
        changed = False
        for T in list(val):
            for (M, perm, _) in G:
                U = tuple(sorted(perm[v] for v in T))
                if U not in val:                           # isometry: |A_U| = |A_T|, sign read off the point
                    val[U] = abs(val[T]) * sg(U); changed = True
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
                    if s == 0 or abs(s) > 12: continue          # degenerate or uninteresting
                    if abs(abs(area2(p, tri[i])) / 2 - abs(s) * tv) > tv / 1000:
                        raise AssertionError('implied area value inconsistent with the point: bug')
                    val[tri[i]] = s; changed = True
        if not changed and par is not None:                   # polynomial identities on the parametrised variety
            basis = [(poly_of(par, T), v) for T, v in val.items()]
            for T in cands:
                if T in val: continue
                imp = poly_identity(poly_of(par, T), basis)
                if imp is not None and imp != 0 and imp.denominator == 1 and abs(imp) <= 12:
                    if abs(abs(area2(p, T)) / 2 - abs(imp) * tv) > tv / 1000:
                        raise AssertionError('implied area value inconsistent with the point: bug')
                    val[T] = int(imp); changed = True
    return {T: v for T, v in val.items() if abs(v) == 1}

def param_walls(par):
    """Core coordinates fixed by the parametrisation at 0 or 1: (j, axis, inward sign)."""
    out = []
    for j, (a, B) in enumerate(par):
        for c in (0, 1):
            if all(b[c] == 0 for l, b in B) and a[c] in (0, 1): out.append((j, c, 1 if a[c] == 0 else -1))
    return out

def precond_inverse(Mx, rho, nnz_change):
    """For an exact matrix Mx (list of Fraction rows) whose entries move by <= rho on at most nnz_change entries per row
    over the box: C ~ Mx^-1 and beta >= ||I - C M|| for every M in the box.  Returns (Cint, sC, Cnorm, beta)."""
    Mf = np.array([[float(v) for v in row] for row in Mx]); Cf = np.linalg.inv(Mf)
    Cint, sC = float_to_int(Cf); Mint, sM = to_int(Mx); N = len(Mx)
    beta0 = inf_norm_I_minus(Cint.dot(Mint), sC * sM)
    Cnorm = Fr(max(sum(abs(v) for v in Cint[i]) for i in range(N))) / sC
    return Cint, sC, Cnorm, beta0 + Cnorm * nnz_change * rho, float(np.linalg.cond(Mf))

def tie_row(p, Tl, nc):
    s = 1 if area2(p, Tl) > 0 else -1; row = {2 * nc: Fr(-1)}
    for j, g in pgrads(p, Tl).items(): row[2 * j] = s * g[0]; row[2 * j + 1] = s * g[1]
    return row

def dec(x, digits=32, up=False):
    v = x.numerator * 10 ** digits
    q = -((-v) // x.denominator) if up else v // x.denominator
    return f'{q}e-{digits}'

def certify(n, e, log=print):
    t0 = time.time(); P = np.array(e['points'], float)
    out = dict(n=n, family=e['family'], source=e['source'], input_value=e['value'])
    fs = float_stage(P)
    if not fs['ok']: out.update(certified=False, reason=fs['reason']); return out
    core = fs['core']; ci = {g: k for k, g in enumerate(core)}; nc = len(core); Pc = P[core]
    lties = [(tuple(ci[v] for v in fs['ties'][k]), fs['sig'][k]) for k in fs['loaded_ties']]
    lwalls = [(ci[fs['walls'][k][0]], fs['walls'][k][1]) for k in fs['loaded_walls']]
    out.update(n_core=nc, rattlers=fs['rattlers'], active_ties=len(fs['ties']), loaded_ties=len(lties),
               loaded_walls=len(lwalls), wall_contacts=len(fs['walls']), tie_spread_input=fs['tie_spread'],
               next_gap_input=fs['next_gap'], dim_full=2 * nc + 1)
    unl = [(ci[i], ed) for (i, ed) in fs['walls'] if i in ci and (ci[i], ed) not in lwalls]
    G = detect_sym(Pc); out['symmetry'] = [g[2] for g in G]
    wallset = {(j, EDGES[ed][0], EDGES[ed][1]) for j, ed in lwalls + unl}
    par, u0 = build_param(Pc, G, wallset)
    du = len(u0); d = du + 1
    seen, eqs = set(), []
    for T, s in lties:
        key = tuple(sorted(T))
        if key in seen: continue
        seen |= {tuple(sorted(perm[v] for v in key)) for (M, perm, _) in G}
        eqs.append((key, 1 if area2(Pc, key) > 0 else -1))
    out.update(reduced_unknowns=d, tie_orbits=len(eqs), unloaded_wall_contacts=len(unl))
    # independent equations (dependencies, e.g. from the 4-point identity, are recovered exactly later)
    u = [grid(x) for x in u0]; tv = grid(np.mean([abs(area2(Pc, T)) / 2 for T, s in eqs]))
    pf = positions(par, [float(x) for x in u]); J = dense(jac_rows(par, pf, eqs, du), d)
    sv = np.linalg.svd(J, compute_uv=False); rank = int((sv > 1e-8 * sv[0]).sum())
    if rank < len(eqs):
        _, _, piv = qr(J.T, pivoting=True); keep = sorted(piv[:rank].tolist()); eqs = [eqs[k] for k in keep]
    out.update(reduced_rank=rank, dropped_tie_orbits=out['tie_orbits'] - len(eqs))
    m = len(eqs); k_left = d - rank
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
        DF = [dict(r_) for r_ in rows]                         # [J, 0]
        Hl = [dict() for _ in range(d)]
        for k_, (H, (T, s)) in enumerate(zip(Hs, eqs)):
            for (l, mm), v in H.items(): Hl[l][mm] = Hl[l].get(mm, 0) + lam[k_] * s * (v if exact else float(v))
        for c in range(d):                                     # [H_lam, J^T]
            row = dict(Hl[c])
            for k_, r_ in enumerate(rows):
                if c in r_: row[d + k_] = r_[c]
            DF.append(row)
        return p, Gv + st, DF
    # ---------------- polish (Newton, exact residuals, 2^-110 grid)
    lam = None
    if mode == 'second-order':
        Jk =dense(jac_rows(par, pf, eqs, du), d); lam0 = np.linalg.lstsq(Jk.T, -np.eye(d)[du], rcond=None)[0]
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
    p, Fv, DFrows = kkt_F(u, tv, lam)
    if max(abs(g) for g in Fv) > Fr(1, 10 ** 20):
        out.update(certified=False, reason=f'Newton polish did not converge (residual {float(max(abs(g) for g in Fv)):.1e})'); return out
    out['t_polished'] = dec(tv)
    # ---------------- existence and uniqueness of the exact point
    L = [hess_l1(par, T) for T, s in eqs]
    if mode == 'second-order':
        Ls = [Fr(0)] * d
        for H in Hs:
            for (l, mm), v in H.items(): Ls[l] += 2 * abs(v)
        L = L + Ls
    C = contraction(Fv, DFrows, len(Fv), L)
    out.update(Y=C['Y'], Z0=C['Z0'], Z2=C['Z2'], cond_reduced=C['cond'])
    if C['r'] is None: out.update(certified=False, reason='contraction inequalities fail (Z0 >= 1 or no radius)'); return out
    r = C['r']
    bmax = max(sum(abs(b[c]) for l, b in B) for a, B in par for c in (0, 1))
    rho = bmax * r
    out.update(existence_radius=float(r), box_rho=float(rho))
    assert rho < Fr(1, 10 ** 9)
    # ---------------- (a) every triangle not exactly tied stays strictly above t; coordinates inside
    Tc = np.array(list(itertools.combinations(range(nc), 3))); pcf = np.array([[float(x), float(y)] for x, y in p])
    Ac = np.abs(sareas(pcf, Tc)); cands = [tuple(int(v) for v in Tc[k_]) for k_ in np.where(np.abs(Ac / float(tv) - 1) < 1e-9)[0]]
    sign = close_exact({T for T, s in eqs}, p, G, par, cands)
    exact_set = {tuple(sorted(core[v] for v in T)) for T in sign}
    out['exact_ties'] = len(exact_set)
    Pfull = [None] * n
    for k_, g in enumerate(core): Pfull[g] = p[k_]
    for g in fs['rattlers']: Pfull[g] = (Fr(P[g, 0]), Fr(P[g, 1]))
    tup = tv + r; slack = 4 * rho + 4 * rho * rho
    Tall = np.array(list(itertools.combinations(range(n), 3)))
    def check_others(Pfull):
        Pa = np.array([[float(x), float(y)] for x, y in Pfull]); A = np.abs(sareas(Pa, Tall))
        near = np.where(A - float(tv) < 1e-6)[0]; worst = None; bad = []
        for k_ in near:
            tri = tuple(int(v) for v in Tall[k_])
            if tri in exact_set: continue
            mg = abs(area2(Pfull, tri)) / 2 - slack - tup
            if worst is None or mg < worst: worst = mg
            if mg <= 0: bad.append(tri)
        return bad, worst
    bad, worst = check_others(Pfull)
    nudged = 0.0
    if bad and fs['rattlers'] and any(any(v in fs['rattlers'] for v in tri) for tri in bad):
        Pa = np.array([[float(x), float(y)] for x, y in Pfull]); rat = fs['rattlers']; orig = Pa[rat].copy()
        withr = np.array([any(v in rat for v in T) for T in Tall])
        for step in (1e-4, 3e-5, 1e-5, 3e-6, 1e-6) * 3:
            A = sareas(Pa, Tall); sel = np.where(withr & (np.abs(A) < float(tv) + 1e-4))[0]; cur = np.abs(A[sel]).min()
            nv = 2 * len(rat) + 1; Aub, bub = [], []
            for k_ in sel:
                g = np.sign(A[k_]) * tri_grad(Pa, Tall[k_], n); row = np.zeros(nv)
                for q, rr in enumerate(rat): row[2 * q:2 * q + 2] = -g[2 * rr:2 * rr + 2]
                row[-1] = 1; Aub.append(row); bub.append(abs(A[k_]) - float(tv))
            bnds = [(max(-step, -Pa[rr, c]), min(step, 1 - Pa[rr, c])) for rr in rat for c in (0, 1)] + [(None, None)]
            cc = np.zeros(nv); cc[-1] = -1
            res = linprog(cc, A_ub=np.array(Aub), b_ub=np.array(bub), bounds=bnds, method='highs')
            if res.status != 0: break
            trial = Pa.copy()
            for q, rr in enumerate(rat): trial[rr] = np.clip(Pa[rr] + res.x[2 * q:2 * q + 2], 0, 1)
            new = np.abs(sareas(trial, Tall)[withr]).min()
            if new > cur: Pa = trial
            if new > float(tv) * (1 + 1e-6): break
        nudged = float(np.abs(Pa[rat] - orig).max())
        for rr in rat: Pfull[rr] = (Fr(Pa[rr, 0]), Fr(Pa[rr, 1]))
        bad, worst = check_others(Pfull)
    out['rattlers_nudged_by'] = nudged
    out['_points'] = dict(points=[[str(x), str(y)] for x, y in Pfull], core=core, rattlers=fs['rattlers'], t=str(tv),
                          exact_ties=sorted([list(T) for T in exact_set]))
    out['min_margin_other_near'] = float(worst) if worst is not None else None    # None: every other triangle > t + 1e-6
    if bad:
        coinc = sum(1 for tri in bad if abs(abs(area2(Pfull, tri)) / 2 - tv) < Fr(1, 10 ** 25))
        out.update(certified=False, reason=f'{len(bad)} triangles not provably above t' + (
                   f'; {coinc} of them equal t to 1e-25 (coincident ties not implied by the solved system, symmetry or '
                   'exact polynomial identities)' if coinc else ' (they lie below t: wrong active set / not a local max)'),
                   bad=[list(b) for b in bad[:12]])
        return out
    fixedc = {(j, c) for j, (a, B) in enumerate(par) for c in (0, 1) if all(b[c] == 0 for l, b in B)}
    for j in range(nc):
        for c in (0, 1):
            if (j, c) in fixedc:
                if not (0 <= p[j][c] <= 1): out.update(certified=False, reason='fixed coordinate outside the square'); return out
            elif not (rho < p[j][c] < 1 - rho):
                out.update(certified=False, reason=f'free coordinate of core point {core[j]} not provably inside [0,1]'); return out
    walls = param_walls(par); N = 2 * nc + 1
    wrows = [{2 * j + c: Fr(sg)} for j, c, sg in walls]
    if mode == 'first-order':
        # ------------ (b) vertex stress on exact constraints, multipliers > 0;  (c) first-order strictness + radius
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
        Cint, sC, Cnorm, beta, condf = precond_inverse(Mx, rho, 6)
        out['cond_full'] = condf
        if beta >= 1: out.update(certified=False, reason=f'stationarity matrix not proven nonsingular (beta={float(beta):.3g})'); return out
        err = Cnorm * beta / (1 - beta)
        ylo = [-Fr(Cint[N - 1, k_]) / sC - err for k_ in range(N)]
        ymin = min(ylo); Minv = Cnorm / (1 - beta)
        nties = sum(1 for k_ in chosen if k_ < len(sign))
        out.update(stress_rows=N, stress_ties=nties, stress_walls=N - nties, y_min_lower=float(ymin), Minv_norm_upper=float(Minv))
        if ymin <= 0: out.update(certified=False, reason=f'multiplier not proven positive (lower bound {float(ymin):.3g})'); return out
        radius = min(ymin / Minv / 4, tv / 10)
        out.update(certified=True, basin_radius=float(radius),
                   reason='proved (first order): unique exact tie solution in box, all other triangles > t, '
                          f'{N} independent loaded constraints with multipliers > 0')
    else:
        # ------------ second order: KKT point of the (symmetric) reduced problem, lifted to full space
        lam_lo = [x - r for x in lam]
        if min(lam_lo) <= 0: out.update(certified=False, reason=f'reduced multiplier not proven positive ({float(min(lam_lo)):.3g})'); return out
        orb = []
        for T, s in eqs:
            O = {tuple(sorted(perm[v] for v in T)) for (M, perm, _) in G}; orb.append(O)
        yT = {}                                          # full-space tie multipliers y_T = lam_k / |orbit_k|
        for k_, O in enumerate(orb):
            for U in O: yT[U] = (lam[k_], Fr(len(O)))
        # wall multipliers mu_w = -sign_w * sum_T y_T dh_T/dx_w  (centre value exact, error bound over the box)
        mus = []
        for j, c, sgw in walls:
            val, er = Fr(0), Fr(0)
            for U, (lk, no) in yT.items():
                if j not in U: continue
                gs = pgrads(p, U)[j][c] * (1 if area2(p, U) > 0 else -1)
                val += lk / no * gs; er += r / no * (abs(gs) + rho) + (lk + r) / no * rho
            mus.append(-sgw * val - er)
        out['wall_mu_min_lower'] = float(min(mus)) if mus else None
        if mus and min(mus) <= 0: out.update(certified=False, reason='wall multiplier not proven positive'); return out
        # sanity (not part of the proof; full stationarity follows from the reduced one by symmetry, see NOTES):
        # residual of y^T M = (0, ..., 0, -1) at the centre with centre multipliers
        g = [Fr(0)] * N
        for U, (lk, no) in yT.items():
            for c_, v in tie_row(p, U, nc).items(): g[c_] += lk / no * v
        for j, c, sgw in walls:
            muc = -sgw * sum((lk / no * pgrads(p, U)[j][c] * (1 if area2(p, U) > 0 else -1) for U, (lk, no) in yT.items() if j in U), Fr(0))
            g[2 * j + c] += muc * sgw
        g[N - 1] += 1
        out['full_stationarity_residual_centre'] = float(max(abs(v) for v in g))
        if out['full_stationarity_residual_centre'] > 1e-20:
            out.update(certified=False, reason='full-space stationarity check failed (lifting from reduced KKT)'); return out
        # SOSC on the null space of the positively loaded rows
        Mp = [tie_row(p, U, nc) for U in sorted(yT)] + wrows
        Mpf = dense(Mp, N); _, _, piv = qr(Mpf.T, pivoting=True); sel = sorted(piv[:N - 1].tolist())
        if np.linalg.matrix_rank(Mpf[sel]) != N - 1:
            out.update(certified=False, reason='loaded rows do not have rank 2nc'); return out
        w = np.linalg.svd(Mpf[sel])[2][-1]; j0 = int(np.abs(w).argmax())
        cols = [c_ for c_ in range(N) if c_ != j0]
        Mx = dense([Mp[k_] for k_ in sel], N, conv=Fr); MB = [[row[c_] for c_ in cols] for row in Mx]; b = [-row[j0] for row in Mx]
        Cint, sC, Cnorm, beta, condf = precond_inverse(MB, rho, 6)
        out['cond_full'] = condf
        if beta >= 1: out.update(certified=False, reason='null-space system not proven nonsingular'); return out
        zf = np.linalg.solve(np.array([[float(v) for v in row] for row in MB]), np.array([float(v) for v in b]))
        z = [grid(float(v)) for v in zf]
        res0 = max(abs(bi - sum(mij * zj for mij, zj in zip(row, z))) for row, bi in zip(MB, b))
        zn = max(abs(v) for v in z)
        ez = Cnorm * (res0 + 6 * rho * zn + rho) / (1 - beta)          # |z* - z~|_inf
        wv = [Fr(0)] * N
        for c_, v in zip(cols, z): wv[c_] = v
        wv[j0] = Fr(1); W = max(abs(v) for v in wv[:2 * nc])
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
    out.update(strict='strict local max' if not fs['rattlers'] else 'strict in the core coordinates; rattlers free (local max, non-strict)',
               t_lower=dec(t_lo), t_upper=dec(t_hi, up=True), seconds=round(time.time() - t0, 1))
    return out

def configs(n):
    L = elites.load(n)
    if n <= 36:
        for e in L:
            if e['family'] in ('record', 'record-like'): return e
        return None
    return L[0] if L else None

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--sizes', default='21-52'); ap.add_argument('--out', default=os.path.join(hlib.REPO, 'output', 'exp44', 'certs.jsonl'))
    a = ap.parse_args(); lo, hi = map(int, a.sizes.split('-'))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rec = {str(k): dict(min_area=v) for k, v in hlib.record_values().items()}
    with open(a.out, 'a') as f:
        for n in range(lo, hi + 1):
            e = configs(n)
            if e is None: print(f'n={n}: no input'); continue
            t0 = time.time()
            try: o = certify(n, e)
            except Exception as ex: o = dict(n=n, certified=False, reason=f'error: {type(ex).__name__}: {ex}')
            o['seconds'] = round(time.time() - t0, 1)
            if str(n) in rec: o['published_value'] = rec[str(n)]['min_area']
            if 't_polished' in o:
                tp = float(Fr(o['t_polished'].split('e')[0]) / 10 ** 32)
                o['t_vs_input_rel'] = tp / o['input_value'] - 1
                if 'published_value' in o: o['t_vs_published_rel'] = tp / o['published_value'] - 1
            pts = o.pop('_points', None)
            if pts is not None:
                pd = os.path.join(os.path.dirname(a.out), 'polished'); os.makedirs(pd, exist_ok=True)
                pts.update(n=n, certified=o['certified'], note='exact rationals; core = polished centre of the proof box, '
                           'rattlers = (possibly nudged) input positions')
                json.dump(pts, open(os.path.join(pd, f'n{n}.json'), 'w'))
            print(f"  -> n={n}: certified={o['certified']} ({o['reason']}); r={o.get('existence_radius')}, "
                  f"basin={o.get('basin_radius')}, ymin={o.get('y_min_lower')}, {o['seconds']}s", flush=True)
            f.write(json.dumps(o) + '\n'); f.flush()
