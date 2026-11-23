"""Exp 245 (Claude, 2026-11-21): exact sharper second-order basin for the triangle n = 10 optimum.
Exact rationals throughout; floats never decide anything. Same inputs and trust base as exp 214 basin10 (which it
imports for existence, value checks and the cover geometry); only the basin step (3) is replaced.

LEMMA (sharp basin). Let P* be the exact KKT root of basin10 (1), d = P - P*, D = |d|_inf <= r, and suppose
Phi(P) >= t*. Then every tie slack sigma_i = s_i area_i(P) - t* and every contact-wall slack is >= 0. Area is
quadratic, so sigma = L d + q(d) exactly (L the linear part at P*, q the quadratic part, zero for walls,
|q_i| <= 2 D^2). Let A* be the bordered matrix [L | -1_ties ; e_pin] at P* and Ainv* its inverse; v = flex
(pin column, v_pin = 1), M = the slack columns, w = - (t row) = the stress. The t-entry of the flex is 0 at P*
(multiply the flex equations by the KKT multipliers of the root: sum_i w_i = 1/2 != 0), hence w^T L = 0 and
    X := w . sigma = w . q(d) = Q(d) = d^T H d >= 0,   H = sum_i w_i s_i Hdet_i / 4,
    d = a v + M (sigma - q(d)),   a = d_pin, |a| <= D.
Expanding Q(a v + e), e = M sigma - M q, with sigma >= 0 and X = w . sigma:
 (i)  X <= -c a^2 + Gamma |a| X + Lambda X^2 + 2 gt |a| D^2 + 4 Psi m D^2 X + 4 Omega D^4,
 (ii) D <= V |a| + Bm X + 2 m D^2,         and   0 <= X <= h D^2,
with c = -Q(v), gamma_j = 2 v^T H M_j, Gamma = max |gamma_j| / w_j, G = M^T H M, Lambda = max G_jk^+ / (w_j w_k),
gt = sum_ties |gamma_j|, Psi = max_j |H M_j|_1 / w_j, m = max_r sum_ties |M_rj|, Omega = sum_ties |G_jk|,
Bm = max_j |M_j|_inf / w_j, V = |v|_inf, h = sum |H_rs|.
Put a = alpha D, X = xi D^2. Every right-hand side is nondecreasing in D for fixed (alpha, xi), so if (i), (ii)
have no solution with alpha in [0,1], xi in [0,h] at D = r, they have none for any 0 < D <= r. Hence Phi(P) < t*
for every feasible P with 0 < |P - P*| <= r, and Phi(P*) = t*.
The constants at P* are enclosed from those at the rational centre: |A* - A(c)| <= 6 rho (row sums), so
|Ainv* - Ainv(c)| <= ev := k0^2 6 rho / (1 - 6 rho k0) entrywise (basin10's ev), propagated by exact
midpoint-radius interval arithmetic. The (alpha, xi) infeasibility is shown by an exact adaptive split of xi."""
from fractions import Fraction as F
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "214-basin10"))
import basin10 as B10  # noqa: E402
from basin10 import require  # noqa: E402

GRID = F(1, 10**45)          # midpoints are rounded to this grid; the rounding is added to the radius


def rnd(x):
    return F(round(x / GRID)) * GRID


class I:
    """midpoint-radius interval with exact rational endpoints"""
    __slots__ = ("m", "r")

    def __init__(self, m, r=F(0)):
        mm = rnd(F(m)); self.m = mm; self.r = F(r) + abs(F(m) - mm)

    def __add__(s, o):
        o = o if isinstance(o, I) else I(o); return I(s.m + o.m, s.r + o.r)

    def __sub__(s, o):
        o = o if isinstance(o, I) else I(o); return I(s.m - o.m, s.r + o.r)

    def __mul__(s, o):
        o = o if isinstance(o, I) else I(o)
        return I(s.m * o.m, abs(s.m) * o.r + abs(o.m) * s.r + s.r * o.r)

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(s):
        return I(-s.m, s.r)

    def hi(s):
        return s.m + s.r

    def lo(s):
        return s.m - s.r

    def absmax(s):
        return abs(s.m) + s.r


def isum(xs):
    m, r = F(0), F(0)
    for x in xs:
        m += x.m; r += x.r
    return I(m, r)


def constants(points, ties, walls, signs, rho, pin=None):
    rows = [[g / 2 for g in B10.tie_grad(points, tr, s)] + [F(-1)] for tr, s in zip(ties, signs)]
    for p, wl in walls:
        row = [F(0)] * 21; (a, b), _ = B10.WALL[wl]; row[2 * p], row[2 * p + 1] = F(a), F(b); rows.append(row)
    for pin_try in ([pin] if pin is not None else range(20)):
        try:
            Ainv = B10.inverse(rows + [[F(i == pin_try) for i in range(21)]])
            pin = pin_try
            break
        except Exception:
            continue
    else:
        require(False, "no regular coordinate pin")
    k0 = B10.norm(Ainv)
    beta = k0 * 6 * rho
    require(beta < 1, "perturbation too large")
    ev = k0 * k0 * 6 * rho / (1 - beta)
    A = [[I(x, ev) for x in row] for row in Ainv]
    w = [-A[20][j] for j in range(20)]
    v = [A[r][20] for r in range(20)]
    M = [[A[r][j] for j in range(20)] for r in range(20)]
    w_lo = [x.lo() for x in w]
    require(min(w_lo) > 0, "stress not robustly positive")
    H = [[I(0) for _ in range(20)] for _ in range(20)]
    for wi, tr, s in zip(w, ties, signs):
        for (r, c), val in B10.det_hessian(tr, s).items():
            H[r][c] = H[r][c] + wi * F(val, 4)
    Hv = [isum(H[r][c] * v[c] for c in range(20)) for r in range(20)]
    HM = [[isum(H[r][c] * M[c][j] for c in range(20)) for j in range(20)] for r in range(20)]
    c_flex = -isum(v[r] * Hv[r] for r in range(20))
    gamma = [2 * isum(Hv[r] * M[r][j] for r in range(20)) for j in range(20)]
    G = [[isum(M[r][j] * HM[r][k] for r in range(20)) for k in range(20)] for j in range(20)]
    K = dict(
        pin=pin, k0=k0, ev=ev,
        c=c_flex.lo(),
        V=max(x.absmax() for x in v),
        Gamma=max(g.absmax() / wl for g, wl in zip(gamma, w_lo)),
        Lambda=max(max(G[j][k].hi(), F(0)) / (w_lo[j] * w_lo[k]) for j in range(20) for k in range(20)),
        gt=sum(g.absmax() for g in gamma[:14]),
        Psi=max(sum(HM[r][j].absmax() for r in range(20)) / w_lo[j] for j in range(20)),
        m=max(sum(M[r][j].absmax() for j in range(14)) for r in range(20)),
        Omega=sum(G[j][k].absmax() for j in range(14) for k in range(14)),
        Bm=max(max(M[r][j].absmax() for r in range(20)) / w_lo[j] for j in range(20)),
        h=sum(x.absmax() for row in H for x in row),
        pin_row_exact=all(Ainv[pin][j] == (j == 20) for j in range(21)))
    require(K["pin_row_exact"], "pin row is not a unit row")
    require(K["c"] > 0, "no negative curvature along the flex")
    return K


def cell_max(K, D, x0, x1):
    """upper bound of F(alpha, xi) = RHS(i) - xi over xi in [x0, x1], alpha in [alpha_lo, 1] (alpha_lo from (ii)),
    or None if (ii) leaves no alpha. F <= -c alpha^2 + beta1 alpha + const (concave in alpha)."""
    c, V, Ga, La, gt, Ps, m, Om, Bm = (K[k] for k in ("c", "V", "Gamma", "Lambda", "gt", "Psi", "m", "Omega", "Bm"))
    a_lo = max(F(0), (1 - Bm * x1 * D - 2 * m * D) / V)
    if a_lo > 1:
        return None
    b1 = Ga * D * x1 + 2 * gt * D
    const = La * D * D * x1 * x1 + 4 * Ps * m * D * D * x1 + 4 * Om * D * D - x0
    vertex = b1 / (2 * c)
    a = min(max(vertex, a_lo), F(1))
    return -c * a * a + b1 * a + const


def infeasible(K, D, max_cells=200000):
    """exact: no (alpha, xi) in [0,1] x [0,h] satisfies (i) and (ii) at this D. Adaptive bisection of xi."""
    todo, cells = [(F(0), K["h"])], 0
    while todo:
        x0, x1 = todo.pop()
        cells += 1
        if cells > max_cells:
            return False, cells
        f = cell_max(K, D, x0, x1)
        if f is None or f < 0:
            continue
        if x1 - x0 < F(1, 10**30):
            return False, cells
        mid = (x0 + x1) / 2
        todo += [(x0, mid), (mid, x1)]
    return True, cells


def verify_sharp(points, ties, walls, t0, cover_radius, pin=12):
    """basin10 steps (1), (2) and feasibility, then the sharp basin at r = cover_radius + rho. Raises on failure."""
    require(len(points) == B10.N and len(ties) == 14 and len(walls) == 6, "unexpected active set")
    require(all(len(t) == 3 and all(type(i) is int for i in t) and 0 <= t[0] < t[1] < t[2] < B10.N for t in ties)
            and len(set(ties)) == 14, "ties must be 14 distinct sorted index triples")
    require(all(0 <= p < B10.N and w in B10.WALL for p, w in walls) and len(set(walls)) == 6,
            "walls must be 6 distinct contacts")
    for p, w in walls:
        (a, b), off = B10.WALL[w]
        require(a * points[p][0] + b * points[p][1] == off, "centre not exactly on its wall")
    ex = B10.existence(points, ties, walls, t0)
    contacts = set(walls)
    for p, (x, y) in enumerate(points):
        for w, ((a, b), off) in B10.WALL.items():
            if (p, w) not in contacts:
                require(a * x + b * y - off > 2 * ex["rho"], "a point may leave the triangle at the exact root")
    B10.value_checks(points, ties, ex["signs"], ex["rho"], ex["t_upper"])
    rho = ex["rho"]
    r = F(cover_radius) + rho
    # tie orientations cannot change within r of P* (as basin10): |det change| <= 4 D + 4 D^2 in doubled area
    require(4 * (r + rho) + 4 * (r + rho) ** 2 < min(abs(B10.area(points, t)) for t in ties), "tie orientation may change")
    K = constants(points, ties, walls, ex["signs"], rho, pin)
    ok, cells = infeasible(K, r)
    require(ok, "sharp basin inequalities have a solution")
    return dict(rho=rho, r=r, cells=cells, t_lower=ex["t_lower"], t_upper=ex["t_upper"], **K)


if __name__ == "__main__":
    pts, ties, walls, t0 = B10.load()
    for R in (F(7, 10**4), F(72, 10**5), F(73, 10**5), F(8, 10**4)):
        try:
            out = verify_sharp(pts, ties, walls, t0, R)
            print("radius", float(R), "OK", {k: (float(v) if isinstance(v, F) else v) for k, v in out.items()})
        except Exception as e:
            print("radius", float(R), "FAIL", e)
