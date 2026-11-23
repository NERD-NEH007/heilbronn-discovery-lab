"""Exp 246 (Claude, 2026-11-21): a wider exact second-order basin for the triangle n = 10 optimum.
Exact rationals throughout; floats never decide anything. Same inputs and trust base as exp 214 basin10 and
exp 245 basin10s (imported for existence, value checks, feasibility and the interval class); only the basin
inequalities are replaced.

IDEA. The sharp chain of exp 245 reduces everything to two scalars (a, X) and pays the worst weight ratio
max_j (.)/w_j over all 20 slacks. Two ties (8 and 12) carry almost no stress (w = 0.0023, 0.0007) and a third (7)
little (w = 0.0098); they alone produce Lambda = 4.6e5, B = 4818, Gamma = 233. Here the slacks of a small "weak"
set W are kept as explicit variables next to the flex amplitude, the quadratic form Q is used exactly on the
"soft" space spanned by the flex v and the columns M_k (k in W), and only the remaining "strong" slacks are
reduced to one scalar. The (alpha, xi) split becomes an exact branch and bound in (alpha, s_W) with xi eliminated
by concavity.

LEMMA (wide basin). Setting and identities as in basin10s: P* the exact KKT root, d = P - P*, D = |d|_inf <= r,
Phi(P) >= t*. Then sigma >= 0 (14 tie slacks, 6 contact-wall slacks), sigma = L d + kappa with kappa_i = q_i(d)
(ties; 0 for walls), |kappa_i| <= 2 D^2, and with A*^-1 = [M v ; -w^T 0]:
    d = a v + M (sigma - kappa),   a = d_pin, |a| <= D,
    X := w . sigma = w . kappa = Q(d) = d^T H d,   H = sum_i w_i s_i Hdet_i / 4,   w > 0.
Justifications (added 2026-11-22 after the API referee panel, gaps G1-G3):
 - |kappa_i| <= 2 D^2: kappa_i is the signed area of the triangle (d_i, d_j, d_k) of displacements, whose vertices
   lie in the square [-D, D]^2; such a triangle has area <= 2 D^2.
 - w . sigma = w . kappa: A* (d, 0) = (sigma - kappa, a), so (d, 0) = A*^-1 (sigma - kappa, a) and the t-entry gives
   0 = -w . (sigma - kappa) + tau a, tau = A*^-1[20][20]. At P* the flex column (v, tau) solves L_T v - tau 1 = 0,
   N v = 0, v_pin = 1; contracting the tie rows with the KKT multipliers of the root (stationarity:
   sum_i lam_i L_i + sum_j om_j N_j = 0, sum lam = 1/2) gives 0 - 0 = tau / 2, so tau = 0. (tau != 0 at the
   rational centre; the code never uses that entry.)
 - Orientation (verify_wide): each |det| changes by at most 4 D' + 4 D'^2 for D' = |P - c|_inf <= r + rho: the
   linear part is the L1 perimeter of the triangle's bounding box times D' (<= 4 D' in the unit triangle), the
   quadratic part is 2 x (signed area of the displacement triangle) <= 4 D'^2. The code compares this with |area| =
   |det| / 2, which is stricter than needed by a factor 2 (sound; it caps r near t*/4).
 - Only the 14 tie triangles and the 6 contact walls enter: Phi(P) <= min over the ties, and P feasible gives the
   contact-wall slacks >= 0. Non-tie triangles and other walls play no part.
Let T = the ties, W = the weak index set, S = its complement in the 20 slacks, and split
    d = z + f,   z = a v + sum_{k in W} sigma_k M_k,   f = sum_{j in S} sigma_j M_j - sum_{i in T} kappa_i M_i,
    X = X_W + X_S,   X_W = sum_{k in W} w_k sigma_k,   X_S = sum_{j in S} w_j sigma_j.
With gamma_j = 2 v^T H M_j, G = M^T H M, c = -Q(v):
    Q(z)      = -c a^2 + a sum_W gamma_k sigma_k + sum_{k,l in W} G_kl sigma_k sigma_l               (exact)
    2 z^T H f = a sum_S gamma_j sigma_j + 2 sum_{W x S} G_kj sigma_k sigma_j
                - a sum_T gamma_i kappa_i - 2 sum_{W x T} G_ki sigma_k kappa_i
             <= |a| GammaS X_S + 2 X_S sum_W Theta_k sigma_k + 2 gt |a| D^2 + 4 D^2 sum_W Om_k sigma_k
    Q(f)      = sigma_S^T G_SS sigma_S - 2 sigma_S^T G_ST kappa + kappa^T G_TT kappa
             <= LambdaS X_S^2 + 4 D^2 PhiS X_S + 4 Omega D^4
using sigma >= 0 (so only positive entries of G count in sigma-sigma couplings, and sum_S |c_j| sigma_j <=
max_S(|c_j| / w_j) X_S) and |kappa_i| <= 2 D^2, where
    GammaS = max_{j in S} |gamma_j| / w_j,           Theta_k = max_{j in S} G_kj^+ / w_j   (k in W),
    gt = sum_T |gamma_i|,                             Om_k = sum_{i in T} |G_ki|            (k in W),
    LambdaS = max_{j,j' in S} G_jj'^+ / (w_j w_j'),   PhiS = max_{j in S} sum_{i in T} |G_ji| / w_j,
    Omega = sum_{T x T} |G_ii'|.
Put a = alpha D, sigma_k = D s_k (k in W), X_S = xi D^2 and divide X = Q(z) + 2 z^T H f + Q(f) by D^2:
 (i')  E := xi [1 - GammaS |alpha| D - 2 D sum_W Theta_k s_k - 4 PhiS D^2] + (sum_W w_k s_k) / D - q3(alpha, s)
            - 2 gt |alpha| D - 4 D sum_W Om_k s_k - LambdaS D^2 xi^2 - 4 Omega D^2   <=  0,
       q3(alpha, s) = -c alpha^2 + alpha sum_W gamma_k s_k + sum_{k,l in W} G_kl s_k s_l.
Coordinatewise, |d_r| <= |a v_r + sum_W M_rk sigma_k| + sum_S |M_rj| sigma_j + 2 D^2 sum_T |M_ri|, and some
coordinate has |d_r| = D, so
 (ii') 1 <= max_r [ |alpha v_r + sum_W M_rk s_k| + D (BS_r xi + 2 m_r) ],
       BS_r = max_{j in S} |M_rj| / w_j,   m_r = sum_T |M_ri|.
Ranges: |alpha| <= 1; 0 <= s_k <= l_k + 2 D (sigma_k = L_k d + kappa_k, l_k = |L_k|_1 at P*);
0 <= xi <= xibar := 2 sum_T w_i (X_S <= X = w . kappa <= 2 D^2 sum_T w_i).
Reduction to D = r: for fixed (alpha, s >= 0, xi >= 0), E is nonincreasing in D and the right side of (ii') is
nondecreasing in D (all constants are >= 0), and the s-range grows with D. So if (i'), (ii') have no solution in
the box at D = r they have none for 0 < D <= r: Phi(P) < t* for every feasible P with 0 < |P - P*|_inf <= r.
(Tie orientations are fixed on the ball, checked as in basin10; Phi(P*) = t* by basin10 (2).)

INFEASIBILITY AT D = r (exact branch and bound over cells of y = (alpha, s_W)). On a cell with centre y_c and
radii rad:
  - A_r := upper bound of |alpha v_r + sum_W M_rk s_k| (linear form: centre value + radius terms). (ii') needs
    xi >= thr_r := (1 - A_r - 2 D m_r) / (D BS_r) for SOME r; xi_lo := max(0, min_r thr_r). If xi_lo > xibar the
    cell is empty.
  - C_lo := lower bound of the bracket multiplying xi; R_lo := lower bound of the xi-free part of E (the
    quadratic q3 by its centred form: q3(y) <= q3(y_c) + |grad q3(y_c)| . rad + rad^T |Q3| rad; no definiteness
    is assumed). Then E >= g(xi) := C_lo xi + R_lo - LambdaS D^2 xi^2 for xi >= 0; g is concave, so
    g(xi_lo) > 0 and g(xibar) > 0 give E > 0 on the whole cell for every xi in [xi_lo, xibar]: no solution.
  - otherwise the widest cell side is bisected; failure to finish raises.
The constants at P* are enclosed from the rational centre exactly as in basin10s (|A*^-1 - A(c)^-1| <= ev
entrywise, midpoint-radius intervals with exact rational endpoints); l_k uses |L*_k - L_k(c)|_1 <= 6 rho.
The lemma holds for ANY choice of W; W only decides how strong the certificate is."""
from fractions import Fraction as F
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "245-basin-sharp"))
import basin10s as S  # noqa: E402
B10 = S.B10
require = B10.require
I, isum = S.I, S.isum

WEAK = (7, 8, 12)
T = range(14)


def matrices(points, ties, walls, signs, rho, pin):
    """interval enclosures at P* of w, v, M, gamma, G, c (same construction as basin10s.constants)"""
    rows = [[g / 2 for g in B10.tie_grad(points, tr, s)] + [F(-1)] for tr, s in zip(ties, signs)]
    for p, wl in walls:
        row = [F(0)] * 21; (a, b), _ = B10.WALL[wl]; row[2 * p], row[2 * p + 1] = F(a), F(b); rows.append(row)
    Ainv = B10.inverse(rows + [[F(i == pin) for i in range(21)]])
    require(all(Ainv[pin][j] == (j == 20) for j in range(21)), "pin row is not a unit row")
    k0 = B10.norm(Ainv)
    beta = k0 * 6 * rho
    require(beta < 1, "perturbation too large")
    ev = k0 * k0 * 6 * rho / (1 - beta)
    A = [[I(x, ev) for x in row] for row in Ainv]
    w = [-A[20][j] for j in range(20)]
    v = [A[r][20] for r in range(20)]
    M = [[A[r][j] for j in range(20)] for r in range(20)]
    require(min(x.lo() for x in w) > 0, "stress not robustly positive")
    H = [[I(0) for _ in range(20)] for _ in range(20)]
    for wi, tr, s in zip(w, ties, signs):
        for (r, c), val in B10.det_hessian(tr, s).items():
            H[r][c] = H[r][c] + wi * F(val, 4)
    Hv = [isum(H[r][c] * v[c] for c in range(20)) for r in range(20)]
    HM = [[isum(H[r][c] * M[c][j] for c in range(20)) for j in range(20)] for r in range(20)]
    c_flex = -isum(v[r] * Hv[r] for r in range(20))
    gamma = [2 * isum(Hv[r] * M[r][j] for r in range(20)) for j in range(20)]
    G = [[isum(M[r][j] * HM[r][k] for r in range(20)) for k in range(20)] for j in range(20)]
    require(c_flex.lo() > 0, "no negative curvature along the flex")
    # |L*_k|_1: tie rows move by at most rho per entry (6 entries); wall rows are constant
    l1 = [sum(abs(x) for x in rows[k][:20]) + (6 * rho if k < 14 else 0) for k in range(20)]
    return dict(w=w, v=v, M=M, gamma=gamma, G=G, c=c_flex, l1=l1, k0=k0, ev=ev)


def constants(points, ties, walls, signs, rho, pin=12, weak=WEAK):
    """the exact bounds (rationals, each taken at the safe end of its interval) used by (i') and (ii')"""
    W = tuple(weak)
    require(len(set(W)) == len(W) and all(type(k) is int and 0 <= k < 20 for k in W),
            "weak set must be distinct slack indices")
    Sx = [j for j in range(20) if j not in W]
    E = matrices(points, ties, walls, signs, rho, pin)
    w, v, M, gamma, G, c = (E[k] for k in ("w", "v", "M", "gamma", "G", "c"))
    wlo = [x.lo() for x in w]

    def pos(x):
        return max(x.hi(), F(0))

    n = len(W) + 1
    # soft quadratic form q3 on y = (alpha, s_W): interval matrix (midpoints and entrywise radii)
    Q3 = [[None] * n for _ in range(n)]
    Q3[0][0] = -c
    for i, k in enumerate(W):
        Q3[0][i + 1] = gamma[k] * F(1, 2); Q3[i + 1][0] = gamma[k] * F(1, 2)
        for j, l in enumerate(W):
            Q3[i + 1][j + 1] = G[k][l]
    return dict(
        pin=pin, weak=W, k0=E["k0"], ev=E["ev"],
        c=c.lo(),
        GammaS=max(gamma[j].absmax() / wlo[j] for j in Sx),
        Theta=[max(pos(G[k][j]) / wlo[j] for j in Sx) for k in W],
        gt=sum(gamma[i].absmax() for i in T),
        Om=[sum(G[k][i].absmax() for i in T) for k in W],
        LambdaS=max(pos(G[j][k]) / (wlo[j] * wlo[k]) for j in Sx for k in Sx),
        PhiS=max(sum(G[j][i].absmax() for i in T) / wlo[j] for j in Sx),
        Omega=sum(G[i][k].absmax() for i in T for k in T),
        BS=[max(M[r][j].absmax() / wlo[j] for j in Sx) for r in range(20)],
        m=[sum(M[r][i].absmax() for i in T) for r in range(20)],
        wW=[wlo[k] for k in W],
        l1=[E["l1"][k] for k in W],
        xibar=2 * sum(w[i].hi() for i in T),
        Q3m=[[x.m for x in row] for row in Q3], Q3r=[[x.r for x in row] for row in Q3],
        Vm=[[v[r].m] + [M[r][k].m for k in W] for r in range(20)],
        Vr=[[v[r].r] + [M[r][k].r for k in W] for r in range(20)])


def cell_bounds(K, D, lo, hi):
    """exact (xi_lo, C_lo, R_lo) on the cell lo <= y <= hi, y = (alpha, s_W); see the docstring.
    xi_lo is None when (ii') cannot hold for any xi <= xibar."""
    n = len(lo)
    yc = [(a + b) / 2 for a, b in zip(lo, hi)]
    rad = [(b - a) / 2 for a, b in zip(lo, hi)]
    ymax = [abs(c) + r for c, r in zip(yc, rad)]
    xi_lo = None
    for r in range(20):
        vm, vr = K["Vm"][r], K["Vr"][r]
        A = (abs(sum(c * y for c, y in zip(vm, yc))) + sum(abs(c) * q for c, q in zip(vm, rad))
             + sum(e * y for e, y in zip(vr, ymax)))
        thr = (1 - A - 2 * D * K["m"][r]) / (D * K["BS"][r])        # BS_r > 0 is required by infeasible()
        if xi_lo is None or thr < xi_lo:
            xi_lo = thr
    xi_lo = max(xi_lo, F(0))
    if xi_lo > K["xibar"]:
        return None, None, None
    amax = ymax[0]
    C_lo = (1 - K["GammaS"] * amax * D - 2 * D * sum(t * s for t, s in zip(K["Theta"], hi[1:]))
            - 4 * K["PhiS"] * D * D)
    Qm, Qr = K["Q3m"], K["Q3r"]
    q_c = sum(Qm[i][j] * yc[i] * yc[j] for i in range(n) for j in range(n))
    grad = [sum((Qm[i][j] + Qm[j][i]) * yc[j] for j in range(n)) for i in range(n)]
    q_hi = (q_c + sum(abs(g) * q for g, q in zip(grad, rad))
            + sum(abs(Qm[i][j]) * rad[i] * rad[j] for i in range(n) for j in range(n))
            + sum(Qr[i][j] * ymax[i] * ymax[j] for i in range(n) for j in range(n)))
    lin = F(0)
    for i in range(n - 1):
        coef = K["wW"][i] / D - 4 * D * K["Om"][i]
        lin += coef * (lo[i + 1] if coef >= 0 else hi[i + 1])
    R_lo = lin - q_hi - 2 * K["gt"] * amax * D - 4 * K["Omega"] * D * D
    return xi_lo, C_lo, R_lo


def infeasible(K, D, max_cells=400000, min_width=F(1, 2**40)):
    """exact: (i'), (ii') have no solution in the box at this D. Returns (ok, cells)."""
    n = len(K["weak"]) + 1
    require(D > 0, "radius must be positive")
    require(all(K[k] >= 0 for k in ("GammaS", "gt", "LambdaS", "PhiS", "Omega", "xibar"))
            and all(x >= 0 for k in ("Theta", "Om", "m", "wW", "l1") for x in K[k])
            and all(x > 0 for x in K["BS"]), "constants must be nonnegative")
    lo0 = [F(-1)] + [F(0)] * (n - 1)
    hi0 = [F(1)] + [l + 2 * D for l in K["l1"]]
    todo, cells = [(lo0, hi0)], 0
    LD = K["LambdaS"] * D * D
    xb = K["xibar"]
    while todo:
        lo, hi = todo.pop()
        cells += 1
        if cells > max_cells:
            return False, cells
        xi_lo, C_lo, R_lo = cell_bounds(K, D, lo, hi)
        if xi_lo is None:
            continue
        if C_lo * xi_lo + R_lo - LD * xi_lo * xi_lo > 0 and C_lo * xb + R_lo - LD * xb * xb > 0:
            continue
        widths = [b - a for a, b in zip(lo, hi)]
        i = max(range(n), key=lambda k: widths[k])
        if widths[i] < min_width:
            return False, cells
        mid = (lo[i] + hi[i]) / 2
        h1 = list(hi); h1[i] = mid
        l2 = list(lo); l2[i] = mid
        todo += [(lo, h1), (l2, hi)]
    return True, cells


def verify_wide(points, ties, walls, t0, cover_radius, pin=12, weak=WEAK):
    """basin10 steps (1), (2) and feasibility (exactly as basin10s.verify_sharp), then the wide basin at
    r = cover_radius + rho. Raises on failure; returns the constants."""
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
    require(F(cover_radius) > 0, "radius must be positive")
    r = F(cover_radius) + rho
    # tie orientations cannot change within r of P* (as basin10): |det change| <= 4 D + 4 D^2 in doubled area
    require(4 * (r + rho) + 4 * (r + rho) ** 2 < min(abs(B10.area(points, t)) for t in ties),
            "tie orientation may change")
    K = constants(points, ties, walls, ex["signs"], rho, pin, weak)
    ok, cells = infeasible(K, r)
    require(ok, "wide basin inequalities have a solution")
    return dict(rho=rho, r=r, cells=cells, t_lower=ex["t_lower"], t_upper=ex["t_upper"], signs=ex["signs"], **K)


def show(out):
    def fl(x):
        if isinstance(x, F):
            return float(x)
        if isinstance(x, list) and x and isinstance(x[0], F):
            return [float(y) for y in x]
        return x
    keys = ("r", "rho", "cells", "weak", "c", "GammaS", "Theta", "gt", "Om", "LambdaS", "PhiS", "Omega", "xibar",
            "wW", "l1")
    d = {k: fl(out[k]) for k in keys}
    d["max BS"] = float(max(out["BS"])); d["max m"] = float(max(out["m"]))
    return d


if __name__ == "__main__":
    import time
    pts, ties, walls, t0 = B10.load()
    args = [F(a) for a in sys.argv[1:]] or [F(5, 1000), F(53, 10000), F(54, 10000)]
    for R in args:
        t = time.time()
        try:
            out = verify_wide(pts, ties, walls, t0, R)
            print("radius", float(R), "OK", show(out), "%.1fs" % (time.time() - t), flush=True)
        except Exception as e:
            print("radius", float(R), "FAIL", e, "%.1fs" % (time.time() - t), flush=True)
