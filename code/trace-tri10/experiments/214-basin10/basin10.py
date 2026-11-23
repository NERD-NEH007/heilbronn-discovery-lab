"""Exp 214 (Claude, 2026-11-18): exact basin checker for the triangle n = 10 optimum (trust base: standard library
plus exp 152 verify_trace helpers). Same role as exp 202 basin8.py; the n = 10 optimum is degenerate at first
order (14 ties + 6 walls on 20 coordinates leave a 1-dimensional flex), so existence uses the full KKT system and
the basin uses the second-order lemma of exp 139 / 170.

Data: exp 138 local certificate (rational centre c of the 10 points, the 14 tie triples, the 6 wall contacts, t0).

(1) Existence (Newton-Kantorovich, sup norm) for the 41 x 41 KKT system in u = (z in R^20, t, w in R^14, om in R^6):
      G1_i = sigma_i det_i(z) - 2t,   G2_j = n_j . p_j - off_j,
      G3   = sum_i w_i sigma_i grad det_i(z) + sum_j om_j N_j   (20 rows),   G4 = 2 sum_i w_i - 1.
    (det = doubled signed area, t = area.) J is affine in u; |J(u) - J(u')| <= L |u - u'| with L = 56:
    a G1 row has 6 coordinate-difference entries (<= 12 |dz|); a G3 row changes by <= 2 |dw| for each of <= 14
    ties through the constant Hessians and by <= 2 |dz| for each of <= 14 w-columns (<= 56 |du|).
    u0: c with t0 and multipliers from one exact Newton step (floats only propose; everything is re-checked).
    An approximate inverse Y (floats, made exact) with d = |I - Y J(u0)| < 1 gives |J^-1| <= |Y|/(1-d) =: beta and
    |J^-1 G(u0)| <= |Y G(u0)|/(1-d) =: eta. If h = beta L eta <= 1/2 a root u* exists with |u* - u0| <= 2 eta.
    The root's coordinates P* are within rho := |z0 - c| + 2 eta of c, and |t* - t0'| <= 2 eta.
(2) Value at P*: every non-tie triangle has |det| > 2 t_hi at P* (gradient + quadratic remainder bounds), the tie
    orientations are those at c, and all 10 points lie in the triangle. So Phi(P*) = t*.
(3) Basin (exp 139 bounds recomputed at c with row perturbation 6 rho, exp 170 inequality): every feasible P with
    |P - P*| <= r has Phi(P) <= Phi(P*), equality only at P*.
(4) Cover: every point within R of c is within R + rho of P*; verify(.., R) checks (3) at r = R + rho."""
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "138-triangle-n10-local" / "local-certificate.json"
sys.path.insert(0, str(HERE.parent / "152-geometry-trace"))
from verify_trace import determinant, require, vertices   # noqa: E402

N = 10
WALL = {"L": ((1, 0), 0), "B": ((0, 1), 0), "H": ((-1, -1), -1)}   # inward normal . p = offset
LIP = 56


def load():
    data = json.loads(SOURCE.read_text())["10"][0]
    require(data["certified"] and data["mode"] == "second-order" and data["n_core"] == 10 and not data["rattlers"],
            "unexpected certificate")
    points = tuple(tuple(map(F, p)) for p in data["points_exact"])
    extra = data["extra"]
    ties = extra["exact_ties"]
    ties = [tuple(t) for t in (json.loads(ties) if isinstance(ties, str) else ties)]
    walls = [(int(p), w) for p, w in extra["exact_wall_contacts"]]
    t0 = F(str(extra["t_polished"]))
    return points, ties, walls, t0


# ---------------------------------------------------------------- linear algebra (exact unless noted)
def inverse(rows):
    n = len(rows)
    aug = [list(map(F, row)) + [F(i == j) for j in range(n)] for i, row in enumerate(rows)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if aug[i][j]), None)
        require(pivot is not None, "singular matrix")
        aug[j], aug[pivot] = aug[pivot], aug[j]
        inv = 1 / aug[j][j]
        aug[j] = [v * inv for v in aug[j]]
        for i in range(n):
            if i != j and aug[i][j]:
                f = aug[i][j]
                aug[i] = [a - f * b for a, b in zip(aug[i], aug[j])]
    return [row[n:] for row in aug]


def float_inverse(rows):
    """Gauss-Jordan in floats with partial pivoting; only proposes Y (checked exactly through |I - Y J|)."""
    n = len(rows)
    aug = [[float(v) for v in row] + [float(i == j) for j in range(n)] for i, row in enumerate(rows)]
    for j in range(n):
        p = max(range(j, n), key=lambda i: abs(aug[i][j]))
        require(aug[p][j] != 0, "singular matrix")
        aug[j], aug[p] = aug[p], aug[j]
        inv = 1 / aug[j][j]; aug[j] = [v * inv for v in aug[j]]
        for i in range(n):
            if i != j and aug[i][j]:
                f = aug[i][j]; aug[i] = [a - f * b for a, b in zip(aug[i], aug[j])]
    return [[F(v) for v in row[n:]] for row in aug]


def matvec(M, x):
    return [sum(a * b for a, b in zip(row, x) if a and b) for row in M]


def matmul(A, B):
    cols = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col) if a and b) for col in cols] for row in A]


def norm(M):
    return max(sum(abs(v) for v in row) for row in M)


def maxabs(values):
    return max(map(abs, values))


def area(points, triple):
    return determinant(*(points[i] for i in triple)) / 2


def tie_grad(points, triple, sign):
    """gradient of sign * det over the 20 coordinates"""
    row = [F(0)] * (2 * N)
    i, j, k = triple
    for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
        row[2 * a] += sign * (points[b][1] - points[c][1])
        row[2 * a + 1] += sign * (points[c][0] - points[b][0])
    return row


def det_hessian(triple, sign):
    """constant Hessian of sign * det: dict {(r, s): value}"""
    H = {}
    i, j, k = triple
    for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
        # d/dx_a det = y_b - y_c ; d/dy_a det = x_c - x_b
        for (r, s, v) in ((2 * a, 2 * b + 1, 1), (2 * a, 2 * c + 1, -1), (2 * a + 1, 2 * c, 1), (2 * a + 1, 2 * b, -1)):
            H[(r, s)] = H.get((r, s), 0) + sign * v
    return H


# ---------------------------------------------------------------- (1) existence
def kkt(u, ties, walls, signs):
    z, t = u[:20], u[20]; w = u[21:35]; om = u[35:41]
    pts = [(z[2 * p], z[2 * p + 1]) for p in range(N)]
    G, J = [], []
    grads = [tie_grad(pts, tr, s) for tr, s in zip(ties, signs)]
    hess = [det_hessian(tr, s) for tr, s in zip(ties, signs)]
    normals = []
    for p, wl in walls:
        (a, b), off = WALL[wl]; nrow = [F(0)] * 20; nrow[2 * p], nrow[2 * p + 1] = F(a), F(b); normals.append((nrow, off, p))
    for i, (tr, s) in enumerate(zip(ties, signs)):
        G.append(s * determinant(*(pts[k] for k in tr)) - 2 * t)
        J.append(grads[i] + [F(-2)] + [F(0)] * 20)
    for nrow, off, p in normals:
        G.append(nrow[2 * p] * z[2 * p] + nrow[2 * p + 1] * z[2 * p + 1] - off)
        J.append(nrow + [F(0)] * 21)
    for r in range(20):
        G.append(sum(w[i] * grads[i][r] for i in range(14)) + sum(om[j] * normals[j][0][r] for j in range(6)))
        row = [F(0)] * 41
        for i in range(14):
            for (a, b), v in hess[i].items():       # hess holds the full symmetric matrix (both orderings)
                if a == r: row[b] += w[i] * v
            row[21 + i] = grads[i][r]
        for j in range(6):
            row[35 + j] = normals[j][0][r]
        J.append(row)
    G.append(2 * sum(w) - 1)
    J.append([F(0)] * 21 + [F(2)] * 14 + [F(0)] * 6)
    return G, J


def existence(points, ties, walls, t0):
    signs = [1 if determinant(*(points[i] for i in tr)) > 0 else -1 for tr in ties]
    z0 = [c for p in points for c in p]
    # multipliers proposed in floats from the stationarity rows, then one exact Newton step on the whole system
    u = z0 + [t0] + [F(1, 28)] * 14 + [F(0)] * 6
    for _ in range(2):
        G, J = kkt(u, ties, walls, signs)
        Y = float_inverse(J)
        step = matvec(Y, G)
        u = [a - b for a, b in zip(u, step)]
        u = [F(round(v * 10**60), 10**60) for v in u]
    G, J = kkt(u, ties, walls, signs)
    Y = float_inverse(J)
    I_YJ = matmul(Y, J)
    d = max(sum(abs((F(i == j)) - I_YJ[i][j]) for j in range(41)) for i in range(41))
    require(d < 1, "approximate inverse too poor")
    beta = norm(Y) / (1 - d)
    eta = maxabs(matvec(Y, G)) / (1 - d)
    h = beta * LIP * eta
    require(2 * h <= 1, "Kantorovich condition fails")
    root = 2 * eta
    shift = maxabs([a - b for a, b in zip(u[:20], z0)])
    rho = shift + root
    t_lo, t_hi = u[20] - root, u[20] + root
    require(t_lo > 0, "nonpositive value")
    require(min(u[21:41]) - root > 0, "multipliers not robustly positive")
    return dict(rho=rho, t_lower=t_lo, t_upper=t_hi, kantorovich_h=h, signs=signs, inverse_defect=d)


# ---------------------------------------------------------------- (2) value at P*, orientation, feasibility
def value_checks(points, ties, signs, rho, t_hi):
    require(all(x >= 0 and y >= 0 and x + y <= 1 for x, y in points), "infeasible centre")
    tieset = set(ties)
    for tri in combinations(range(N), 3):
        dval = abs(determinant(*(points[i] for i in tri)))
        grad = sum(abs(v) for v in tie_grad(points, tri, 1))
        if tri in tieset:
            require(dval - grad * rho - 4 * rho * rho > 0, "tie orientation may differ at P*")
        else:
            require(dval - grad * rho - 4 * rho * rho > 2 * t_hi, "a non-tie triangle may be minimal at P*")


# ---------------------------------------------------------------- (3) second-order basin (exp 139 + exp 170)
def basin(points, ties, walls, signs, rho, radius):
    rows = []
    for tr, s in zip(ties, signs):
        row = [g / 2 for g in tie_grad(points, tr, s)] + [F(-1)]      # derivatives of signed areas, minus t
        rows.append(row)
    for p, wl in walls:
        row = [F(0)] * 21; (a, b), _ = WALL[wl]; row[2 * p], row[2 * p + 1] = F(a), F(b); rows.append(row)
    for pin in range(20):                         # as exp 139: first pin whose bordered matrix is regular and well scaled
        try:
            Ainv = inverse(rows + [[F(i == pin) for i in range(21)]])
        except Exception:
            continue
        if maxabs(row[-1] for row in Ainv) <= F(11, 10):
            break
    else:
        require(False, "no well-scaled coordinate pin")
    k0 = norm(Ainv)
    beta = k0 * 6 * rho
    require(beta < 1, "perturbation too large")
    k_bound = k0 / (1 - beta)
    ev = k_bound * beta
    v0 = [row[-1] for row in Ainv]
    v_bound = maxabs(v0) + ev
    stress = [-x for x in Ainv[-1][:-1]]
    stress_error = sum(map(abs, Ainv[-1])) * beta / (1 - beta)
    weights = [x - stress_error for x in stress]
    require(min(weights) > 0, "stress not robustly positive")
    b_bound = max((maxabs(row[j] for row in Ainv) + ev) / wj for j, wj in enumerate(weights))
    tangent = list(zip(v0[:20:2], v0[1:20:2]))
    q0 = sum(wi * s * area(tangent, tr) for wi, s, tr in zip(stress, signs, ties))
    q_error = 4 * maxabs(v0) ** 2 * stress_error + 8 * maxabs(v0) * ev + 4 * ev ** 2
    c = -(q0 + q_error)
    require(c > 0, "no negative curvature along the flex")
    r = radius
    require(4 * (r + rho) + 4 * (r + rho) ** 2 < min(abs(area(points, t)) for t in ties), "tie orientation may change")
    coef = 8 * v_bound + 4 * (1 + v_bound)
    den = 1 - b_bound * coef * r
    require(den > 0, "absorption fails")
    u = 2 * k_bound / den * r
    require(u < 1, "transverse bound too large")
    margin = c * ((1 - u) / v_bound) ** 2 - (8 * v_bound + 4 * u) * u
    require(margin > 0, "second-order basin inequality fails")
    return dict(pin=pin, K=k_bound, B=b_bound, V=v_bound, c=c, margin=margin)


def verify(points, ties, walls, t0, cover_radius):
    """(1)-(3) with basin radius r = cover_radius + rho about P*. Returns the bounds; raises on any failure."""
    require(len(points) == N and len(ties) == 14 and len(walls) == 6, "unexpected active set")
    # structural validation (audit, 2026-11-18): distinct, well-formed ties and wall contacts
    require(all(len(t) == 3 and all(type(i) is int for i in t) and 0 <= t[0] < t[1] < t[2] < N for t in ties)
            and len(set(ties)) == 14, "ties must be 14 distinct sorted index triples")
    require(all(0 <= p < N and w in WALL for p, w in walls) and len(set(walls)) == 6, "walls must be 6 distinct contacts")
    for p, w in walls:
        (a, b), off = WALL[w]
        require(a * points[p][0] + b * points[p][1] == off, "centre not exactly on its wall")
    ex = existence(points, ties, walls, t0)
    # feasibility of the exact root (audit): every point keeps a slack > 2 rho to each wall it is not in contact with
    # (|z* - c| <= rho per coordinate; the hypotenuse slack 1 - x - y changes by at most 2 rho). Contact walls hold
    # exactly at the root because they are equations of the KKT system.
    contacts = set(walls)
    for p, (x, y) in enumerate(points):
        for w, ((a, b), off) in WALL.items():
            if (p, w) not in contacts:
                require(a * x + b * y - off > 2 * ex["rho"], "a point may leave the triangle at the exact root")
    value_checks(points, ties, ex["signs"], ex["rho"], ex["t_upper"])
    bs = basin(points, ties, walls, ex["signs"], ex["rho"], cover_radius + ex["rho"])
    return dict(rho=ex["rho"], t_lower=ex["t_lower"], t_upper=ex["t_upper"], kantorovich_h=ex["kantorovich_h"],
                inverse_defect=ex["inverse_defect"], **{k: v for k, v in bs.items()})


def verify_cover(box, q, points, radius, barycentric_permutation, point_assignment):
    """As exp 153 / basin8: every vertex of every point domain, mapped by the simplex symmetry, lies within
    `radius` (sup norm) of its assigned centre point."""
    require(len(box) == len(points), "cover requires a full local box")
    require(sorted(barycentric_permutation) == [0, 1, 2], "invalid simplex symmetry")
    require(sorted(point_assignment) == list(range(len(points))), "invalid point assignment")
    for domain, target in zip(box, point_assignment):
        tx, ty = points[target]
        for x, y in vertices(domain, q):
            bary = (q - x - y, x, y)
            px, py = (F(bary[i], q) for i in barycentric_permutation[1:])
            require(max(abs(px - tx), abs(py - ty)) <= radius, "point outside the basin")


def production_radius():
    """The cover radius the search engine uses (exp 157/212 run.load_cert): 1/163840 (1 - 1e-6) - box_rho."""
    e = json.loads(SOURCE.read_text())["10"][0]["extra"]
    return F(1, 163840) * F(999999, 1000000) - F(e["box_rho_exact"])


if __name__ == "__main__":
    pts, ties, walls, t0 = load()
    for r in (production_radius(), F(1, 120000) - F(1, 10**30), F(1, 50000)):
        try:
            out = verify(pts, ties, walls, t0, r)
            print("radius", float(r), "OK", {k: float(v) for k, v in out.items()})
        except Exception as e:
            print("radius", float(r), "FAIL", e)
    print("source sha256", sha256(SOURCE.read_bytes()).hexdigest())
