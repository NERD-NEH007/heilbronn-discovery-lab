"""Exp 202 (Claude, 2026-11-18): exact basin checker for the triangle n = 8 optimum (algebraic value).

Unlike n = 9, the optimum P* is not rational. The certificate (exp 113) gives a rational centre c with the
active set: 11 ties and 6 wall contacts. Every check below is exact rational arithmetic.

(1) Existence (Newton-Kantorovich, infinity norm). The unknowns are z = (16 coordinates, t). G(z) = 0 is
    sigma_i det_i(P) - 2t = 0 for the ties, and the wall equations (x = 0, y = 0, x + y = 1). Take
    z0 = (c, t0), A = J(z0)^-1, eta = |A G(z0)|, beta = |A|. J is affine in z: each tie row has 6 coordinate
    entries, each a difference of two coordinates, so |J(z) - J(z')| <= 12 |z - z'| and L = 12. If
    h = beta L eta <= 1/2, G has a zero z* with |z* - z0| <= 2 eta =: rho. Then P* has exactly tied
    triangles with common doubled area 2t*, and |t* - t0| <= rho.
(2) Value at P*. Every non-tie triangle has |det| > 2t* at P*, checked from c with the gradient and
    quadratic remainder bounds. So Phi(P*) = t*.
(3) First-order basin. Let M = [J_coords | 1] be the bordered 17 x 17 matrix of the loaded rows at P*. M
    differs from M_c (at c) by E with |E| <= 12 rho. With K = M_c^-1, |K| 12 rho < 1, and Neumann bounds,
    every entry of M^-1 is within eta2 = |K|^2 12 rho / (1 - |K| 12 rho) of K. The stress y (last row of
    M^-1) is then positive, and the displacement bound of the exp 153 argument holds with
    alpha = 1 / max_{i,j} (|K_ij| + eta2) / (y_c_j - eta2).
    Suppose P is feasible with |P - P*| <= r and Phi(P) >= t*. Then for each tie, g_i . d >= -4 r^2 (the
    doubled area of the displacement triangle is at most 4 r^2, and orientations are fixed). Each wall
    gives n_j . d >= 0. Stationarity then gives |d| <= 4 r^2 / alpha < r, a contradiction unless d = 0
    (rescale to s = |d|). So within radius r of P*, Phi <= t*, with equality only at P*.
(4) Cover radius for the trace: R = r - rho around c, so every point within R of c is within r of P*."""
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "113-triangle-n8" / "cert" / "local_certs_tri.json"
sys.path.insert(0, str(HERE.parent / "152-geometry-trace"))
from verify_trace import determinant, require, vertices   # noqa: E402

N = 8
WALL = {"L": ((1, 0), 0), "B": ((0, 1), 0), "H": ((-1, -1), -1)}   # inward normal . p = offset


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


def norm(mat):
    return max(sum(abs(v) for v in row) for row in mat)


def tie_row(points, triple, sign):
    row = [F(0)] * (2 * N)
    i, j, k = triple
    for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
        row[2 * a] += sign * (points[b][1] - points[c][1])
        row[2 * a + 1] += sign * (points[c][0] - points[b][0])
    return row


def production_radius():
    """The cover radius used by the search engine (exp 127 run.load_cert): basin * (1 - 1e-6) - box_rho."""
    e = json.loads(SOURCE.read_text())["8"][0]["extra"]
    return F(e["basin_radius_exact"]) * F(999999, 1000000) - F(e["box_rho_exact"])


def load():
    data = json.loads(SOURCE.read_text())["8"][0]
    raw = data["points_exact"]
    raw = json.loads(raw.replace("'", '"')) if isinstance(raw, str) else raw
    points = tuple(tuple(map(F, p)) for p in raw)
    extra = data["extra"]
    ties = [tuple(t) for t in extra["exact_ties"]]
    walls = [(int(p), w) for p, w in extra["exact_wall_contacts"]]
    t0 = F(str(extra["t_polished"]))
    return points, ties, walls, t0


def verify(points, ties, walls, t0, cover_radius):
    """Check existence, the value at P* and the first-order basin of radius r = cover_radius + rho around P*,
    where rho is the certified centre error. Every point within cover_radius of the rational centre c is then
    within r of P*. Returns the bounds. Raises on any failure."""
    require(len(points) == N and len(ties) == 11 and len(walls) == 6, "unexpected active set")
    require(all(x >= 0 and y >= 0 and x + y <= 1 for x, y in points), "infeasible centre")
    for p, w in walls:
        (a, b), off = WALL[w]
        require(a * points[p][0] + b * points[p][1] == off, "centre not exactly on its wall")
    sign = {t: (1 if determinant(*(points[i] for i in t)) > 0 else -1) for t in ties}
    # (1) existence
    G = [sign[t] * determinant(*(points[i] for i in t)) - 2 * t0 for t in ties]
    J = [tie_row(points, t, sign[t]) + [F(-2)] for t in ties]
    for p, w in walls:
        (a, b), _ = WALL[w]
        row = [F(0)] * (2 * N + 1); row[2 * p], row[2 * p + 1] = F(a), F(b); J.append(row); G.append(F(0))
    A = inverse(J)
    eta = max(abs(sum(a * g for a, g in zip(row, G))) for row in A)
    beta = norm(A)
    h = beta * 12 * eta
    require(2 * h <= 1, "Kantorovich condition fails")
    rho = 2 * eta
    t_lo, t_hi = t0 - rho, t0 + rho
    require(t_lo > 0, "nonpositive value")
    # (2) every non-tie triangle stays strictly above 2t* at P*
    for tri in combinations(range(N), 3):
        if tri in sign:
            continue
        d = abs(determinant(*(points[i] for i in tri)))
        grad = sum(abs(v) for v in tie_row(points, tri, 1))
        require(d - grad * rho - 4 * rho * rho > 2 * t_hi, "a non-tie triangle may be minimal at P*")
    # (3) first-order basin around P*
    radius = cover_radius + rho
    rows = [tie_row(points, t, sign[t]) for t in ties]
    for p, w in walls:
        (a, b), _ = WALL[w]
        row = [F(0)] * (2 * N); row[2 * p], row[2 * p + 1] = F(a), F(b); rows.append(row)
    K = inverse([row + [F(1)] for row in rows])
    kn = norm(K)
    require(kn * 12 * rho < 1, "perturbation too large")
    eta2 = kn * kn * 12 * rho / (1 - kn * 12 * rho)
    stress = K[-1]
    require(min(stress) - eta2 > 0, "stress not robustly positive")
    max_disp = max((abs(K[i][j]) + eta2) / (stress[j] - eta2) for i in range(2 * N) for j in range(len(K)))
    alpha = 1 / max_disp
    require(4 * radius < alpha, "quadratic remainder can exceed coercivity")
    gmax = max(sum(abs(v) for v in row) for row in rows[:len(ties)]) + 12 * rho
    require(gmax * radius + 4 * radius * radius < 2 * t_lo, "tie orientation may change")
    return dict(rho=rho, t_lower=t_lo, t_upper=t_hi, alpha=alpha, kantorovich_h=h, min_stress=min(stress) - eta2)


def verify_cover(box, q, points, radius, barycentric_permutation, point_assignment):
    """Same geometry as exp 153: every vertex of every point domain, mapped by the simplex symmetry, lies
    within `radius` (sup norm) of its assigned centre point."""
    require(len(box) == len(points), "cover requires a full local box")
    require(sorted(barycentric_permutation) == [0, 1, 2], "invalid simplex symmetry")
    require(sorted(point_assignment) == list(range(len(points))), "invalid point assignment")
    for domain, target in zip(box, point_assignment):
        tx, ty = points[target]
        for x, y in vertices(domain, q):
            bary = (q - x - y, x, y)
            px, py = (F(bary[i], q) for i in barycentric_permutation[1:])
            require(max(abs(px - tx), abs(py - ty)) <= radius, "point outside the basin")


if __name__ == "__main__":
    pts, ties, walls, t0 = load()
    for r in (production_radius(), F(6, 10**3)):
        try:
            out = verify(pts, ties, walls, t0, r)
            print("radius", r, "OK", {k: float(v) for k, v in out.items()})
        except Exception as e:
            print("radius", r, "FAIL", e)
    print("source sha256", sha256(SOURCE.read_bytes()).hexdigest())
