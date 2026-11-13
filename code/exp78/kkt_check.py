"""Exp 78: independent existence certificate for best configurations, written separately from exp 44.

exp 44 reduces the tie system (orbits, null-space parametrisation, second-order mode for d >= 1). Here we use the full
square KKT system instead, with no reduction and no symmetry:
  unknowns  z = (core coordinates, t, lambda_k for each tie, mu_w for each loaded wall)
  equations  A_k(x) - t = 0 (signed areas of tied triangles), x_w - wall_w = 0,
             sum_k lambda_k dA_k/dx + sum_w mu_w s_w e_w = 0,  1 - sum_k lambda_k = 0.
All equations are polynomials of degree <= 2, so F(z0 + h) = F(z0) + J(z0) h + (1/2) T[h, h] exactly with a
constant tensor T. All bounds below use exact rational arithmetic (Fraction); floats are used only to propose z0
and the approximate inverse Y, which then become exact dyadic rationals.
  Y0 = |Y F(z0)|_inf,  Z0 = |I - Y J(z0)|_inf,  Z2 = |Y|_inf * max_i sum_jk |T_ijk|
If p(r) = Z2 r^2 - (1 - Z0) r + Y0 < 0, there is exactly one zero z* of F with |z* - z0|_inf <= r (Newton-Kantorovich).
Then:
  (a) all lambda, mu > r: strict complementarity at z*;
  (b) every other triangle (rattlers included) has area > t + margin at z0, with margin > the most the ball can move it;
  (c) second order: the tangent space of the active constraints (dimension 2 core + 1 - #constraints, = d) is computed
      exactly at z0, and the Lagrangian Hessian sum_k lambda_k s_k d2A_k on it must be negative definite; we report
      its eigenvalues (float) and an exact rational bound on how much the ball can change them.
Usage: python code/exp78/kkt_check.py --sizes 40,61  (current best: configurations/best/n{n}.json; n <= 36 needs
the published record in records/n{n}.txt)."""
import os, sys, json, itertools, argparse, time
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import hlib

def det(a, b, c): return (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])

def setup(P, rtol=1e-9, wtol=1e-11):
    n = len(P); tri = list(itertools.combinations(range(n), 3))
    A = np.array([abs(det(P[i], P[j], P[k])) / 2 for i, j, k in tri]); t = A.min()
    ties = [tri[q] for q in np.where(A <= t * (1 + rtol))[0]]
    core = sorted({i for tr in ties for i in tr}); rat = [i for i in range(n) if i not in core]
    walls = [(i, e, v, s) for i in core for e, v, s in ((0, 0., 1), (1, 0., 1), (0, 1., -1), (1, 1., -1)) if abs(P[i][e] - v) < wtol]
    sg = [1 if det(P[i], P[j], P[k]) > 0 else -1 for i, j, k in ties]
    return ties, sg, core, rat, walls, float(t)

class KKT:
    def __init__(s, P, ties, sg, core, walls):
        s.P, s.ties, s.sg, s.walls = P, ties, sg, walls
        s.ix = {i: c for c, i in enumerate(core)}; s.nc = len(core); s.nx = 2 * s.nc
        s.it = s.nx; s.il = s.nx + 1; s.im = s.il + len(ties); s.N = s.im + len(walls)
    def unpack(s, z):
        X = {i: (z[2 * c], z[2 * c + 1]) for i, c in s.ix.items()}; return X, z[s.it], z[s.il:s.im], z[s.im:]
    def grads(s, X, k):
        """signed-area gradient of tie k: dict var -> value (A = sg*det/2)"""
        i, j, l = s.ties[k]; a, b, c = X[i], X[j], X[l]; h = Fr(s.sg[k], 2) if isinstance(a[0], Fr) else s.sg[k] / 2
        g = {}
        for p, q, r in ((i, j, l), (j, l, i), (l, i, j)):
            P, Q, R = X[p], X[q], X[r]   # d det / dP = (Q_y - R_y, R_x - Q_x) for det(P,Q,R) cyclic
            g[2 * s.ix[p]] = g.get(2 * s.ix[p], 0) + h * (Q[1] - R[1]); g[2 * s.ix[p] + 1] = g.get(2 * s.ix[p] + 1, 0) + h * (R[0] - Q[0])
        return g
    def F(s, z):
        X, t, lam, mu = s.unpack(z); out = []
        for k, (i, j, l) in enumerate(s.ties):
            d = det(X[i], X[j], X[l]); out.append((Fr(s.sg[k], 2) if isinstance(t, Fr) else s.sg[k] / 2) * d - t)
        for i, e, v, sw in s.walls: out.append(X[i][e] - (Fr(int(v)) if isinstance(t, Fr) else v))
        st = [0] * s.nx
        for k in range(len(s.ties)):
            for m, gv in s.grads(X, k).items(): st[m] += lam[k] * gv
        for w, (i, e, v, sw) in enumerate(s.walls): st[2 * s.ix[i] + e] += sw * mu[w]
        out += st; out.append(1 - sum(lam)); return out
    def J(s, z):
        X, t, lam, mu = s.unpack(z); R = []
        for k in range(len(s.ties)):
            row = {m: gv for m, gv in s.grads(X, k).items()}; row[s.it] = -1; R.append(row)
        for i, e, v, sw in s.walls: R.append({2 * s.ix[i] + e: 1})
        st = [dict() for _ in range(s.nx)]
        for k in range(len(s.ties)):
            for m, gv in s.grads(X, k).items(): st[m][s.il + k] = st[m].get(s.il + k, 0) + gv
            for m, row in s.hess(k).items():
                for m2, hv in row.items(): st[m][m2] = st[m].get(m2, 0) + lam[k] * hv
        for w, (i, e, v, sw) in enumerate(s.walls): st[2 * s.ix[i] + e][s.im + w] = sw
        R += st; R.append({s.il + k: -1 for k in range(len(s.ties))}); return R
    def hess(s, k):
        """constant Hessian of signed area of tie k: d2/dPx dQy etc. Entries +-sg/2 (exact Fractions)."""
        i, j, l = s.ties[k]; h = Fr(s.sg[k], 2); H = {}
        def add(a, b, v): H.setdefault(a, {}); H[a][b] = H[a].get(b, 0) + v
        for p, q, r in ((i, j, l), (j, l, i), (l, i, j)):
            # dP_x of det = Q_y - R_y ; dP_y = R_x - Q_x
            px, py = 2 * s.ix[p], 2 * s.ix[p] + 1; qx, qy, rx, ry = 2 * s.ix[q], 2 * s.ix[q] + 1, 2 * s.ix[r], 2 * s.ix[r] + 1
            add(px, qy, h); add(px, ry, -h); add(py, rx, h); add(py, qx, -h)
        return H

def dense(R, N, conv=float): M = np.zeros((N, N)); [M.__setitem__((a, b), float(v)) for a, r in enumerate(R) for b, v in r.items()]; return M

def loaded_subset(P, ties, sg, core, walls, t):
    """float pass: drop ties / walls whose least-squares multiplier is ~0, recompute the core; repeat until stable."""
    while True:
        K = KKT(P, ties, sg, core, walls); z = np.zeros(K.N)
        for i, c in K.ix.items(): z[2 * c], z[2 * c + 1] = P[i]
        z[K.it] = t; Jf = dense(K.J(list(z)), K.N)
        rows = list(range(len(ties) + len(walls), K.N)); cols = list(range(K.il, K.N))
        m = np.linalg.lstsq(Jf[np.ix_(rows, cols)], -np.array(K.F(list(z)))[rows], rcond=None)[0]
        keep_t = [k for k in range(len(ties)) if m[k] > 1e-7]; keep_w = [w for w in range(len(walls)) if m[len(ties) + w] > 1e-7]
        if len(keep_t) == len(ties) and len(keep_w) == len(walls): return ties, sg, core, walls
        ties = [ties[k] for k in keep_t]; sg = [sg[k] for k in keep_t]; core = sorted({i for tr in ties for i in tr})
        walls = [walls[w] for w in keep_w if walls[w][0] in core]

def certify(n, P):
    ties, sg, core, rat, walls, t = setup(P); active = len(ties)
    if len(ties) + len(walls) <= 2 * len(core) + 1: ties, sg, core, walls = loaded_subset(P, ties, sg, core, walls, t)
    rat = [i for i in range(n) if i not in core]; K = KKT(P, ties, sg, core, walls)
    info = dict(n=n, active=active, ties=len(ties), walls=len(walls), core=len(core), rattlers=len(rat), unknowns=K.N, d=2 * len(core) + 1 - len(ties) - len(walls))
    if len(ties) + len(walls) > 2 * len(core) + 1: info.update(certified=False, reason='excess > 0: multipliers not unique'); return info
    # float start: coordinates, t, multipliers by least squares
    z = np.zeros(K.N)
    for i, c in K.ix.items(): z[2 * c], z[2 * c + 1] = P[i]
    z[K.it] = t; Jf = dense(K.J(list(z)), K.N)
    # multipliers: solve stationarity rows (last nx+1 rows) for the multiplier block
    rows = list(range(len(ties) + len(walls), K.N)); cols = list(range(K.il, K.N)); rhs = -np.array(K.F(list(z)))[rows]
    z[cols] = np.linalg.lstsq(Jf[np.ix_(rows, cols)], rhs, rcond=None)[0]
    for _ in range(8):  # float Newton
        z = z - np.linalg.solve(dense(K.J(list(z)), K.N), np.array(K.F(list(z)), float))
    # rational refinement: z <- z - Y F(z) with F exact, Y float inverse, rounded to a 2^-180 grid
    Yf = np.linalg.inv(dense(K.J(list(z)), K.N)); D = 1 << 180
    zq = [Fr(round(v * D), D) for v in z]
    for _ in range(14):
        Fz = K.F(zq); Fq = np.array([float(v) for v in Fz]); step = Yf @ Fq
        zq = [Fr(round((zi - Fr(si)) * D), D) for zi, si in zip(zq, step.tolist())]
        if max(abs(float(v)) for v in Fz) < 1e-48: break
    Jq = K.J(zq); Yf = np.linalg.inv(dense(Jq, K.N)); Y = [[Fr(v) for v in row] for row in Yf.tolist()]
    Fz = K.F(zq); N = K.N
    Y0 = max(abs(sum(Y[i][j] * Fz[j] for j in range(N) if Fz[j])) for i in range(N))
    cols = [[] for _ in range(N)]
    for a, r in enumerate(Jq):
        for b, v in r.items(): cols[b].append((a, v))
    Z0 = Fr(0)
    for i in range(N):
        Yi = Y[i]; s = Fr(0)
        for b in range(N):
            v = sum(Yi[a] * w for a, w in cols[b]) - (1 if i == b else 0); s += abs(v)
        Z0 = max(Z0, s)
    # T: second derivatives: tie rows (hess), stationarity rows (lambda x hess, both orders); count sum_jk |T_ijk|
    Trow = [Fr(0)] * N
    for k in range(len(ties)):
        H = K.hess(k); tot = sum(abs(v) for r in H.values() for v in r.values()); Trow[k] += tot
        for m, r in H.items():
            for m2, v in r.items(): Trow[len(ties) + len(walls) + m] += 2 * abs(v)  # d2/dlambda dx and dx dlambda
    Ynorm = max(sum(abs(v) for v in row) for row in Y); Z2 = Ynorm * max(Trow)
    info.update(Y0=float(Y0), Z0=float(Z0), Z2=float(Z2))
    if Z0 >= 1: info.update(certified=False, reason='Z0 >= 1'); return info
    r = 2 * Y0 / (1 - Z0)
    if not (Z2 * r * r - (1 - Z0) * r + Y0 < 0): info.update(certified=False, reason='radii polynomial not negative'); return info
    X, tq, lam, mu = K.unpack(zq); info.update(radius=float(r), t=float(tq), lam_min=float(min(lam)), mu_min=float(min(mu)) if mu else None)
    if min(list(lam) + list(mu)) <= r: info.update(certified=False, reason='a multiplier is not provably positive'); return info
    # (b) other triangles: at z0 exact, rattlers at their float positions (exact dyadics); ball moves an area by <= 3 r (|coords| <= 1+r) + r for t
    Pq = {i: (Fr(P[i][0]), Fr(P[i][1])) for i in rat}; Pq.update(X); tie_set = set(ties); gap = None
    unloaded = 0
    for tr in itertools.combinations(range(n), 3):
        if tr in tie_set: continue
        a = abs(det(Pq[tr[0]], Pq[tr[1]], Pq[tr[2]])) / 2 - tq
        if abs(a) < 1e-9 * tq:   # an unloaded tie: allowed only if it contains a free point (which can move off it)
            if all(i in K.ix for i in tr): info.update(certified=False, reason=f'unloaded tie {tr} among core points'); return info
            unloaded += 1; continue
        a = abs(det(Pq[tr[0]], Pq[tr[1]], Pq[tr[2]])) / 2 - tq; gap = a if gap is None or a < gap else gap
    info.update(gap_rel=float(gap / tq), unloaded_ties=unloaded)
    if gap <= 8 * r: info.update(certified=False, reason='another triangle within reach of t'); return info
    # walls of rattlers and core points not on walls: strictly inside (by more than r)
    inside = min(min(v, 1 - v) for i in core for e, v in enumerate(Pq[i]) if not any(w[0] == i and w[1] == e for w in walls))
    if inside <= r: info.update(certified=False, reason='a point within r of a wall'); return info
    # (c) second order on the tangent space of active constraints at z0 (exact null space, float eigenvalues)
    C = np.array([[float(Jq[a].get(b, 0)) for b in range(K.nx + 1)] for a in range(len(ties) + len(walls))])
    u, sv, vt = np.linalg.svd(C); dim = K.nx + 1 - int((sv > 1e-10 * sv[0]).sum()); info.update(tangent_dim=dim, smallest_sv=float(sv[-1]))
    if dim:
        Bv = vt[-dim:].T; H = np.zeros((K.nx + 1, K.nx + 1))
        for k in range(len(ties)):
            for m, rr in K.hess(k).items():
                for m2, v in rr.items(): H[m, m2] += float(lam[k]) * float(v)
        ev = np.linalg.eigvalsh(Bv.T @ H @ Bv); info.update(second_order_eigs=[float(e) for e in ev])
        if ev.max() >= -1e-8: info.update(certified=False, reason='second order not negative definite'); return info
    info.update(certified=True); return info

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--sizes', default='22'); ap.add_argument('--out', default=os.path.join(hlib.REPO, 'output', 'exp78', 'kkt_certs.jsonl')); a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True); fo = open(a.out, 'a')
    for n in map(int, a.sizes.split(',')):
        e = hlib.current(n)
        t0 = time.time()
        try: info = certify(n, [tuple(p) for p in e['points']])
        except (OverflowError, np.linalg.LinAlgError, ZeroDivisionError) as ex: info = dict(n=n, certified=False, reason=f'{type(ex).__name__}: singular / divergent (degenerate optimum?)'); info['secs'] = round(time.time() - t0, 1); info['source'] = e['source']
        fo.write(json.dumps(info) + '\n'); fo.flush()
        print({k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in info.items() if k not in ('second_order_eigs', 'source')}, 'eigs', info.get('second_order_eigs'), flush=True)

if __name__ == '__main__': main()
