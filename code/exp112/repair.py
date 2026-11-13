"""Exp 112: can single-point moves that keep the minimum area t raise the number of
critical triangles to 2n-7?  (Sudermann-Merx's reply: push the n=22 rattlers onto
critical triangles.)  Greedy: for each point p (others fixed), candidate positions are
vertices of p's feasible region = intersections of the lines |area(p,a,b)| = t and the
walls; keep the one with the most ties involving p (feasible: every triangle >= t).
Repeat until no point improves.  Float search, rel tol TOL; exact check separately."""
import json, sys, itertools, numpy as np

TOL = 1e-9
import os
RAD = float(os.environ.get("RAD", "0.2"))

def areas_with(p, Q):
    # |area(p, Q[i], Q[j])| for all i<j ; p: (m,2) candidates, Q: (k,2)
    I, J = np.triu_indices(len(Q), 1)
    a, b = Q[I], Q[J]
    d = (b - a)
    # area = 0.5*|cross(b-a, p-a)|
    return 0.5 * np.abs(d[None, :, 0] * (p[:, None, 1] - a[None, :, 1]) - d[None, :, 1] * (p[:, None, 0] - a[None, :, 0])), (I, J)

def min_area(P):
    n = len(P); best = np.inf
    for i in range(n - 2):
        A, _ = areas_with(P[i:i+1], P[i+1:])
        best = min(best, A.min())
    return best

def ties(P, t):
    n = len(P); c = 0
    for i in range(n - 2):
        A, _ = areas_with(P[i:i+1], P[i+1:])
        c += int((A <= t * (1 + TOL)).sum())
    return c

def lines_for(Q, t):
    # lines L: u.x = c  with |area(p,a,b)| = t  <=>  cross(b-a, p-a) = +-2t
    I, J = np.triu_indices(len(Q), 1)
    a, b = Q[I], Q[J]; d = b - a
    u = np.stack([-d[:, 1], d[:, 0]], 1)            # cross(d, p-a) = u.(p-a)
    c0 = (u * a).sum(1)
    L = [np.concatenate([u, u]), np.concatenate([c0 + 2 * t, c0 - 2 * t])]
    W = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], float); wc = np.array([0, 1, 0, 1], float)
    return np.concatenate([L[0], W]), np.concatenate([L[1], wc])

def best_spot(P, k, t):
    Q = np.delete(P, k, 0)
    U, C = lines_for(Q, t)
    best = None
    A0, _ = areas_with(P[k:k+1], Q)
    cur = int((A0 <= t * (1 + TOL)).sum())
    m = len(U)
    for s in range(0, m, 64):
        u1 = U[s:s+64][:, None, :]; c1 = C[s:s+64][:, None]
        det = u1[..., 0] * U[None, :, 1] - u1[..., 1] * U[None, :, 0]
        ok = np.abs(det) > 1e-14
        with np.errstate(divide='ignore', invalid='ignore'):
            x = (c1 * U[None, :, 1] - u1[..., 1] * C[None, :]) / det
            y = (u1[..., 0] * C[None, :] - c1 * U[None, :, 0]) / det
        pts = np.stack([x[ok], y[ok]], 1)
        pts = pts[(pts >= -1e-15).all(1) & (pts <= 1 + 1e-15).all(1)]
        if not len(pts): continue
        pts = pts[((pts - P[k]) ** 2).sum(1) < RAD ** 2]
        if not len(pts): continue
        pts = np.unique(np.round(pts, 13), axis=0)
        for q in range(0, len(pts), 2000):
            cand = pts[q:q+2000]
            A, _ = areas_with(cand, Q)
            feas = A.min(1) >= t * (1 - TOL)
            if not feas.any(): continue
            cnt = (A[feas] <= t * (1 + TOL)).sum(1)
            j = cnt.argmax()
            if best is None or cnt[j] > best[0]:
                best = (int(cnt[j]), cand[feas][j])
    return cur, best

def repair(P, t, maxit=6):
    P = P.copy(); log = []
    for it in range(maxit):
        moved = False
        for k in range(len(P)):
            A0, _ = areas_with(P[k:k+1], np.delete(P, k, 0))
            if (A0 <= t * (1 + TOL)).sum() > 1: continue   # only rattler-like points
            cur, b = best_spot(P, k, t)
            if b and b[0] > cur:
                P[k] = b[1]; moved = True; log.append((k, cur, b[0]))
        if not moved: break
    return P, log

if __name__ == '__main__':
    src, n = sys.argv[1], sys.argv[2]
    d = json.load(open(src))
    v = d[n] if n in d else d
    P = np.array(v['points'], float)
    t = min_area(P)
    c0 = ties(P, t)
    P2, log = repair(P, t)
    t2 = min_area(P2)
    out = dict(n=int(n), t=t, ties_before=c0, ties_after=ties(P2, t), bound=2 * int(n) - 7,
               min_after_rel=(t2 - t) / t, moves=log, points_after=P2.tolist())
    print(json.dumps({k: out[k] for k in out if k != 'points_after'}))
    os.makedirs('output/exp112', exist_ok=True)
    json.dump(out, open(f'output/exp112/n{n}{os.environ.get("TAG","")}.json', 'w'), indent=1)
