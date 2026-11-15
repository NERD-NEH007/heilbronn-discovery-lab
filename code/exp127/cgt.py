"""Exp 127: cell-graph proof method ported to the triangle Omega = {x, y >= 0, x + y <= 1} (area 1/2).
Our own code: adapted from experiments/126-cellgraph/cg.py (square), which re-implements the method of
Panorias (Zenodo 24186293 v1.1) from the mathematics.  Nothing in external/ was opened or run.

Statement proved by a complete run of a family F:  no configuration in F has Phi > T  (Phi = min |area|,
raw, in Omega), T rational.  The families and the normalization lemma are in NOTES.md.

Exactness: closing steps use integer arithmetic (int64 with static bounds, or Python ints).  Floats only
propose LP weights (re-checked exactly) and choose branchings.

Coordinates: integers in units 1/Q.  Point domains:
  kind 0: rectangle R[p] = [xlo, xhi, ylo, yhi] intersected with {x + y <= Q}  (free points, fixed points,
          points on the legs x = 0 / y = 0, which have one zero width);
  kind 1: hypotenuse segment {(s, Q - s): s in [R[p,0], R[p,1]]}  (R[p,2:4] kept = [Q - xhi, Q - xlo]).
D = det(Pj - Pi, Pk - Pi) is affine in each point, so its extremes over a product of convex polygons are
attained at products of vertices (Lemma 7.1 of the paper, polygon form); we enumerate the exact integer
vertices of each domain.
"""
import itertools, math, time
from fractions import Fraction as Fr
import numpy as np
import numba as nb

N = 9
TRI = None


def configure(n):
    global N, TRI
    N = n
    TRI = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int64)


configure(9)


# ------------------------------------------------------------------------------------------------ kernels
@nb.njit(cache=True)
def build_verts(R, K, Q, VX, VY, NV):
    """exact integer vertices of every point domain (<= 5 each)."""
    for p in range(R.shape[0]):
        x0 = R[p, 0]; x1 = R[p, 1]; y0 = R[p, 2]; y1 = R[p, 3]
        if K[p] == 1:
            VX[p, 0] = x0; VY[p, 0] = Q - x0
            if x1 != x0:
                VX[p, 1] = x1; VY[p, 1] = Q - x1; NV[p] = 2
            else:
                NV[p] = 1
            continue
        if x1 + y1 <= Q:
            if x0 == x1 and y0 == y1:
                VX[p, 0] = x0; VY[p, 0] = y0; NV[p] = 1
            elif x0 == x1:
                VX[p, 0] = x0; VY[p, 0] = y0; VX[p, 1] = x0; VY[p, 1] = y1; NV[p] = 2
            elif y0 == y1:
                VX[p, 0] = x0; VY[p, 0] = y0; VX[p, 1] = x1; VY[p, 1] = y0; NV[p] = 2
            else:
                VX[p, 0] = x0; VY[p, 0] = y0; VX[p, 1] = x1; VY[p, 1] = y0
                VX[p, 2] = x1; VY[p, 2] = y1; VX[p, 3] = x0; VY[p, 3] = y1; NV[p] = 4
            continue
        # Sutherland-Hodgman clip of the rectangle by x + y <= Q (requires x0 + y0 <= Q, ensured by propagate)
        cx = (x0, x1, x1, x0); cy = (y0, y0, y1, y1)
        m = 0
        for e in range(4):
            ax = cx[e]; ay = cy[e]; bx = cx[(e + 1) % 4]; by = cy[(e + 1) % 4]
            ain = ax + ay <= Q; bin_ = bx + by <= Q
            if ain:
                VX[p, m] = ax; VY[p, m] = ay; m += 1
            if ain != bin_:
                if ay == by:          # horizontal edge
                    VX[p, m] = Q - ay; VY[p, m] = ay
                else:                 # vertical edge
                    VX[p, m] = ax; VY[p, m] = Q - ax
                m += 1
        NV[p] = m


@nb.njit(cache=True)
def tri_range_v(VX, VY, NV, i, j, k):
    lo = 1 << 62
    hi = -(1 << 62)
    for a in range(NV[i]):
        xi = VX[i, a]; yi = VY[i, a]
        for b in range(NV[j]):
            ux = VX[j, b] - xi; uy = VY[j, b] - yi
            for c in range(NV[k]):
                v = ux * (VY[k, c] - yi) - uy * (VX[k, c] - xi)
                if v < lo: lo = v
                if v > hi: hi = v
    return lo, hi


@nb.njit(cache=True)
def all_ranges(R, K, Q, TRI):
    n = R.shape[0]
    VX = np.empty((n, 5), np.int64); VY = np.empty((n, 5), np.int64); NV = np.empty(n, np.int64)
    build_verts(R, K, Q, VX, VY, NV)
    m = TRI.shape[0]
    LO = np.empty(m, np.int64); HI = np.empty(m, np.int64)
    for t in range(m):
        l, h = tri_range_v(VX, VY, NV, TRI[t, 0], TRI[t, 1], TRI[t, 2])
        LO[t] = l; HI[t] = h
    return LO, HI


@nb.njit(cache=True)
def range3(R, K, Q):
    VX = np.empty((3, 5), np.int64); VY = np.empty((3, 5), np.int64); NV = np.empty(3, np.int64)
    build_verts(R, K, Q, VX, VY, NV)
    return tri_range_v(VX, VY, NV, 0, 1, 2)


@nb.njit(cache=True)
def propagate(R, K, ORD, SUM, Q):
    """ORD rows (p1, c1, p2, c2, g): coord(p1,c1) + g <= coord(p2,c2).  SUM rows (p1, c1, p2, c2):
    coord(p1,c1) + coord(p2,c2) <= Q.  Plus x + y <= Q for kind 0, y = Q - x for kind 1.  Only shrinks to
    necessary values (all integer); returns False iff some domain is empty."""
    changed = True
    it = 0
    while changed and it < 50:
        changed = False
        it += 1
        for t in range(ORD.shape[0]):
            p1 = ORD[t, 0]; c1 = ORD[t, 1]; p2 = ORD[t, 2]; c2 = ORD[t, 3]; g = ORD[t, 4]
            h = R[p2, 2 * c2 + 1] - g
            if h < R[p1, 2 * c1 + 1]:
                R[p1, 2 * c1 + 1] = h; changed = True
            l = R[p1, 2 * c1] + g
            if l > R[p2, 2 * c2]:
                R[p2, 2 * c2] = l; changed = True
        for t in range(SUM.shape[0]):
            p1 = SUM[t, 0]; c1 = SUM[t, 1]; p2 = SUM[t, 2]; c2 = SUM[t, 3]
            h = Q - R[p2, 2 * c2]
            if h < R[p1, 2 * c1 + 1]:
                R[p1, 2 * c1 + 1] = h; changed = True
            h = Q - R[p1, 2 * c1]
            if h < R[p2, 2 * c2 + 1]:
                R[p2, 2 * c2 + 1] = h; changed = True
        for p in range(R.shape[0]):
            if K[p] == 1:
                R[p, 2] = Q - R[p, 1]; R[p, 3] = Q - R[p, 0]
            else:
                h = Q - R[p, 2]
                if h < R[p, 1]:
                    R[p, 1] = h; changed = True
                h = Q - R[p, 0]
                if h < R[p, 3]:
                    R[p, 3] = h; changed = True
            if R[p, 0] > R[p, 1] or R[p, 2] > R[p, 3]:
                return False
    for p in range(R.shape[0]):
        if R[p, 0] > R[p, 1] or R[p, 2] > R[p, 3]:
            return False
    return True


@nb.njit(cache=True)
def choose_split(R, SPL, LO, HI, TRI, rq):
    """branching heuristic (not part of the proof)."""
    n = R.shape[0]
    score = np.zeros(2 * n)
    for t in range(TRI.shape[0]):
        lo = LO[t]; hi = HI[t]
        if lo >= -rq:
            ex = hi - rq
        elif hi <= rq:
            ex = -rq - lo
        else:
            ex = min(hi - rq, -rq - lo) + (hi - lo)
        if ex < 1: ex = 1
        fe = float(ex)
        w8 = 1.0 / (fe * fe)
        i = TRI[t, 0]; j = TRI[t, 1]; k = TRI[t, 2]
        for (a, b, c) in ((i, j, k), (j, k, i), (k, i, j)):
            gx = max(abs(R[b, 3] - R[c, 2]), abs(R[b, 2] - R[c, 3]))
            gy = max(abs(R[c, 1] - R[b, 0]), abs(R[c, 0] - R[b, 1]))
            score[2 * a] += (R[a, 1] - R[a, 0]) * (gx + gy) * w8
            score[2 * a + 1] += (R[a, 3] - R[a, 2]) * gy * w8
    best = -1; bs = 0.0
    for v in range(2 * n):
        p = v // 2; c = v % 2
        if SPL[p, c] and R[p, 2 * c + 1] - R[p, 2 * c] >= 2 and score[v] > bs:
            bs = score[v]; best = v
    return best


@nb.njit(cache=True)
def cell_excluded(S, SK, cells, CB, Q, rq):
    """cell removed if some two supports a<b give |D(S_a, S_b, P)| <= rho for all points of the domains."""
    k = S.shape[0]
    out = np.zeros(cells.shape[0], np.bool_)
    R = np.empty((3, 4), np.int64); KK = np.zeros(3, np.int64)
    for u in range(cells.shape[0]):
        for z in range(4): R[2, z] = CB[u, z]
        done = False
        for a in range(k):
            if done: break
            for b in range(a + 1, k):
                for z in range(4):
                    R[0, z] = S[a, z]; R[1, z] = S[b, z]
                KK[0] = SK[a]; KK[1] = SK[b]
                lo, hi = range3(R, KK, Q)
                if lo >= -rq and hi <= rq:
                    done = True; break
        out[u] = done
    return out


@nb.njit(cache=True)
def edge_matrix(S, SK, CB, Q, rq):
    m = CB.shape[0]; k = S.shape[0]
    A = np.zeros((m, m), np.bool_)
    R = np.empty((3, 4), np.int64); KK = np.zeros(3, np.int64)
    for u in range(m):
        for v in range(u + 1, m):
            for z in range(4):
                R[1, z] = CB[u, z]; R[2, z] = CB[v, z]
            ok = True
            for a in range(k):
                for z in range(4):
                    R[0, z] = S[a, z]
                KK[0] = SK[a]
                lo, hi = range3(R, KK, Q)
                if lo >= -rq and hi <= rq:
                    ok = False; break
            A[u, v] = ok; A[v, u] = ok
    return A


@nb.njit(cache=True)
def three_cell_blocked(CB, sel, ns, v, Q, rq):
    R = np.empty((3, 4), np.int64); KK = np.zeros(3, np.int64)
    for z in range(4):
        R[2, z] = CB[v, z]
    for a in range(ns):
        for z in range(4):
            R[0, z] = CB[sel[a], z]
        for b in range(a + 1, ns):
            for z in range(4):
                R[1, z] = CB[sel[b], z]
            lo, hi = range3(R, KK, Q)
            if lo >= -rq and hi <= rq:
                return True
    return False


# ------------------------------------------------------------------------------------------------ families
class Family:
    def __init__(self, name, sup, ords, sums, Q):
        """sup: list of (kind, xlo, xhi, ylo, yhi, splx, sply) in units of Q (Fractions or ints * Q)."""
        self.name = name; self.k = len(sup); self.m = N - self.k; self.Q = Q
        self.S0 = np.array([[s[1], s[2], s[3], s[4]] for s in sup], np.int64).reshape(-1, 4)
        self.SK = np.array([s[0] for s in sup], np.int64)
        self.SSPL = np.array([[s[5], s[6]] for s in sup], np.bool_).reshape(-1, 2)
        self.ORD = np.array(ords, dtype=np.int64).reshape(-1, 5)
        self.SUM = np.array(sums, dtype=np.int64).reshape(-1, 4)
        self.K = np.concatenate([self.SK, np.zeros(self.m, np.int64)])
        self.SPL = np.concatenate([self.SSPL, np.ones((self.m, 2), np.bool_)])
        self.nparam = int(self.SSPL.sum())

    def root(self):
        return self.S0.copy()


FAMILIES = ['MMM', 'MMS', 'MSS0', 'MSS1', 'SSS0', 'SSS1', 'SSS2', 'SSS3']


def make_family(name, Q, l0u, sym=True):
    """support points (see NOTES.md).  B: y = 0 (param x), L: x = 0 (param y), H: x + y = 1 (param x).
    Contacts in the 'near' half have param in [0, Q/2], 'far' half [Q/2, Q].  l0u = floor(l0 Q)."""
    h = Q // 2
    assert 2 * h == Q
    fix = lambda x, y: (0, x, x, y, y, False, False)
    B = lambda lo, hi: (0, lo, hi, 0, 0, True, False)
    L = lambda lo, hi: (0, 0, 0, lo, hi, False, True)
    H = lambda lo, hi: (1, lo, hi, Q - hi, Q - lo, True, False)
    mB, mL, mH, O, X, Y = fix(h, 0), fix(0, h), fix(h, h), fix(0, 0), fix(Q, 0), fix(0, Q)
    ords, sums = [], []
    if name == 'MMM':
        sup = [mB, mL, mH]
    elif name == 'MMS':                       # M on L and H; S on B: B1 (0), B2 (1)
        sup = [B(0, h), B(h, Q), mL, mH]
        ords = [(0, 0, 1, 0, l0u)]
        if sym: sums = [(0, 0, 1, 0)]         # reflection O<->X:  B1.x <= 1 - B2.x
    elif name == 'MSS0':                      # M on H; S on B (B1, B2) and L (L1, L2)
        sup = [B(0, h), B(h, Q), L(0, h), L(h, Q), mH]
        ords = [(0, 0, 1, 0, l0u), (2, 1, 3, 1, l0u)]
        if sym: ords.append((0, 0, 2, 1, 0))  # mirror x<->y:  B1.x <= L1.y
    elif name == 'MSS1':                      # M on H; O occupied; B2, L2
        sup = [O, B(h, Q), L(h, Q), mH]
        if sym: ords = [(2, 1, 1, 0, 0)]      # mirror:  L2.y <= B2.x
    elif name == 'SSS0':                      # B1 B2 L1 L2 H1 (x in [0,1/2], near Y) H2 (near X)
        sup = [B(0, h), B(h, Q), L(0, h), L(h, Q), H(0, h), H(h, Q)]
        ords = [(0, 0, 1, 0, l0u), (2, 1, 3, 1, l0u), (4, 0, 5, 0, l0u)]
        if sym:                               # S3: flag (B, O) carries the least vertex distance
            ords += [(0, 0, 2, 1, 0), (0, 0, 4, 0, 0)]
            sums = [(0, 0, 1, 0), (0, 0, 3, 1), (0, 0, 5, 0)]
    elif name == 'SSS1':                      # O occupied; B2, L2, H1, H2
        sup = [O, B(h, Q), L(h, Q), H(0, h), H(h, Q)]
        ords = [(3, 0, 4, 0, l0u)]
        if sym: ords.append((2, 1, 1, 0, 0))  # mirror:  L2.y <= B2.x
    elif name == 'SSS2':                      # O, X occupied; L2, H1
        sup = [O, X, L(h, Q), H(0, h)]
        if sym: sums = [(3, 0, 2, 1)]         # reflection O<->X:  H1.x <= 1 - L2.y
    elif name == 'SSS3':
        sup = [O, X, Y]
    else:
        raise ValueError(name)
    return Family(name, sup, ords, sums, Q)


# ------------------------------------------------------------------------------------------------ P2
TERMS = [(+1, 0, 1), (-1, 0, 2), (-1, 1, 0), (+1, 1, 2), (+1, 2, 0), (-1, 2, 1)]   # D = sum sg * x_a * y_b


def coord_model(R, K, Q):
    """coordinate v = 2p + c equals (C[v] + W[v] * s[var[v]]) / 2 with s in [-1, 1]^(2n) (each var used by
    one point only).  kind 1: y = Q - x."""
    n = R.shape[0]
    C = np.empty(2 * n, np.int64); W = np.empty(2 * n, np.int64); V = np.arange(2 * n, dtype=np.int64)
    C[0::2] = R[:, 0] + R[:, 1]; W[0::2] = R[:, 1] - R[:, 0]
    C[1::2] = R[:, 2] + R[:, 3]; W[1::2] = R[:, 3] - R[:, 2]
    for p in range(n):
        if K[p] == 1:
            C[2 * p + 1] = 2 * Q - C[2 * p]; W[2 * p + 1] = -W[2 * p]; V[2 * p + 1] = 2 * p
    return C, W, V


def p2_try(R, K, LO, HI, rq, rho, Q, TRI, lp_tol=-1e-9):
    """positive integer combination certificate.  For every triple i with determined sign sigma_i and every
    point in the box:  4 sigma_i D_i <= K_i + A_i . s  (bilinear terms of distinct variables bounded by
    |W W|).  If sum lam_i (K_i + |sum lam A|) <= 4 rho Q^2 sum lam (checked in integers) no point of the box has
    all sigma_i D_i > rho Q^2.  Returns (closed, branching hint)."""
    from scipy.optimize import linprog
    detmask = (LO >= -rq) | (HI <= rq)
    idx = np.nonzero(detmask)[0]
    if len(idx) < 2:
        return False, -1
    sig = np.where(LO[idx] >= -rq, 1, -1).astype(np.int64)
    n = R.shape[0]
    C, W, V = coord_model(R, K, Q)
    m = len(idx)
    Kc = np.zeros(m, np.int64); A = np.zeros((m, 2 * n), np.int64)
    tri = TRI[idx]
    ar = np.arange(m)
    for sg, a, b in TERMS:
        xa = 2 * tri[:, a]; yb = 2 * tri[:, b] + 1
        s = sig * sg
        Kc += s * C[xa] * C[yb] + np.abs(W[xa] * W[yb])
        np.add.at(A, (ar, V[xa]), s * W[xa] * C[yb])
        np.add.at(A, (ar, V[yb]), s * C[xa] * W[yb])
    R4 = float(4 * rho * Q * Q)
    sc = 1.0 / (Q * Q)
    nv = 2 * n
    used = np.zeros(nv, bool); used[V[W != 0]] = True
    cols = [v for v in range(nv) if used[v]]
    nc = len(cols)
    c = np.concatenate([(Kc - R4) * sc, np.ones(nc)])
    Af = A[:, cols].T.astype(float) * sc
    Aub = np.block([[Af, -np.eye(nc)], [-Af, -np.eye(nc)]])
    bub = np.zeros(2 * nc)
    Aeq = np.concatenate([np.ones(m), np.zeros(nc)])[None, :]
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[1.0], bounds=[(0, None)] * (m + nc), method='highs')
    if res.status != 0:
        return False, -1
    lam = res.x[:m]
    if res.fun > lp_tol:
        contrib = np.zeros(nv)
        contrib[cols] = np.abs(lam @ A[:, cols].astype(float)) * sc
        for sg, a, b in TERMS:
            xa = 2 * tri[:, a]; yb = 2 * tri[:, b] + 1
            q = np.abs(W[xa] * W[yb]).astype(float) * sc * lam / 2
            np.add.at(contrib, V[xa], q); np.add.at(contrib, V[yb], q)
        wv = np.zeros(nv, np.int64)
        for v in range(nv):
            wv[V[v]] = max(wv[V[v]], abs(int(W[v])))
        contrib[wv < 2] = -1
        return False, int(np.argmax(contrib)) if contrib.max() > 0 else -1
    lam_i = [int(round(x * 2 ** 30)) for x in lam]
    if sum(lam_i) <= 0 or min(lam_i) < 0:
        return False, -1
    Wsum = sum(lam_i)
    Ctot = sum(l * int(Kc[i]) for i, l in enumerate(lam_i) if l)
    Atot = 0
    for v in cols:
        Atot += abs(sum(l * int(A[i, v]) for i, l in enumerate(lam_i) if l))
    return (Ctot + Atot) * rho.denominator <= 4 * rho.numerator * Q * Q * Wsum, -1


# ------------------------------------------------------------------------------------------------ cover
_V3 = ((0, 0), (1, 0), (0, 1))


def s3_maps():
    """the 6 affine symmetries g(p) = M p + c of Omega (vertex permutations), integer M, c."""
    out = []
    for s in itertools.permutations(range(3)):
        o, x, y = (_V3[k] for k in s)
        M = ((x[0] - o[0], y[0] - o[0]), (x[1] - o[1], y[1] - o[1]))
        out.append((M, o))
    return out


def _inv(g):
    (a, b), (c, d) = g[0]; det = a * d - b * c
    assert det in (1, -1)
    Mi = ((d * det, -b * det), (-c * det, a * det))
    ci = (-(Mi[0][0] * g[1][0] + Mi[0][1] * g[1][1]), -(Mi[1][0] * g[1][0] + Mi[1][1] * g[1][1]))
    return Mi, ci


def make_cover(xstar, radius, Q):
    """xstar: exact certificate centre (Fractions), radius: Fraction R (with R + centre error < basin radius).
    A box is closed if for some S3 map g and some injective labelling j: every vertex q of every point domain
    satisfies || g^{-1}(q) - xstar_j(i) ||_inf <= R (exact).  By convexity the whole box then maps into the
    sup-ball of radius R around a relabelling of xstar, in the certificate's own frame."""
    invs = [_inv(g) for g in s3_maps()]
    X = np.array([[float(x), float(y)] for x, y in xstar])
    RQ = radius * Q
    rf = float(radius)

    def covered(R, K, want=False):
        n = R.shape[0]
        VX = np.empty((n, 5), np.int64); VY = np.empty((n, 5), np.int64); NV = np.empty(n, np.int64)
        build_verts(R, K, Q, VX, VY, NV)
        cx = (R[:, 0] + R[:, 1]) / (2 * Q); cy = (R[:, 2] + R[:, 3]) / (2 * Q)
        for Mi, ci in invs:
            ux = Mi[0][0] * cx + Mi[0][1] * cy + ci[0]; uy = Mi[1][0] * cx + Mi[1][1] * cy + ci[1]
            d = np.maximum(np.abs(ux[:, None] - X[None, :, 0]), np.abs(uy[:, None] - X[None, :, 1]))
            j = d.argmin(1)
            if len(set(j.tolist())) < n or d[np.arange(n), j].max() > rf:
                continue
            ok = True
            for i in range(n):
                px, py = xstar[j[i]]
                for a in range(NV[i]):
                    qx = int(VX[i, a]); qy = int(VY[i, a])
                    gx = Mi[0][0] * qx + Mi[0][1] * qy + ci[0] * Q
                    gy = Mi[1][0] * qx + Mi[1][1] * qy + ci[1] * Q
                    if abs(gx - px * Q) > RQ or abs(gy - py * Q) > RQ:
                        ok = False; break
                if not ok: break
            if ok:
                return ((Mi, ci), j.tolist()) if want else True
        return False
    return covered


# ------------------------------------------------------------------------------------------------ local tree
def local_solve(R0, fam, rq, rho, Q, budget, stats, lp_w, covered=None, lp_d=0):
    stack = [(R0.copy(), 0)]
    nodes = 0
    K = fam.K; SPL = fam.SPL
    while stack:
        R, dep = stack.pop()
        nodes += 1
        if nodes > budget:
            stats['fail_nodes'] += nodes
            return False
        if not propagate(R, K, fam.ORD, fam.SUM, Q):
            stats['E'] += 1; continue
        LO, HI = all_ranges(R, K, Q, TRI)
        if np.any((LO >= -rq) & (HI <= rq)):
            stats['T'] += 1; continue
        wmax = int(max((R[:, 1] - R[:, 0]).max(), (R[:, 3] - R[:, 2]).max()))
        v = -1
        if wmax <= lp_w and dep >= lp_d:
            stats['lp_calls'] += 1
            ok, v = p2_try(R, K, LO, HI, rq, rho, Q, TRI)
            if ok:
                stats['P2'] += 1; continue
            if covered is not None and covered(R, K):
                stats['cover'] += 1; continue
        if v >= 0 and not SPL[v // 2, v % 2]:
            v = -1
        if v < 0:
            v = choose_split(R, SPL, LO, HI, TRI, rq)
        if v < 0:
            stats['fail_nodes'] += nodes
            return False
        p, c = divmod(v, 2)
        lo, hi = R[p, 2 * c], R[p, 2 * c + 1]
        mid = (lo + hi) // 2
        R1 = R.copy(); R1[p, 2 * c + 1] = mid
        R2 = R.copy(); R2[p, 2 * c] = mid
        stack.append((R1, dep + 1)); stack.append((R2, dep + 1))
        stats['splits'] += 1
    stats['local_nodes'] += nodes
    return True


# ------------------------------------------------------------------------------------------------ cliques
def k_cliques_iter(adj, k, CB, Q, rq, stats, first=None):
    """generator of all k-cliques (sorted index tuples) with 3-cell pruning; optionally only those whose
    smallest vertex is `first`."""
    m = adj.shape[0]
    nb_ = [0] * m
    for u in range(m):
        row = np.nonzero(adj[u, u + 1:])[0] + u + 1
        b = 0
        for v in row.tolist():
            b |= 1 << v
        nb_[u] = b
    selarr = np.zeros(max(k, 1), np.int64)
    sel = []

    def rec(cand, need):
        if need == 0:
            yield tuple(sel); return
        while cand:
            if bin(cand).count('1') < need:
                return
            low = cand & -cand
            v = low.bit_length() - 1
            cand ^= low
            if len(sel) >= 2:
                for z, u in enumerate(sel): selarr[z] = u
                if three_cell_blocked(CB, selarr, len(sel), v, Q, rq):
                    stats['tri3'] += 1
                    continue
            sel.append(v)
            yield from rec(cand & nb_[v], need - 1)
            sel.pop()
    if k == 0:
        yield (); return
    if first is None:
        yield from rec((1 << m) - 1, k)
    else:
        sel.append(first)
        yield from rec(nb_[first], k - 1)
        sel.pop()


_TRI_SUB = {}


def p2_cell_prune(S, fam, CB, adj, Q, rq, rho, cap, stats):
    """optional P2 cell exclusion (flag cfg['cell_p2'] = cap): for up to `cap` highest-degree cells, an exact
    positive-combination certificate (p2_try) over the k supports in their boxes plus one point anywhere in the
    cell (k + 1 points, all their triples) proves that no configuration with Phi > T has a free point in that
    cell; such cells are removed.  Sound by the P2 lemma applied to the sub-configuration (a subset of the
    triples of any full configuration)."""
    k = fam.k
    if cap <= 0 or k < 2 or len(CB) == 0:
        return CB, adj
    tri = _TRI_SUB.get(k + 1)
    if tri is None:
        tri = _TRI_SUB[k + 1] = np.array(list(itertools.combinations(range(k + 1), 3)), np.int64)
    KK = np.concatenate([fam.SK, np.zeros(1, np.int64)])
    deg = adj.sum(1)
    order = np.argsort(-deg, kind='stable')[:cap]
    drop = np.zeros(len(CB), bool)
    for u in order.tolist():
        R = np.vstack([S, CB[u:u + 1]]).astype(np.int64)
        if not propagate(R, KK, fam.ORD, fam.SUM, Q):
            drop[u] = True; continue
        LO, HI = all_ranges(R, KK, Q, tri)
        if np.any((LO >= -rq) & (HI <= rq)) or p2_try(R, KK, LO, HI, rq, rho, Q, tri)[0]:
            drop[u] = True
        stats['cell_p2_tests'] = stats.get('cell_p2_tests', 0) + 1
    stats['cell_p2_excl'] = stats.get('cell_p2_excl', 0) + int(drop.sum())
    keep = ~drop
    return CB[keep], adj[np.ix_(keep, keep)]


def cell_graph(S, fam, G, Q, rq, stats, rho=None, cell_p2=0):
    cs = Q // G
    cells = np.array([cx * G + cy for cx in range(G) for cy in range(G) if cx + cy <= G - 1], np.int64)
    CB = np.empty((len(cells), 4), np.int64)
    CB[:, 0] = (cells // G) * cs; CB[:, 1] = CB[:, 0] + cs; CB[:, 2] = (cells % G) * cs; CB[:, 3] = CB[:, 2] + cs
    excl = cell_excluded(S, fam.SK, cells, CB, Q, rq)
    stats['cells_excl'] += int(excl.sum())
    CB = CB[~excl]
    adj = edge_matrix(S, fam.SK, CB, Q, rq)
    if cell_p2:
        CB, adj = p2_cell_prune(S, fam, CB, adj, Q, rq, rho, cell_p2, stats)
    return CB, adj


def solve_outer(S, fam, G, rq, rho, Q, cfg, stats, covered=None, first=None, graph=None):
    S = S.copy()
    if not propagate(S, fam.SK, fam.ORD, fam.SUM, Q):
        stats['outer_E'] += 1; return 'closed'
    k = fam.k
    for t in itertools.combinations(range(k), 3):
        lo, hi = range3(S[list(t)].copy(), fam.SK[list(t)].copy(), Q)
        if lo >= -rq and hi <= rq:
            stats['outer_T'] += 1; return 'closed'
    if fam.nparam and int(((S[:, 1::2] - S[:, 0::2]) * fam.SSPL).max()) > cfg['max_sw']:
        return 'split'
    CB, adj = graph if graph is not None else cell_graph(S, fam, G, Q, rq, stats, rho, cfg.get('cell_p2', 0))
    if len(CB) < fam.m:
        stats['outer_deficit'] += 1; return 'closed'
    cnt = 0
    for c in k_cliques_iter(adj, fam.m, CB, Q, rq, stats, first):
        cnt += 1
        if fam.nparam and cnt > cfg['max_cliques']:
            stats['cliques'] += cnt
            return 'split'
        R = np.zeros((N, 4), np.int64)
        R[:k] = S
        for i, u in enumerate(c):
            R[k + i] = CB[u]
        if not local_solve(R, fam, rq, rho, Q, cfg['local_budget'], stats, cfg['lp_w'], covered, cfg.get('lp_d', 0)):
            stats['local_fail'] += 1
            if fam.nparam:
                stats['cliques'] += cnt
                return 'split'
            stats['unresolved_cliques'] += 1
    stats['cliques'] += cnt
    return 'closed'


STAT_KEYS = ('outer_leaves', 'outer_splits', 'outer_E', 'outer_T', 'outer_deficit', 'cells_excl', 'cliques',
             'local_nodes', 'T', 'E', 'P2', 'lp_calls', 'cover', 'splits', 'fail_nodes', 'local_fail', 'unresolved',
             'tri3', 'unresolved_cliques', 'cell_p2_tests', 'cell_p2_excl')


def new_stats():
    return {k: 0 for k in STAT_KEYS}


def outer_tree(fam, G, rq, rho, Q, cfg, stats, S0=None, covered=None, min_w=None, t_end=None):
    stack = [fam.root() if S0 is None else S0]
    unresolved = []
    leftover = stats.setdefault('_leftover', [])
    min_w = min_w if min_w is not None else Q // 4096
    while stack:
        S = stack.pop()
        if t_end is not None and time.time() > t_end:
            leftover.append(S); leftover.extend(stack); break
        r = solve_outer(S, fam, G, rq, rho, Q, cfg, stats, covered)
        if r == 'closed':
            stats['outer_leaves'] += 1; continue
        w = (S[:, 1::2] - S[:, 0::2]) * fam.SSPL
        p, c = np.unravel_index(int(np.argmax(w)), w.shape)
        if w[p, c] <= min_w:
            unresolved.append(S); stats['unresolved'] += 1; continue
        lo, hi = S[p, 2 * c], S[p, 2 * c + 1]; mid = (lo + hi) // 2
        S1 = S.copy(); S1[p, 2 * c + 1] = mid
        S2 = S.copy(); S2[p, 2 * c] = mid
        stack += [S1, S2]; stats['outer_splits'] += 1
    return unresolved
