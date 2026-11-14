"""exp 113: rigorous branch-and-bound for the Heilbronn problem in the TRIANGLE (port of exp 110 bb2.py / exp 103 bb.py).

Frame: Omega = {x >= 0, y >= 0, x + y <= 1} (area 1/2); f(P) = min over triples of |area|.  Site value = 2 f.
Claim proved for given n and threshold T (dyadic float):
    every configuration of n points in Omega has f <= T,
    except configurations that, after a symmetry of Omega (S3) and a relabelling, lie in the certified basin ball
    of the exp 113 local certificate (exact mode only).
Self-contained copy (exp 103 / 110 are NOT imported, and not modified).  Changes vs bb2 (see NOTES.md):
  * normalised domain (all reductions proved in NOTES.md): x sorted; x_0 = 0; y_b = 0; x_c + y_c = 1 (one root per
    (b, c), n^2 roots); x_i + y_i <= 1; A >= B >= C with A = sum x, B = sum y, C = sum (1 - x - y).
    Dropped: mean(x) <= 1/2, x_{n-1} = 1, y-mirror, diagonal (square only).
  * P0: exact int64 propagation of sorting, x_i + y_i <= 1, x_c + y_c >= 1 and A >= B >= C (outward rounding).
  * P1: unchanged (corner ranges over the box bound 2A on Omega intersect box).
  * P2: fixed LP + rows  x_i + y_i <= 1 (always on), -x_i - y_i <= -1 (on only for i = c, multiplier zeroed otherwise),
    -sum x + sum y <= 0, -sum x - 2 sum y <= -n.  Same exact integer certificate; overflow bound recomputed from column
    sums of |coefficients| (McCormick entries bounded by D) and asserted.
  * cover: certified basin ball of local_certs_tri.json, tested against all 6 S3 images by mapping the 4 corners of every
    point rectangle back with g^-1 exactly (g is affine but not a sup-norm isometry, so the test is done in the frame
    of the certificate).
"""
import os, sys, math, itertools, json
HERE = os.path.dirname(os.path.abspath(__file__))
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
from fractions import Fraction as Fr
import numpy as np
import numba as nb
import highspy

LOGD = 24
D = 1 << LOGD
INF = highspy.kHighsInf
CERT = os.path.normpath(os.path.join(HERE, '..', 'cert', 'local_certs_tri.json'))
POOL = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'configurations', 'triangle'))
PERMS = list(itertools.permutations(range(3)))          # S3 acting on barycentrics (x, y, 1 - x - y)


def s3_apply(perm, p):
    """g_perm(p): barycentric b = (x, y, 1-x-y), b'_k = b_{perm[k]}, return (b'_0, b'_1).  Exact for Fractions."""
    b = (p[0], p[1], 1 - p[0] - p[1]); return (b[perm[0]], b[perm[1]])


def s3_inverse_apply(perm, p):
    """g_perm^{-1}(p): b'_k = b_{perm[k]}  ->  b_{perm[k]} = b'_k."""
    bp = (p[0], p[1], 1 - p[0] - p[1]); b = [None] * 3
    for k in range(3): b[perm[k]] = bp[k]
    return (b[0], b[1])


def s3_inverse_linear(perm):
    """Exact linear part M and offset of g_perm^{-1}(p) = M p + v (from evaluations at 0, e1, e2)."""
    z = s3_inverse_apply(perm, (Fr(0), Fr(0))); e1 = s3_inverse_apply(perm, (Fr(1), Fr(0))); e2 = s3_inverse_apply(perm, (Fr(0), Fr(1)))
    M = ((e1[0] - z[0], e2[0] - z[0]), (e1[1] - z[1], e2[1] - z[1])); return M, z


# Sup-norm operator norms of g^{-1} (NOTES.md, "cover"): 1 for the identity and the diagonal swap, 2 for the four maps
# that move the hypotenuse.  Recorded and asserted; the cover test does NOT rely on them (it maps the 4 corners of every
# point rectangle back with g^{-1} exactly and measures the distance in the frame of the certificate).
OPNORM = {g: max(abs(r[0]) + abs(r[1]) for r in s3_inverse_linear(g)[0]) for g in PERMS}
assert sorted(OPNORM.values()) == [1, 1, 2, 2, 2, 2], OPNORM
for _g in PERMS:                                           # g^{-1} really inverts g (exact, on 3 test points)
    for _p in ((Fr(1, 3), Fr(1, 5)), (Fr(0), Fr(1)), (Fr(2, 7), Fr(0))):
        assert s3_inverse_apply(_g, s3_apply(_g, _p)) == _p


# ------------------------------------------------------------------------------------------------ compiled kernels
@nb.njit(cache=True)
def k_propagate(lo, hi, n, c, Dg):
    """P0 in exact int64 (grid units).  Domain: x sorted; 0 <= x, y; x_i + y_i <= D; x_c + y_c = D;
    sum y <= sum x (A >= B); sum x + 2 sum y >= n D (B >= C).  Returns False iff the box provably contains no domain
    point.  Only shrinks the box, never removes a domain point: every new bound is an integer that is implied by the
    constraints for every REAL point of the box (the single division by 2 is rounded down for a lower bound)."""
    for it in range(60):
        ch = False
        for i in range(n - 1):
            if lo[i + 1] < lo[i]: lo[i + 1] = lo[i]; ch = True
        for i in range(n - 2, -1, -1):
            if hi[i] > hi[i + 1]: hi[i] = hi[i + 1]; ch = True
        # x_i + y_i <= D
        for i in range(n):
            u = Dg - lo[n + i]
            if hi[i] > u: hi[i] = u; ch = True
            u = Dg - lo[i]
            if hi[n + i] > u: hi[n + i] = u; ch = True
        # x_c + y_c >= D
        v = Dg - hi[n + c]
        if lo[c] < v: lo[c] = v; ch = True
        v = Dg - hi[c]
        if lo[n + c] < v: lo[n + c] = v; ch = True
        for i in range(2 * n):
            if lo[i] > hi[i]: return False
        # A >= B:  sum y <= sum x
        shx = 0; sly = 0
        for i in range(n): shx += hi[i]; sly += lo[n + i]
        if sly > shx: return False
        for i in range(n):
            u = shx - (sly - lo[n + i])
            if hi[n + i] > u: hi[n + i] = u; ch = True
            v = sly - (shx - hi[i])
            if lo[i] < v: lo[i] = v; ch = True
        for i in range(2 * n):
            if lo[i] > hi[i]: return False
        # B >= C:  sum x + 2 sum y >= n D
        shx = 0; shy = 0
        for i in range(n): shx += hi[i]; shy += hi[n + i]
        if shx + 2 * shy < n * Dg: return False
        for i in range(n):
            v = n * Dg - 2 * shy - (shx - hi[i])
            if lo[i] < v: lo[i] = v; ch = True
            v2 = n * Dg - shx - 2 * (shy - hi[n + i])       # 2 y_i >= v2  ->  y_i >= floor(v2 / 2) (outward)
            v = v2 // 2
            if lo[n + i] < v: lo[n + i] = v; ch = True
        for i in range(2 * n):
            if lo[i] > hi[i]: return False
        if not ch: break
    for i in range(2 * n):
        if lo[i] > hi[i]: return False
    return True


@nb.njit(cache=True)
def k_ranges(lo, hi, tri, n, T2i):
    """P1 (unchanged from bb2): exact int64 corner range of 2A*D^2 per triple.  Returns (dead, sig)."""
    m = tri.shape[0]; sig = np.zeros(m, np.int64)
    for t in range(m):
        i = tri[t, 0]; j = tri[t, 1]; k = tri[t, 2]
        mn = 1 << 62; mx = -(1 << 62)
        for ci in range(4):
            xi = lo[i] if ci < 2 else hi[i]; yi = lo[n + i] if (ci & 1) == 0 else hi[n + i]
            for cj in range(4):
                xj = lo[j] if cj < 2 else hi[j]; yj = lo[n + j] if (cj & 1) == 0 else hi[n + j]
                for ck in range(4):
                    xk = lo[k] if ck < 2 else hi[k]; yk = lo[n + k] if (ck & 1) == 0 else hi[n + k]
                    v = (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)
                    if v < mn: mn = v
                    if v > mx: mx = v
        if max(mx, -mn) <= T2i: return True, sig
        if mn >= -T2i: sig[t] = 1
        elif mx <= T2i: sig[t] = -1
    return False, sig


@nb.njit(cache=True)
def k_choose(lo, hi, sig, tri, n):
    cnt = np.zeros(n, np.int64)
    for t in range(tri.shape[0]):
        if sig[t] == 0:
            cnt[tri[t, 0]] += 1; cnt[tri[t, 1]] += 1; cnt[tri[t, 2]] += 1
    best = -1.0; kb = -1
    for k in range(2 * n):
        w = hi[k] - lo[k]
        if w < 2: continue
        sc = float(w) * (1.0 + cnt[k % n])
        if sc > best: best = sc; kb = k
    return kb


@nb.njit(cache=True)
def k_fill(lo, hi, n, pa, pb, vals, H, mcc_y, mcc_x, mcc_r, clo, chi):
    P = pa.shape[0]
    for p in range(P):
        XL = lo[pa[p]]; XH = hi[pa[p]]; YL = lo[n + pb[p]]; YH = hi[n + pb[p]]
        vals[mcc_y[0, p]] = XL; vals[mcc_x[0, p]] = YL; H[mcc_r[0, p]] = XL * YL
        vals[mcc_y[1, p]] = XH; vals[mcc_x[1, p]] = YH; H[mcc_r[1, p]] = XH * YH
        vals[mcc_y[2, p]] = -XH; vals[mcc_x[2, p]] = -YL; H[mcc_r[2, p]] = -XH * YL
        vals[mcc_y[3, p]] = -XL; vals[mcc_x[3, p]] = -YH; H[mcc_r[3, p]] = -XL * YH
        a1 = XL * YL; a2 = XL * YH; a3 = XH * YL; a4 = XH * YH
        clo[1 + 2 * n + p] = min(min(a1, a2), min(a3, a4)); chi[1 + 2 * n + p] = max(max(a1, a2), max(a3, a4))
    for k in range(2 * n):
        clo[1 + k] = lo[k]; chi[1 + k] = hi[k]


@nb.njit(cache=True)
def k_matvec(rows, cols, vals, mh, ml, nv):
    rh = np.zeros(nv, np.int64); rl = np.zeros(nv, np.int64)
    for e in range(rows.shape[0]):
        rh[cols[e]] += vals[e] * mh[rows[e]]; rl[cols[e]] += vals[e] * ml[rows[e]]
    return rh, rl


# ------------------------------------------------------------------------------------------------ covers
def _parse_points(pe):
    if isinstance(pe, str):
        import ast; pe = ast.literal_eval(pe)
    return [(Fr(x), Fr(y)) for x, y in pe]


def load_covers(n, path=CERT, margin=Fr(1, 2 ** 60)):
    """Certified basin balls (same fields as exp 103 local_certs.json).  Assumption (stated in NOTES): points and radii
    are in the exp 113 frame Omega and the radius is a sup-norm radius around the certified optimum x*."""
    if not os.path.exists(path): return []
    L = json.load(open(path)).get(str(n), [])
    out = []
    for c in L:
        if not c.get('certified') or c.get('basin_radius') is None: continue
        P = _parse_points(c['points_exact'])
        R = Fr(c['basin_radius']) * (1 - Fr(1, 10 ** 6)) - Fr(c['box_rho']) - margin
        assert R > 0
        imgs = [[s3_apply(g, p) for p in P] for g in PERMS]
        tl = c['t_lower']; tl = Fr(int(tl.split('e')[0]), 10 ** int(tl.split('e-')[1])) if 'e-' in str(tl) else Fr(tl)
        out.append(dict(R=R, pts=P, imgs=imgs, fimgs=[np.array([[float(a), float(b)] for a, b in I]) for I in imgs],
                        t_lower=tl, pseudo=False))
    return out


def pseudo_covers(n, R=1e-4):
    """ESTIMATES ONLY (probe without a certificate): a fake 'basin ball' of radius R around the float optimum of
    exp 105 (pool_tri_n{n}.json).  Never used in eps / neg / exact modes."""
    d = json.load(open(os.path.join(POOL, f'pool_tri_n{n}.json')))
    P = [(Fr(x), Fr(y)) for x, y in d[0]['points']]
    imgs = [[s3_apply(g, p) for p in P] for g in PERMS]
    return [dict(R=Fr(R), pts=P, imgs=imgs, fimgs=[np.array([[float(a), float(b)] for a, b in I]) for I in imgs],
                 t_lower=None, pseudo=True)]


# ------------------------------------------------------------------------------------------------ problem
class ProblemT:
    def __init__(self, n, T, covers=(), cover_width=None, skip_gap=None, skip_rho=0.0):
        self.n = n; self.T = Fr(T); assert self.T > 0
        self.T2f = float(2 * self.T); assert Fr(self.T2f) == 2 * self.T, 'T must be a float-representable dyadic'
        self.tri = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64); self.m = len(self.tri)
        self.pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
        self.pid = {p: k for k, p in enumerate(self.pairs)}
        self.tw = np.array([[self.pid[(i, j)], self.pid[(i, k)], self.pid[(j, k)], self.pid[(j, i)], self.pid[(k, i)], self.pid[(k, j)]]
                            for i, j, k in self.tri], dtype=np.int64)
        self.tws = np.array([1, -1, 1, -1, 1, -1], dtype=np.int64)
        self.nv = 1 + 2 * n + len(self.pairs)
        self.pa = np.array([p[0] for p in self.pairs], np.int64); self.pb = np.array([p[1] for p in self.pairs], np.int64)
        self.colscale = np.concatenate([[1.0 / D ** 2], np.full(2 * n, 1.0 / D), np.full(len(self.pairs), 1.0 / D ** 2)])
        self.highs = highspy.Highs(); self.highs.setOptionValue('output_flag', False)
        self.covers = list(covers); self.cover_width = cover_width
        self.audit = 0.0; self.audit_count = 0; self.rng = np.random.default_rng(12345)
        self.T2i = math.floor(2 * self.T * D * D)
        th = 2 * self.T * D * D; self.thr_num, self.thr_den = th.numerator, th.denominator
        self.skip_gap = skip_gap; self.skip_rho = skip_rho
        self._build_lp()

    def roots(self):
        """n^2 root boxes (lo, hi, c): x_0 = 0, y_b = 0, and x_c + y_c = 1 (enforced by P0 / LP via c)."""
        n = self.n; out = []
        for b in range(n):
            for c in range(n):
                lo = np.zeros(2 * n, np.int64); hi = np.full(2 * n, D, np.int64)
                hi[0] = 0; hi[n + b] = 0
                out.append((lo, hi, c))
        return out

    # -------------------------------------------------------------------------------- fixed LP structure
    def _build_lp(self):
        n, m = self.n, self.m; P = len(self.pairs); nv = self.nv
        R, C, V = [], [], []
        def add(r, c, v): R.append(r); C.append(c); V.append(v)
        for t in range(m):                                   # (R1) area rows 2t (s=+1), 2t+1 (s=-1)
            for si, s in enumerate((1, -1)):
                add(2 * t + si, 0, 1)
                for q in range(6): add(2 * t + si, 1 + 2 * n + int(self.tw[t, q]), -s * int(self.tws[q]))
        base = 2 * m
        mcc_y = np.zeros((4, P), np.int64); mcc_x = np.zeros((4, P), np.int64); mcc_r = np.zeros((4, P), np.int64)
        for blk, cw in enumerate((-1, -1, 1, 1)):            # (R2) McCormick
            for p in range(P):
                r = base + blk * P + p; a, b = self.pairs[p]
                add(r, 1 + 2 * n + p, cw)
                mcc_y[blk, p] = len(V); add(r, 1 + n + b, 0)
                mcc_x[blk, p] = len(V); add(r, 1 + a, 0)
                mcc_r[blk, p] = r
        r0 = base + 4 * P
        for i in range(n - 1):                               # (R3) ordering  D X_i - D X_{i+1} <= 0
            add(r0 + i, 1 + i, D); add(r0 + i, 2 + i, -D)
        self.r_hyp = r0 + n - 1                              # (R4) D X_i + D Y_i <= D^2           (n rows, always on)
        for i in range(n): add(self.r_hyp + i, 1 + i, D); add(self.r_hyp + i, 1 + n + i, D)
        self.r_heq = self.r_hyp + n                          # (R5) -D X_i - D Y_i <= -D^2         (n rows, on iff i = c)
        for i in range(n): add(self.r_heq + i, 1 + i, -D); add(self.r_heq + i, 1 + n + i, -D)
        self.r_ab = self.r_heq + n                           # (R6) -D sum X + D sum Y <= 0        (A >= B)
        for i in range(n): add(self.r_ab, 1 + i, -D); add(self.r_ab, 1 + n + i, D)
        self.r_bc = self.r_ab + 1                            # (R7) -D sum X - 2D sum Y <= -n D^2  (B >= C)
        for i in range(n): add(self.r_bc, 1 + i, -D); add(self.r_bc, 1 + n + i, -2 * D)
        nr = self.r_bc + 1
        self.nr = nr; self.mcc_y, self.mcc_x, self.mcc_r = mcc_y, mcc_x, mcc_r
        self.G_r = np.array(R, np.int64); self.G_c = np.array(C, np.int64); self.G_v = np.array(V, np.int64)
        self.H = np.zeros(nr, np.int64)
        self.H[self.r_hyp:self.r_hyp + n] = D * D; self.H[self.r_heq:self.r_heq + n] = -D * D
        self.H[self.r_ab] = 0; self.H[self.r_bc] = -n * D * D
        # static overflow bound for k_matvec: |entry| <= bound_e (McCormick entries are box endpoints in [0, D]);
        # |partial sum| <= sum_e bound_e * 2^25 per column.  Asserted < 2^62.
        bnd = np.abs(self.G_v).astype(np.int64); var_e = np.concatenate([mcc_y.ravel(), mcc_x.ravel()]); bnd[var_e] = D
        colsum = np.bincount(self.G_c, weights=bnd.astype(float), minlength=nv)
        self.colsum_max = int(colsum.max()); self.colcount_max = int(np.bincount(self.G_c, minlength=nv).max())
        assert self.colsum_max * (1 << 25) < (1 << 62)
        assert int(np.abs(self.H).max()) < (1 << 62)
        self.clo = np.zeros(nv, np.int64); self.chi = np.zeros(nv, np.int64); self.chi[0] = 2 * D * D
        cs = self.colscale; self.cs = cs
        lo0 = np.zeros(2 * n, np.int64); hi0 = np.full(2 * n, D, np.int64)
        k_fill(lo0, hi0, n, self.pa, self.pb, self.G_v, self.H, mcc_y, mcc_x, mcc_r, self.clo, self.chi)
        fl = self.G_v.astype(float) / D ** 2 / cs[self.G_c]
        import scipy.sparse as sp
        A = sp.csc_matrix((fl, (self.G_r, self.G_c)), shape=(nr, nv))
        lp = highspy.HighsLp(); lp.num_col_ = nv; lp.num_row_ = nr
        cost = np.zeros(nv); cost[0] = -1.0
        fl0 = self.clo * cs; fl0[0] = -INF
        lp.col_cost_ = cost; lp.col_lower_ = fl0; lp.col_upper_ = self.chi * cs
        self.row_up = self.H.astype(float) / D ** 2
        self.cond = np.zeros(nr, bool); self.cond[:2 * m] = True; self.cond[self.r_heq:self.r_heq + n] = True
        self.row_up[self.cond] = INF                                    # conditional rows off initially
        lp.row_lower_ = np.full(nr, -INF); lp.row_upper_ = self.row_up.copy()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr.astype(np.int32); lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data
        hs = self.highs; hs.clearModel(); hs.passModel(lp)
        self.cur_v = self.G_v.copy(); self.cur_up = self.row_up.copy()
        self.colidx = np.arange(nv, dtype=np.int32)
        self.var_e = var_e
        self.lp_fail = 0; self.fail_status = {}; self.n_infeas = 0; self.n_farkas = 0

    def active_rows(self, sig, c):
        """Boolean mask of rows valid on the box: all unconditional rows, area rows of determined orientation, and the
        hypotenuse-equality row of label c only."""
        m = self.m; act = ~self.cond
        act[0:2 * m:2] = sig == 1; act[1:2 * m:2] = sig == -1
        act[self.r_heq + c] = True
        return act

    # -------------------------------------------------------------------------------- P2
    def lp_bound2(self, lo, hi, sig, c, force=False):
        n, m = self.n, self.m; hs = self.highs; cs = self.cs
        k_fill(lo, hi, n, self.pa, self.pb, self.G_v, self.H, self.mcc_y, self.mcc_x, self.mcc_r, self.clo, self.chi)
        ve = self.var_e; ch = ve[self.G_v[ve] != self.cur_v[ve]]
        for e in ch.tolist():
            cc = int(self.G_c[e]); hs.changeCoeff(int(self.G_r[e]), cc, float(self.G_v[e]) / D ** 2 / cs[cc])
        self.cur_v[ch] = self.G_v[ch]
        act = self.active_rows(sig, c)
        up = self.H.astype(float) / D ** 2
        up[~act] = INF
        cr = np.nonzero(up != self.cur_up)[0].astype(np.int32)
        if len(cr): hs.changeRowsBounds(len(cr), cr, np.full(len(cr), -INF), up[cr]); self.cur_up[cr] = up[cr]
        flo_ = self.clo * cs; flo_[0] = -INF          # tau free below in the float LP (its bounds are not used by
        hs.changeColsBounds(self.nv, self.colidx, flo_, self.chi * cs)   # the certificate); avoids 'infeasible' at tau >= 0
        hs.run()
        status = hs.getModelStatus()
        if status == highspy.HighsModelStatus.kInfeasible:
            # The rows valid on the box are inconsistent (P0 is interval propagation and can miss this).  Try the dual
            # ray as a Farkas multiplier; the SAME exact test decides (s = 0 case: prune iff N < 0).
            self.n_infeas += 1
            _st, has, ray = hs.getDualRay()
            if has:
                ray = np.array(ray); assert len(ray) == self.nr   # review 2026-11-13 (minor)
                for sgn in (1.0, -1.0):
                    ok, _, _ = self._exact(lo, hi, sig, c, np.maximum(sgn * ray, 0.0), act)
                    if ok: self.n_farkas += 1; hs.clearSolver(); return True, None
            self.lp_fail += 1; st_ = str(status); self.fail_status[st_] = self.fail_status.get(st_, 0) + 1
            hs.clearSolver(); return False, None
        if status != highspy.HighsModelStatus.kOptimal:
            self.lp_fail += 1; st_ = str(status); self.fail_status[st_] = self.fail_status.get(st_, 0) + 1
            hs.clearSolver(); return False, None
        val = -hs.getInfo().objective_function_value
        if not force and val > self.T2f * (1 + 1e-7) + 1e-12: return False, val
        lam = np.maximum(-np.array(hs.getSolution().row_dual), 0.0)
        ok, N, s = self._exact(lo, hi, sig, c, lam, act)
        if force and s is not None and s > 0: self._last_bound = Fr(N, s) / (D * D); self._last_mu = self._mu
        return ok, val

    def _exact(self, lo, hi, sig, c, lam, act):
        """Exact integer certificate (Lemma P2' in NOTES.md).  lam: float multipliers proposed by HiGHS (any values).
        mu = floor(lam 2^S) with rows not valid on the box zeroed; r = G^T mu exactly; s = r_tau;
        N = mu^T H + sum_{u != tau} max(-r_u lo_u, -r_u hi_u).  Prune iff (s > 0 and N <= 2 T D^2 s) or (s == 0 and N < 0).
        Returns (ok, N, s)."""
        lam = np.array(lam, float); lam[~act] = 0.0          # SOUNDNESS: only rows valid on this box may be used
        lam[~np.isfinite(lam)] = 0.0
        lmax = lam.max(); self._mu = None
        if not (lmax > 0): return False, None, None
        S = 49 - max(0, int(math.ceil(math.log2(lmax))))
        mu = np.floor(np.ldexp(lam, S)).astype(np.int64); self._mu = mu
        assert mu.min() >= 0 and mu.max() < (1 << 50)
        assert not mu[~act].any()
        rh, rl = k_matvec(self.G_r, self.G_c, self.G_v, mu >> 25, mu & ((1 << 25) - 1), self.nv)
        rhl = rh.tolist(); rll = rl.tolist()
        s = rhl[0] * (1 << 25) + rll[0]
        if s < 0: return False, None, s
        Hl = self.H.tolist(); mul = mu.tolist()
        N = 0
        for i in np.nonzero(mu)[0].tolist(): N += mul[i] * Hl[i]
        cl = self.clo.tolist(); chh = self.chi.tolist()
        for u in range(1, self.nv):
            ru = rhl[u] * (1 << 25) + rll[u]
            if ru: N += max(-ru * cl[u], -ru * chh[u])
        ok = (N * self.thr_den <= s * self.thr_num) if s > 0 else (N < 0)
        if self.audit and ok and self.rng.random() < self.audit:
            self.audit_count += 1
            if not self.audit2(lo, hi, sig, c, mu): raise RuntimeError('AUDIT MISMATCH')
        return ok, N, s

    def audit2(self, lo, hi, sig, c, mu):
        """Independent Fraction re-derivation from the row definitions (natural units), rows in _build_lp order;
        rows not valid on the box must carry mu = 0.  Returns True iff the aggregated inequality gives tau <= 2T."""
        n = self.n; F = lambda v: Fr(int(v), D)
        x = lambda i: ('p', i); y = lambda i: ('p', n + i); w = lambda a, b: ('w', a, b)
        rows = []
        for t in range(self.m):
            i, j, k = (int(v) for v in self.tri[t])
            for sg in (1, -1):
                terms = {('tau',): Fr(1)}
                for (a, b), cc in (((i, j), 1), ((i, k), -1), ((j, k), 1), ((j, i), -1), ((k, i), 1), ((k, j), -1)):
                    terms[w(a, b)] = terms.get(w(a, b), 0) - sg * cc
                rows.append((terms, Fr(0), int(sig[t]) == sg))
        for blk in range(4):
            for (a, b) in self.pairs:
                xl, xh, yl, yh = F(lo[a]), F(hi[a]), F(lo[n + b]), F(hi[n + b])
                if blk == 0: rows.append(({w(a, b): Fr(-1), y(b): xl, x(a): yl}, xl * yl, True))
                if blk == 1: rows.append(({w(a, b): Fr(-1), y(b): xh, x(a): yh}, xh * yh, True))
                if blk == 2: rows.append(({w(a, b): Fr(1), y(b): -xh, x(a): -yl}, -xh * yl, True))
                if blk == 3: rows.append(({w(a, b): Fr(1), y(b): -xl, x(a): -yh}, -xl * yh, True))
        for i in range(n - 1): rows.append(({x(i): Fr(1), x(i + 1): Fr(-1)}, Fr(0), True))
        for i in range(n): rows.append(({x(i): Fr(1), y(i): Fr(1)}, Fr(1), True))
        for i in range(n): rows.append(({x(i): Fr(-1), y(i): Fr(-1)}, Fr(-1), i == c))
        ab = {}; bc = {}
        for i in range(n): ab[x(i)] = Fr(-1); ab[y(i)] = Fr(1); bc[x(i)] = Fr(-1); bc[y(i)] = Fr(-2)
        rows.append((ab, Fr(0), True)); rows.append((bc, Fr(-n), True))
        assert len(rows) == len(mu)
        agg = {}; rhs = Fr(0)
        for (terms, h, valid), mm in zip(rows, mu):
            if mm == 0: continue
            if not valid: return False
            mm = int(mm)
            for v, cc in terms.items(): agg[v] = agg.get(v, 0) + mm * cc
            rhs += mm * h
        s = agg.pop(('tau',), 0)
        if s < 0: return False
        for v, cc in agg.items():
            if cc == 0: continue
            if v[0] == 'p': vl, vh = F(lo[v[1]]), F(hi[v[1]])
            else:
                a, b = v[1], v[2]; P_ = [F(lo[a]) * F(lo[n + b]), F(lo[a]) * F(hi[n + b]), F(hi[a]) * F(lo[n + b]), F(hi[a]) * F(hi[n + b])]
                vl, vh = min(P_), max(P_)
            rhs += max(-cc * vl, -cc * vh)
        return (rhs / s <= 2 * self.T) if s > 0 else (rhs < 0)

    # -------------------------------------------------------------------------------- cover
    def covered(self, lo, hi):
        """Exact: exists g in S3 and a bijection j such that for every label i and every corner q of the rectangle of
        point i, || g^{-1}(q) - x~_{j(i)} ||_inf <= R.  (|.| of an affine function is convex, so the corners suffice:
        then every point P of the box has || g^{-1}(P) - pi(x~) ||_inf <= R.)  Floats only propose g and j."""
        if not self.covers: return False
        n = self.n
        cx = (lo[:n] + hi[:n]) / (2 * D); cy = (lo[n:] + hi[n:]) / (2 * D)
        cpt = np.stack([cx, cy], 1)
        for cv in self.covers:
            Rf = float(cv['R'])
            for gi, Fm in enumerate(cv['fimgs']):
                d = np.abs(cpt[:, None, :] - Fm[None, :, :]).max(2)
                j = d.argmin(1)
                if len(set(j.tolist())) < n or d[np.arange(n), j].max() > 2.5 * Rf: continue
                g = PERMS[gi]; X0 = cv['pts']; R = cv['R']; good = True
                for i in range(n):
                    px, py = X0[j[i]]
                    for qx in (Fr(int(lo[i]), D), Fr(int(hi[i]), D)):
                        for qy in (Fr(int(lo[n + i]), D), Fr(int(hi[n + i]), D)):
                            ux, uy = s3_inverse_apply(g, (qx, qy))
                            if abs(ux - px) > R or abs(uy - py) > R: good = False; break
                        if not good: break
                    if not good: break
                if good: return True
        return False

    # -------------------------------------------------------------------------------- one node
    def process(self, lo, hi, pred, stats, c):
        n = self.n
        if not k_propagate(lo, hi, n, c, D): return 'p0', None
        dead, sig = k_ranges(lo, hi, self.tri, n, self.T2i)
        if dead: return 'p1', None
        wmax = int((hi - lo).max())
        if self.covers and wmax <= self.cover_width and self.covered(lo, hi): return 'cover', None
        if self.skip_gap is not None and pred is not None and pred > self.skip_gap:
            stats['lpskip'] = stats.get('lpskip', 0) + 1; gch = pred * self.skip_rho
        else:
            stats['lp'] += 1
            ok, val = self.lp_bound2(lo, hi, sig, c)
            if ok: return 'p2', None
            gch = None if val is None else (val / self.T2f - 1)
        k = k_choose(lo, hi, sig, self.tri, n)
        if k < 0: return 'fail', None
        mid = (int(lo[k]) + int(hi[k])) // 2
        l2, h2 = lo.copy(), hi.copy(); h2[k] = mid
        l3, h3 = lo.copy(), hi.copy(); l3[k] = mid
        f = (mid - int(lo[k])) / max(1, int(hi[k] - lo[k]))
        return 'split', ((l2, h2, f), (l3, h3, 1 - f), gch)

    def solve2(self, lo, hi, c, stats, node_budget=None, fails=None, depth=0, weight=1.0, pred=None, max_fail=None):
        """DFS; stack items (lo, hi, depth, weight, pred, c).  Returns the unexplored remainder."""
        stack = [(np.array(lo, np.int64), np.array(hi, np.int64), depth, weight, pred, c)]
        stats.setdefault('closed_w', 0.0); nodes0 = stats['nodes']
        while stack:
            if node_budget is not None and stats['nodes'] - nodes0 >= node_budget: return stack
            lo, hi, dep, wt, pr, cc = stack.pop()
            stats['nodes'] += 1
            if dep > stats['maxdepth']: stats['maxdepth'] = dep
            kind, ch = self.process(lo, hi, pr, stats, cc)
            if kind == 'split':
                (l2, h2, f2), (l3, h3, f3), g = ch
                stack.append((l3, h3, dep + 1, wt * f3, g, cc)); stack.append((l2, h2, dep + 1, wt * f2, g, cc))
                continue
            stats[kind] += 1; stats['closed_w'] += wt
            if kind == 'fail' and fails is not None:
                fails.append((lo.tolist(), hi.tolist(), cc))
                if max_fail is not None and len(fails) >= max_fail: return stack + [None]
        return []


def new_stats():
    return dict(nodes=0, p0=0, p1=0, p2=0, lp=0, cover=0, fail=0, maxdepth=0, lpskip=0)


# ------------------------------------------------------------------------------------------------ thresholds
def fstar_bounds(n):
    """Rational (lower, upper) bounds of the known / conjectured f*(n) (f = min area, Omega of area 1/2)."""
    if n == 5:                                            # (3 - 2 sqrt 2) / 2
        K = 10 ** 30; s = math.isqrt(2 * K * K)           # s <= sqrt2 K < s + 1
        return (3 - 2 * Fr(s + 1, K)) / 2, (3 - 2 * Fr(s, K)) / 2
    if n == 6: return Fr(1, 16), Fr(1, 16)
    if n == 7: return Fr(7, 144), Fr(7, 144)
    if n == 8:                                            # numerical only: 0.0338946066270986
        v = Fr('0.0338946066270986'); return v * (1 - Fr(1, 10 ** 13)), v * (1 + Fr(1, 10 ** 13))
    raise ValueError(n)


def make_problem(n, mode, eps=1e-3, pseudo_R=1e-4, **kw):
    """eps:   T >= f*_upper (1 + eps), no covers  -> claim max f <= T.
       neg:   T <= f*_lower (1 - eps), no covers  -> must FAIL (negative control).
       exact: T = t_lower (1 - 1e-9) rounded down, certified covers from local_certs_tri.json (required).
       probe: as exact, but if no certificate exists, pseudo covers (radius pseudo_R around the exp 105 float
              optimum) and T = f*_lower (1 - 1e-9).  ESTIMATES ONLY; never a proof."""
    flo, fhi = fstar_bounds(n)
    cw = D // 1000
    if mode == 'eps':
        T = np.nextafter(float(fhi * (1 + Fr(eps))), 1.0); assert Fr(float(T)) >= fhi * (1 + Fr(eps)); covers = []
    elif mode == 'neg':
        T = np.nextafter(float(flo * (1 - Fr(eps))), 0.0); assert Fr(float(T)) <= flo * (1 - Fr(eps)); covers = []
    elif mode in ('exact', 'probe'):
        covers = load_covers(n)
        if covers:
            tl = covers[0]['t_lower']; T = np.nextafter(float(tl * (1 - Fr(1, 10 ** 9))), 0.0); assert Fr(float(T)) < tl
        elif mode == 'exact':
            raise RuntimeError(f'exact mode needs {CERT} with a certified entry for n = {n}')
        else:
            covers = pseudo_covers(n, pseudo_R); T = np.nextafter(float(flo * (1 - Fr(1, 10 ** 9))), 0.0)
    else:
        raise ValueError(mode)
    return ProblemT(n, Fr(float(T)), covers=covers, cover_width=cw, **kw), (flo, fhi)
