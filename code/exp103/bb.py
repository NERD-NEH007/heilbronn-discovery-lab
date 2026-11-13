"""Rigorous branch-and-bound for the Heilbronn problem in the unit square (exp 103).

Claim proved for given n and dyadic threshold T:
    every configuration of n points in [0,1]^2 has  min triangle area <= T,
    except configurations lying (up to a square symmetry and relabelling) in a box that is covered by the
    local certificate of exp 44 (sup-ball of the proven basin radius around the certified exact optimum x*).
With T < t* and every surviving box covered, this gives: max min-area = t*, attained only at x* (up to symmetry).

Search domain (all reductions are exact, see NOTES.md):
  * labels sorted by x:  x_0 <= x_1 <= ... <= x_{n-1}
  * stretching: wlog x_0 = 0, x_{n-1} = 1, and y_b = 0, y_t = 1 for some b != t; by the mirror y -> 1-y wlog b < t
    (b = smallest label on the bottom wall, t = smallest label on the top wall).  One root box per pair (b, t).
Coordinates are integers on the grid 1/D, D = 2^24 (box endpoints only; points inside boxes are arbitrary reals).

Pruning tests (each is exact):
  P0  ordering propagation empties the box.
  P1  some triple has max |2A| <= 2T over the box (|A| is convex in each point, so the max over the product of
      boxes is attained at a corner tuple; the 64 corner values are computed in float64, which is exact here because
      all endpoints are multiples of 2^-24 in [0,1]).
  P2  LP relaxation (McCormick envelopes of w_ab = x_a*y_b, triples with determined orientation): an integer
      Farkas/Neumaier-Shcherbina bound computed in exact Python integers from the (float) LP duals.
  COVER  box inside a certified basin ball.
"""
import os, sys, json, time, itertools, math
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
from fractions import Fraction as Fr
import numpy as np
import scipy.sparse as sp
import highspy
from scipy.optimize import linprog

LOGD = 24
D = 1 << LOGD
MAPS = [((1, 0), (0, 1)), ((0, -1), (1, 0)), ((-1, 0), (0, -1)), ((0, 1), (-1, 0)),
        ((-1, 0), (0, 1)), ((1, 0), (0, -1)), ((0, 1), (1, 0)), ((0, -1), (-1, 0))]


def apply_map(M, p):                       # exact: c + M (p - c), c = (1/2, 1/2)
    h = Fr(1, 2); dx, dy = p[0] - h, p[1] - h
    return (h + M[0][0] * dx + M[0][1] * dy, h + M[1][0] * dx + M[1][1] * dy)


class Problem:
    def __init__(self, n, T, covers=(), lp_width=None, branch='invol', cover_width=None, fbbt_rounds=0, centroid=None, domain='walls'):
        self.n = n
        self.T = Fr(T)                     # prune threshold (exact rational; must be dyadic-friendly, any Fraction ok)
        assert self.T > 0
        self.T2f = float(2 * self.T)       # used only in P1 comparisons; must be exact:
        assert Fr(self.T2f) == 2 * self.T, 'T must be a float-representable dyadic'
        self.tri = np.array(list(itertools.combinations(range(n), 3)), dtype=np.int64)
        self.m = len(self.tri)
        self.pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
        self.pid = {p: k for k, p in enumerate(self.pairs)}
        # 2A(i,j,k) = w_ij - w_ik + w_jk - w_ji + w_ki - w_kj
        self.tw = np.array([[self.pid[(i, j)], self.pid[(i, k)], self.pid[(j, k)], self.pid[(j, i)], self.pid[(k, i)], self.pid[(k, j)]]
                            for i, j, k in self.tri], dtype=np.int64)
        self.tws = np.array([1, -1, 1, -1, 1, -1], dtype=np.int64)
        self.pt_tri = [np.where((self.tri == i).any(1))[0] for i in range(n)]
        self.nv = 1 + 2 * n + len(self.pairs)          # tau, x_0..x_{n-1}, y_0..y_{n-1}, w
        self.covers = covers                            # list of dict(R=Fraction radius, imgs=[list of exact points], fimgs=float arrays)
        self.lp_width = lp_width if lp_width is not None else D   # call LP only when max free width <= this (grid units)
        self.cover_width = cover_width
        self.fbbt_rounds = fbbt_rounds
        self.domain = domain
        self.centroid = (domain == 'basic') if centroid is None else centroid
        assert not (self.centroid and domain in ('walls', 'xwall')), 'x-mirror is used by the walls domains'
        self.cur_cons = []; self.cons_table = [[]]
        self.audit = 0.0; self.audit_count = 0; self.rng = np.random.default_rng(12345)
        self._last = None
        self.pa = np.array([p[0] for p in self.pairs]); self.pb = np.array([p[1] for p in self.pairs])
        self.colscale = np.concatenate([[1.0 / D ** 2], np.full(2 * n, 1.0 / D), np.full(len(self.pairs), 1.0 / D ** 2)])
        self.highs = highspy.Highs(); self.highs.setOptionValue('output_flag', False)
        self.branch = branch

    # ---------------------------------------------------------------------------------------- root boxes
    def roots(self):
        """list of (lo, hi, cid); self.cons_table[cid] = extra constraints [(i, j): v_i <= v_j] (coordinate indices)."""
        n = self.n; out = []
        X = lambda i: i
        Y = lambda i: n + i
        if self.domain == 'basic':
            self.cons_table = [[]]
            for b in range(n):
                for t in range(b + 1, n):
                    lo = np.zeros(2 * n, np.int64); hi = np.full(2 * n, D, np.int64)
                    hi[0] = 0; lo[n - 1] = D                       # x_0 = 0, x_{n-1} = 1
                    hi[n + b] = 0; lo[n + t] = D                   # y_b = 0, y_t = 1
                    out.append((lo, hi, 0))
            return out
        if self.domain == 'xwall':
            # Lemma W on the vertical walls only (labels lex-sorted by (x, y); x-mirror so that |L| >= |R|)
            self.cons_table = [[(Y(0), Y(1)), (Y(0), Y(n - 1))],          # XA: |L| >= 2: x_0 = x_1 = 0, y_0 <= y_1, y_0 <= y_{n-1}
                               [(Y(0), Y(n - 1)), (Y(n - 1), Y(0))]]       # XB: |L| = |R| = 1: y_0 = y_{n-1}
            for cid in (0, 1):
                for b in range(n):
                    for t in range(b + 1, n):
                        lo = np.zeros(2 * n, np.int64); hi = np.full(2 * n, D, np.int64)
                        hi[0] = 0; lo[n - 1] = D
                        if cid == 0: hi[1] = 0
                        hi[n + b] = 0; lo[n + t] = D
                        if (lo > hi).any(): continue
                        out.append((lo, hi, cid))
            return out
        # domain 'walls': necessary conditions at a global maximiser (NOTES.md, Lemma W)
        self.cons_table = []
        xcases = []
        xcases.append(([0, 1], [n - 2, n - 1], [(Y(0), Y(1)), (Y(n - 2), Y(n - 1)), (Y(0), Y(n - 1))]))     # X1
        xcases.append(([0, 1], [n - 1], [(Y(0), Y(1)), (Y(0), Y(n - 1))]))                                   # X2
        xcases.append(([0], [n - 1], [(Y(0), Y(n - 1)), (Y(n - 1), Y(0))]))                                   # X3
        ycases = []
        for b1, b2 in itertools.combinations(range(n), 2):
            rest = [i for i in range(n) if i not in (b1, b2)]
            for t1, t2 in itertools.combinations(rest, 2):                                                     # Y1
                ycases.append(([b1, b2], [t1, t2], [(X(b1), X(t2)), (X(t1), X(b2))]))
            for t in rest:                                                                                     # Y2
                ycases.append(([b1, b2], [t], [(X(b1), X(t)), (X(t), X(b2))]))
        for b in range(n):
            for t in range(n):
                if t != b: ycases.append(([b], [t], [(X(b), X(t)), (X(t), X(b))]))                            # Y3
        for (L, R, cx) in xcases:
            for (Bs, Ts, cy) in ycases:
                lo = np.zeros(2 * n, np.int64); hi = np.full(2 * n, D, np.int64)
                for i in L: hi[X(i)] = 0
                for i in R: lo[X(i)] = D
                for i in Bs: hi[Y(i)] = 0
                for i in Ts: lo[Y(i)] = D
                if (lo > hi).any(): continue
                self.cons_table.append(cx + cy)
                out.append((lo, hi, len(self.cons_table) - 1))
        return out

    # ---------------------------------------------------------------------------------------- P0
    def propagate(self, lo, hi):
        n = self.n
        for it in range(20):
            ch = False
            for i in range(n - 1):
                if lo[i + 1] < lo[i]: lo[i + 1] = lo[i]; ch = True
            for i in range(n - 2, -1, -1):
                if hi[i] > hi[i + 1]: hi[i] = hi[i + 1]; ch = True
            for (i, j) in self.cur_cons:                       # v_i <= v_j
                if lo[j] < lo[i]: lo[j] = lo[i]; ch = True
                if hi[i] > hi[j]: hi[i] = hi[j]; ch = True
            if self.centroid:
                cap = n * D // 2; sl = int(lo[:n].sum())
                if sl > cap: return False
                for i in range(n):
                    u = cap - (sl - int(lo[i]))
                    if hi[i] > u: hi[i] = u; ch = True
            if (lo > hi).any(): return False
            if not ch: break
        return bool((lo <= hi).all())

    # ---------------------------------------------------------------------------------------- P1
    def triple_ranges(self, lo, hi):
        n = self.n
        xl, xh = lo[:n] / D, hi[:n] / D; yl, yh = lo[n:] / D, hi[n:] / D     # exact floats (24-bit)
        cx = np.stack([xl, xl, xh, xh], 1); cy = np.stack([yl, yh, yl, yh], 1)  # (n,4) corners
        I, J, K = self.tri[:, 0], self.tri[:, 1], self.tri[:, 2]
        xi = cx[I][:, :, None, None]; yi = cy[I][:, :, None, None]
        xj = cx[J][:, None, :, None]; yj = cy[J][:, None, :, None]
        xk = cx[K][:, None, None, :]; yk = cy[K][:, None, None, :]
        A2 = (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)                 # exact (see module doc)
        A2 = A2.reshape(self.m, 64)
        return A2.min(1), A2.max(1)

    # ---------------------------------------------------------------------------------------- FBBT
    def corners_A2(self, L6, H6):
        """L6, H6: (m',6) int arrays [x_i, x_j, x_k, y_i, y_j, y_k] -> (m',64) exact 2A at corner tuples."""
        X = [np.stack([L6[:, q], L6[:, q], H6[:, q], H6[:, q]], 1) / D for q in range(3)]
        Y = [np.stack([L6[:, 3 + q], H6[:, 3 + q], L6[:, 3 + q], H6[:, 3 + q]], 1) / D for q in range(3)]
        xi, yi = X[0][:, :, None, None], Y[0][:, :, None, None]
        xj, yj = X[1][:, None, :, None], Y[1][:, None, :, None]
        xk, yk = X[2][:, None, None, :], Y[2][:, None, None, :]
        return ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)).reshape(len(L6), 64)

    def tighten(self, lo, hi, sig):
        """Bound tightening from determined triples.  Float interval arithmetic only proposes new bounds; each is
        accepted only after an exact corner check that the cut-off slab has sigma*2A <= 2T for that triple (so it
        holds no configuration with all |A| > T).  Returns number of tightened bounds."""
        n = self.n; T2 = self.T2f
        det = np.where(sig != 0)[0]
        if len(det) == 0: return 0
        tri = self.tri[det]; s = sig[det].astype(float)
        fl, fh = lo / D, hi / D
        def iv_mul(al, ah, bl, bh):
            p = np.stack([al * bl, al * bh, ah * bl, ah * bh]); return p.min(0), p.max(0)
        cands = []                                       # (triple row in tri, coord global index, kind, new int bound)
        for rot in range(3):
            I, J, K = tri[:, rot], tri[:, (rot + 1) % 3], tri[:, (rot + 2) % 3]
            xI, xJ, xK = (fl[I], fh[I]), (fl[J], fh[J]), (fl[K], fh[K])
            yI, yJ, yK = (fl[n + I], fh[n + I]), (fl[n + J], fh[n + J]), (fl[n + K], fh[n + K])
            # x_I: coef = yJ - yK ; rest = xJ (yK - yI) + xK (yI - yJ)
            cl, ch = yJ[0] - yK[1], yJ[1] - yK[0]
            a = iv_mul(xJ[0], xJ[1], yK[0] - yI[1], yK[1] - yI[0]); b = iv_mul(xK[0], xK[1], yI[0] - yJ[1], yI[1] - yJ[0])
            rl, rh = a[0] + b[0], a[1] + b[1]
            cands.append((I, cl, ch, rl, rh))
            # y_I: coef = xK - xJ ; rest = xI (yJ - yK) + xJ yK - xK yJ
            cl2, ch2 = xK[0] - xJ[1], xK[1] - xJ[0]
            a = iv_mul(xI[0], xI[1], yJ[0] - yK[1], yJ[1] - yK[0]); b = iv_mul(xJ[0], xJ[1], yK[0], yK[1]); c_ = iv_mul(xK[0], xK[1], yJ[0], yJ[1])
            rl2, rh2 = a[0] + b[0] - c_[1], a[1] + b[1] - c_[0]
            cands.append((n + I, cl2, ch2, rl2, rh2))
        ks, kinds, vals, rows = [], [], [], []
        for (kc, cl, ch, rl, rh) in cands:
            # sigma-adjusted: sigma*coef in [ccl, cch], sigma*rest <= rrh
            ccl = np.where(s > 0, cl, -ch); cch = np.where(s > 0, ch, -cl); rrh = np.where(s > 0, rh, -rl)
            q = T2 - rrh
            with np.errstate(divide='ignore', invalid='ignore'):
                m1 = (q > 0) & (cch > 0)
                nl = np.where(m1, q / np.where(cch > 0, cch, 1), -1.0)
                m2 = (q <= 0) & (cch < 0)
                nh = np.where(m2, q / np.where(cch < 0, cch, -1), 2.0)
            Lp = np.floor(nl * D).astype(np.int64) - 1; Up = np.ceil(nh * D).astype(np.int64) + 1
            g1 = m1 & (Lp > lo[kc]) & (Lp <= hi[kc]); g2 = m2 & (Up < hi[kc]) & (Up >= lo[kc])
            for g, kind, val in ((g1, 0, Lp), (g2, 1, Up)):
                w = np.where(g)[0]
                if len(w): ks.append(kc[w]); kinds.append(np.full(len(w), kind)); vals.append(val[w]); rows.append(w)
        if not ks: return 0
        ks = np.concatenate(ks); kinds = np.concatenate(kinds); vals = np.concatenate(vals); rows = np.concatenate(rows)
        # exact verification: slab box for each candidate
        tr = tri[rows]
        idx6 = np.concatenate([tr, n + tr], 1)                       # global coord indices of the 6 coords
        L6 = lo[idx6].copy(); H6 = hi[idx6].copy()
        col = (idx6 == ks[:, None])
        # slab: kind 0 (cut [lo, L']): coord range [lo, L'];  kind 1 (cut [U', hi]): coord range [U', hi]
        H6 = np.where(col & (kinds[:, None] == 0), vals[:, None], H6)
        L6 = np.where(col & (kinds[:, None] == 1), vals[:, None], L6)
        A2 = self.corners_A2(L6, H6) * sig[det][rows][:, None]
        okv = A2.max(1) <= T2
        nt = 0
        for k_, kind, v in zip(ks[okv], kinds[okv], vals[okv]):
            if kind == 0 and v > lo[k_]: lo[k_] = v; nt += 1
            elif kind == 1 and v < hi[k_]: hi[k_] = v; nt += 1
        return nt

    # ---------------------------------------------------------------------------------------- P2
    def lp_bound(self, lo, hi, sig, force=False):
        """sig: int array (m,) in {+1,-1,0}.  Returns (proved_pruned, float_lp_value, info)."""
        n = self.n; T2 = self.T2f
        det = np.where(sig != 0)[0]
        if len(det) == 0: return False, None, None
        used = np.unique(self.tw[det].ravel())
        a = np.array([self.pairs[k][0] for k in used]); b = np.array([self.pairs[k][1] for k in used])
        XL, XH = lo[a], hi[a]; YL, YH = lo[n + b], hi[n + b]                # int64 grid units
        nu = len(used); wcol = 1 + 2 * n + used; xcol = 1 + a; ycol = 1 + n + b
        # ---- integer (scaled by D^2) rows
        rows, cols, vals, rhs = [], [], [], []
        r0 = 0
        # area rows: tau - sig * sum(+-w) <= 0
        nd = len(det)
        rr = np.repeat(np.arange(nd), 7)
        cc = np.concatenate([np.zeros((nd, 1), np.int64), 1 + 2 * n + self.tw[det]], 1).ravel()
        vv = np.concatenate([np.ones((nd, 1), np.int64), -sig[det][:, None] * self.tws[None, :]], 1).ravel()
        rows.append(rr); cols.append(cc); vals.append(vv); rhs.append(np.zeros(nd, np.int64)); r0 = nd
        # McCormick rows (4 per used pair)
        k = np.arange(nu)
        for (cw, cy_, cx_, rh) in ((-1, XL, YL, XL * YL), (-1, XH, YH, XH * YH), (1, -XH, -YL, -XH * YL), (1, -XL, -YH, -XL * YH)):
            rows.append(np.concatenate([r0 + k, r0 + k, r0 + k])); cols.append(np.concatenate([wcol, ycol, xcol]))
            vals.append(np.concatenate([np.full(nu, cw, np.int64), cy_, cx_])); rhs.append(rh); r0 += nu
        # ordering rows: D*x_i - D*x_{i+1} <= 0
        ko = np.arange(n - 1)
        rows.append(np.concatenate([r0 + ko, r0 + ko])); cols.append(np.concatenate([1 + ko, 2 + ko]))
        vals.append(np.concatenate([np.full(n - 1, D, np.int64), np.full(n - 1, -D, np.int64)])); rhs.append(np.zeros(n - 1, np.int64)); r0 += n - 1
        if self.cur_cons:     # extra constraints D*v_i - D*v_j <= 0
            cc_ = np.array(self.cur_cons, np.int64); kc = len(cc_)
            rows.append(np.concatenate([r0 + np.arange(kc), r0 + np.arange(kc)])); cols.append(np.concatenate([1 + cc_[:, 0], 1 + cc_[:, 1]]))
            vals.append(np.concatenate([np.full(kc, D, np.int64), np.full(kc, -D, np.int64)])); rhs.append(np.zeros(kc, np.int64)); r0 += kc
        if self.centroid:     # D * sum_i x^_i <= D^2 n / 2   (x-mirror normalisation: mean x <= 1/2)
            rows.append(np.full(n, r0)); cols.append(1 + np.arange(n)); vals.append(np.full(n, D, np.int64))
            rhs.append(np.array([D * D * n // 2], np.int64)); r0 += 1
        R = np.concatenate(rows); C = np.concatenate(cols); V = np.concatenate(vals); H = np.concatenate(rhs)
        Gc = sp.csc_matrix((V, (R, C)), shape=(r0, self.nv), dtype=np.int64)
        # column bounds (integers, scaled): tau in [0, 2 D^2]; p in box; w in product hull
        L_ = lo[:n, None] * lo[None, n:]; LH = lo[:n, None] * hi[None, n:]; HL = hi[:n, None] * lo[None, n:]; H_ = hi[:n, None] * hi[None, n:]
        Wl = np.minimum(np.minimum(L_, LH), np.minimum(HL, H_)); Wh = np.maximum(np.maximum(L_, LH), np.maximum(HL, H_))
        clo = np.concatenate([[0], lo, Wl[self.pa, self.pb]]); chi = np.concatenate([[2 * D * D], hi, Wh[self.pa, self.pb]])
        # ---- float LP in natural units: row_nat = row_int / D^2, v_nat = v_int * colscale
        cs = self.colscale
        colof = np.repeat(np.arange(self.nv), np.diff(Gc.indptr))
        fv = Gc.data.astype(float) / (D ** 2) / cs[colof]
        hf = H.astype(float) / D ** 2
        lp = highspy.HighsLp()
        lp.num_col_ = self.nv; lp.num_row_ = r0
        cost = np.zeros(self.nv); cost[0] = -1.0
        lp.col_cost_ = cost; lp.col_lower_ = clo * cs; lp.col_upper_ = chi * cs
        lp.row_lower_ = np.full(r0, -highspy.kHighsInf); lp.row_upper_ = hf
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = Gc.indptr.astype(np.int32); lp.a_matrix_.index_ = Gc.indices.astype(np.int32); lp.a_matrix_.value_ = fv
        hs = self.highs; hs.clearModel(); hs.passModel(lp); hs.run()
        if hs.getModelStatus() != highspy.HighsModelStatus.kOptimal: return False, None, dict(status=str(hs.getModelStatus()))
        val = -hs.getInfo().objective_function_value
        xsol = np.array(hs.getSolution().col_value); self._last = (xsol, used)
        lam = np.maximum(-np.array(hs.getSolution().row_dual), 0.0)
        if not force and val > T2 * (1 + 1e-7) + 1e-12: return False, val, None          # LP itself says no
        # ---- exact integer certificate:  mu = floor(lam * 2^S)
        lmax = lam.max()
        if not (lmax > 0): return False, val, None
        S = 49 - max(0, int(math.ceil(math.log2(lmax))))
        mu = np.floor(np.ldexp(lam, S)).astype(np.int64)                    # 0 <= mu < 2^50
        assert mu.min() >= 0 and mu.max() < (1 << 50)
        assert np.abs(V).max() <= D
        mh, ml = mu >> 25, mu & ((1 << 25) - 1)
        GT = Gc.T                                                            # csr, exact int64
        assert int(abs(Gc).sum(0).max()) < (1 << 37)                         # column |coef| sums < 2^37 -> |GT@mh| < 2^62
        rh = GT @ mh; rl = GT @ ml
        r = [int(x) * (1 << 25) + int(y) for x, y in zip(rh, rl)]           # exact Python ints
        s = r[0]                                                             # coefficient of tau^ = sum of area multipliers
        if s <= 0: return False, val, None
        N = sum(int(m_) * int(h_) for m_, h_ in zip(mu, H) if m_)
        for u in range(1, self.nv):
            ru = r[u]
            if ru: N += max(-ru * int(clo[u]), -ru * int(chi[u]))
        # tau^ <= N / s  and tau^ = D^2 * tau = D^2 * 2 min sigma A ;  prune iff N/s <= D^2 * 2T
        thr = 2 * self.T * D * D
        ok = Fr(N) <= s * thr
        if self.audit and ok and self.rng.random() < self.audit:
            ok2 = self.audit_cert(lo, hi, sig, det, used, mu)
            self.audit_count += 1
            if ok2 != ok: raise RuntimeError('AUDIT MISMATCH')
        return ok, val, dict(bound=float(Fr(N, s) / (D * D)) / 2)

    def audit_cert(self, lo, hi, sig, det, used, mu):
        """Independent Fraction re-derivation of the LP certificate (natural units, rows rebuilt from their
        definitions in the same order as lp_bound).  Returns True iff the aggregated inequality proves
        min over determined triples of sigma*2A <= 2T on the box."""
        n = self.n; F = lambda v: Fr(int(v), D)
        x = lambda i: ('p', i); y = lambda i: ('p', n + i); w = lambda a, b: ('w', a, b)
        rows = []                                          # (dict var->coef, rhs) meaning sum coef*var <= rhs
        for t in det:
            i, j, k = (int(v) for v in self.tri[t]); sg = int(sig[t])
            terms = {('tau',): Fr(1)}
            for (a, b), c in (((i, j), 1), ((i, k), -1), ((j, k), 1), ((j, i), -1), ((k, i), 1), ((k, j), -1)):
                terms[w(a, b)] = terms.get(w(a, b), 0) - sg * c
            rows.append((terms, Fr(0)))
        prs = [self.pairs[u] for u in used]
        XLs = [F(lo[a]) for a, b in prs]; XHs = [F(hi[a]) for a, b in prs]; YLs = [F(lo[n + b]) for a, b in prs]; YHs = [F(hi[n + b]) for a, b in prs]
        for blk in range(4):
            for q, (a, b) in enumerate(prs):
                xl, xh, yl, yh = XLs[q], XHs[q], YLs[q], YHs[q]
                if blk == 0: rows.append(({w(a, b): Fr(-1), y(b): xl, x(a): yl}, xl * yl))      # w >= xl y + yl x - xl yl
                if blk == 1: rows.append(({w(a, b): Fr(-1), y(b): xh, x(a): yh}, xh * yh))      # w >= xh y + yh x - xh yh
                if blk == 2: rows.append(({w(a, b): Fr(1), y(b): -xh, x(a): -yl}, -xh * yl))    # w <= xh y + yl x - xh yl
                if blk == 3: rows.append(({w(a, b): Fr(1), y(b): -xl, x(a): -yh}, -xl * yh))    # w <= xl y + yh x - xl yh
        for i in range(n - 1): rows.append(({x(i): Fr(1), x(i + 1): Fr(-1)}, Fr(0)))
        for (i, j) in self.cur_cons: rows.append(({('p', i): Fr(1), ('p', j): Fr(-1)}, Fr(0)))
        if self.centroid: rows.append(({x(i): Fr(1) for i in range(n)}, Fr(n, 2)))
        assert len(rows) == len(mu)
        agg = {}; rhs = Fr(0)
        for (terms, h), m in zip(rows, mu):
            if m == 0: continue
            m = int(m)
            for v, c in terms.items(): agg[v] = agg.get(v, 0) + m * c
            rhs += m * h
        s = agg.pop(('tau',), 0)
        if s <= 0: return False
        for v, c in agg.items():                           # max over box of -c*v
            if c == 0: continue
            if v[0] == 'p': vl, vh = F(lo[v[1]]), F(hi[v[1]])
            else:
                a, b = v[1], v[2]; P_ = [F(lo[a]) * F(lo[n + b]), F(lo[a]) * F(hi[n + b]), F(hi[a]) * F(lo[n + b]), F(hi[a]) * F(hi[n + b])]
                vl, vh = min(P_), max(P_)
            rhs += max(-c * vl, -c * vh)
        return rhs / s <= 2 * self.T

    # ---------------------------------------------------------------------------------------- COVER
    def covered(self, lo, hi):
        if not self.covers: return False
        n = self.n
        cx = (lo[:n] + hi[:n]) / (2 * D); cy = (lo[n:] + hi[n:]) / (2 * D)
        c = np.stack([cx, cy], 1)
        for cv in self.covers:
            Rf = float(cv['R'])
            for g, F in enumerate(cv['fimgs']):
                d = np.abs(c[:, None, :] - F[None, :, :]).max(2)      # (n box points, n image points)
                j = d.argmin(1)
                if len(set(j.tolist())) < n or d[np.arange(n), j].max() > Rf: continue
                E = cv['imgs'][g]; R = cv['R']; good = True
                for i in range(n):
                    px, py = E[j[i]]
                    for v_lo, v_hi, p in ((lo[i], hi[i], px), (lo[n + i], hi[n + i], py)):
                        if max(abs(Fr(int(v_lo), D) - p), abs(Fr(int(v_hi), D) - p)) > R: good = False; break
                    if not good: break
                if good: return True
        return False

    # ---------------------------------------------------------------------------------------- branching
    def choose(self, lo, hi, sig, A2lo, A2hi, lpinfo=None):
        """returns (coordinate, split point) or None"""
        n = self.n; w = (hi - lo).astype(float)
        if self.branch == 'lpgap' and lpinfo is not None:
            xs, used = lpinfo
            a = self.pa[used]; b = self.pb[used]
            xa = xs[1 + a]; yb = xs[1 + n + b]; ww = xs[1 + 2 * n + used]
            gap = np.abs(ww - xa * yb)
            sc = np.zeros(2 * n); np.add.at(sc, a, gap); np.add.at(sc, n + b, gap)
            sc[w < 2] = -1
            k = int(sc.argmax())
            if sc[k] > 1e-12:
                v = xs[1 + k] * D
                q = int(round(min(max(v, lo[k] + 0.25 * w[k]), hi[k] - 0.25 * w[k])))
                q = min(max(q, int(lo[k]) + 1), int(hi[k]) - 1) if w[k] >= 2 else q
                return k, q
        if self.branch == 'widest':
            score = w
        else:
            und = (sig == 0)
            cnt = np.zeros(n)
            np.add.at(cnt, self.tri[und].ravel(), 1)
            score = w * (1 + np.concatenate([cnt, cnt]))
        score[w < 2] = -1
        k = int(score.argmax())
        return (k, (int(lo[k]) + int(hi[k])) // 2) if score[k] > 0 else None

    # ---------------------------------------------------------------------------------------- DFS
    def solve(self, lo, hi, stats, node_budget=None, fails=None, max_fail=50, depth=0, weight=1.0, cid=0):
        stack = [(lo.copy(), hi.copy(), depth, weight, cid)]
        stats.setdefault('closed_w', 0.0)
        T2 = self.T2f; nodes0 = stats['nodes']
        while stack:
            if node_budget is not None and stats['nodes'] - nodes0 >= node_budget:
                return stack
            lo, hi, dep, wt, cid = stack.pop(); self.cur_cons = self.cons_table[cid]
            stats['nodes'] += 1
            if 'hist' in stats:
                lw = int(np.log2(max(1, int((hi - lo).max())))); stats['hist'][lw] = stats['hist'].get(lw, 0) + 1
            if dep > stats['maxdepth']: stats['maxdepth'] = dep
            if not self.propagate(lo, hi): stats['p0'] += 1; stats['closed_w'] += wt; continue
            dead = False
            for rnd in range(self.fbbt_rounds + 1):
                A2lo, A2hi = self.triple_ranges(lo, hi)
                if (np.maximum(A2hi, -A2lo) <= T2).any(): dead = True; break
                sig = np.where(A2lo >= -T2, 1, np.where(A2hi <= T2, -1, 0)).astype(np.int64)
                if rnd == self.fbbt_rounds: break
                nt = self.tighten(lo, hi, sig)
                stats['tight'] += nt
                if nt == 0: break
                if not self.propagate(lo, hi): dead = True; break
            if dead: stats['p1'] += 1; stats['closed_w'] += wt; continue
            wmax = int((hi - lo).max())
            if self.covers and wmax <= self.cover_width and self.covered(lo, hi): stats['cover'] += 1; stats['closed_w'] += wt; continue
            lpinfo = None
            if wmax <= self.lp_width:
                stats['lp'] += 1
                self._last = None
                ok, val, info = self.lp_bound(lo, hi, sig)
                lpinfo = self._last
                if ok:
                    stats['p2'] += 1; stats['closed_w'] += wt
                    if 'hist2' in stats: stats['hist2'][lw] = stats['hist2'].get(lw, 0) + 1
                    continue
            kq = self.choose(lo, hi, sig, A2lo, A2hi, lpinfo)
            if kq is None:
                stats['fail'] += 1; stats['closed_w'] += wt
                if fails is not None:
                    fails.append((lo.tolist(), hi.tolist(), cid))
                    if len(fails) >= max_fail: return stack + [None]     # aborted
                continue
            k, mid = kq
            f = (mid - int(lo[k])) / max(1, int(hi[k] - lo[k]))
            l2, h2 = lo.copy(), hi.copy(); h2[k] = mid
            l3, h3 = lo.copy(), hi.copy(); l3[k] = mid
            stack.append((l3, h3, dep + 1, wt * (1 - f), cid)); stack.append((l2, h2, dep + 1, wt * f, cid))
        return []


def new_stats():
    return dict(nodes=0, p0=0, p1=0, p2=0, lp=0, cover=0, fail=0, maxdepth=0, tight=0)


def load_covers(n, path=None, margin=Fr(1, 2 ** 60)):
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                'certificates', 'exp103', 'local_certs.json')
    L = json.load(open(path)).get(str(n), [])
    out = []
    for c in L:
        if not c.get('certified') or c.get('basin_radius') is None: continue
        P = [(Fr(x), Fr(y)) for x, y in c['points_exact']]
        # basin radius is stored as float (rounded); use a rational strictly below it, minus the existence radius
        R = Fr(c['basin_radius']) * (1 - Fr(1, 10 ** 6)) - Fr(c['box_rho']) - margin
        imgs = [[apply_map(M, p) for p in P] for M in MAPS]
        out.append(dict(R=R, imgs=imgs, fimgs=[np.array([[float(a), float(b)] for a, b in I]) for I in imgs],
                        t_lower=Fr(int(c['t_lower'].split('e')[0]), 10 ** int(c['t_lower'].split('e-')[1]))))
    return out
