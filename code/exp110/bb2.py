"""exp 110: faster rigorous branch-and-bound (builds on exp 103 bb.py, which is imported unchanged).

Same statement, search domain ('basic'), pruning tests and certificate logic as exp 103, plus (see NOTES.md):
  (a) one persistent HiGHS model with a FIXED row/column structure; per node only bounds / McCormick coefficients
      are changed, so HiGHS hot-starts from the previous basis.  Area rows exist for both orientations of every
      triple; a row is switched on (upper 0) only if the triple's orientation is determined on the box, and the
      exact certificate zeroes the multiplier of every switched-off row (so only rows valid on the box are used).
  (b) hot code compiled with numba; P1 corner ranges now in exact int64 (no float exactness argument needed).
  (c) diagonal symmetry: wlog sum (x_i - 1/2)^2 >= sum (y_i - 1/2)^2, used as an exact integer prune and an exact
      (outward-rounded) bound tightening inside P0.
  (d) LP skipping: a node whose predicted LP gap (from the last solved ancestor) is large skips the LP and is split.
      A skip never closes a box, so it cannot affect soundness.
"""
import os, sys, math, itertools
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
B103 = os.path.join(REPO, 'code', 'exp103')
if B103 not in sys.path: sys.path.insert(0, B103)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
from fractions import Fraction as Fr
import numpy as np
import numba as nb
import highspy
import bb                                     # exp 103 (unchanged): D, MAPS, load_covers, Problem (covers, roots)

D = bb.D
INF = highspy.kHighsInf


# ------------------------------------------------------------------------------------------------ compiled kernels
@nb.njit(cache=True)
def _isqrt(v):
    if v <= 0: return 0
    s = int(math.sqrt(float(v)))
    while s * s > v: s -= 1
    while (s + 1) * (s + 1) <= v: s += 1
    return s


@nb.njit(cache=True)
def k_propagate(lo, hi, n, centroid, diag, Dg):
    """P0 in exact int64.  Constraints: x sorted; sum x <= n/2 (centroid); sum(2x-1)^2 >= sum(2y-1)^2 (diag).
    Returns False iff the box provably contains no point of the domain.  Only shrinks the box, never removes a
    domain point (all new bounds rounded outward)."""
    amax = np.zeros(n, np.int64); bmin = np.zeros(n, np.int64)
    for it in range(40):
        ch = False
        for i in range(n - 1):
            if lo[i + 1] < lo[i]: lo[i + 1] = lo[i]; ch = True
        for i in range(n - 2, -1, -1):
            if hi[i] > hi[i + 1]: hi[i] = hi[i + 1]; ch = True
        if centroid:
            cap = n * Dg // 2; sl = 0
            for i in range(n): sl += lo[i]
            if sl > cap: return False
            for i in range(n):
                u = cap - (sl - lo[i])
                if hi[i] > u: hi[i] = u; ch = True
        for i in range(2 * n):
            if lo[i] > hi[i]: return False
        if diag:
            # a_i = 2 x_i - Dg, b_i = 2 y_i - Dg (x, y real in grid units);  domain: sum a^2 >= sum b^2
            at = 0; bt = 0
            for i in range(n):
                al = 2 * lo[i] - Dg; ah = 2 * hi[i] - Dg
                amax[i] = max(al * al, ah * ah); at += amax[i]
                bl = 2 * lo[n + i] - Dg; bh = 2 * hi[n + i] - Dg
                if bl <= 0 and bh >= 0: bmin[i] = 0
                else: bmin[i] = min(bl * bl, bh * bh)
                bt += bmin[i]
            if at < bt: return False
            for i in range(n):
                # y_i: b_i^2 <= cap  ->  |b_i| <= sqrt(cap) < s + 1
                cap = at - (bt - bmin[i]); s = _isqrt(cap)
                nh = (Dg + s + 2) // 2; nl = (Dg - s - 1) // 2
                if hi[n + i] > nh: hi[n + i] = nh; ch = True
                if lo[n + i] < nl: lo[n + i] = nl; ch = True
                # x_i: a_i^2 >= need  ->  |a_i| >= sqrt(need) >= r
                need = bt - (at - amax[i])
                if need > 0:
                    r = _isqrt(need)
                    al = 2 * lo[i] - Dg; ah = 2 * hi[i] - Dg
                    if al > -r and ah < r: return False
                    if al > -r:
                        nl2 = (Dg + r) // 2
                        if lo[i] < nl2: lo[i] = nl2; ch = True
                    elif ah < r:
                        nh2 = (Dg - r + 1) // 2
                        if hi[i] > nh2: hi[i] = nh2; ch = True
            for i in range(2 * n):
                if lo[i] > hi[i]: return False
        if not ch: break
    for i in range(2 * n):
        if lo[i] > hi[i]: return False
    return True


@nb.njit(cache=True)
def k_ranges(lo, hi, tri, n, T2i):
    """P1 in exact int64: corner range of 2A*D^2 per triple.  Returns (dead, sig).
    dead: some triple has max |2A| <= 2T (T2i = floor(2 T D^2), exact since 2A*D^2 is an integer at corners)."""
    m = tri.shape[0]; sig = np.zeros(m, np.int64)
    cx = np.empty(4, np.int64); cy = np.empty(4, np.int64)
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
    """exp 103 'invol' rule: maximise width * (1 + #undetermined triples containing the point)."""
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
    """Exact integer (scaled) McCormick coefficients/rhs and column bounds for the box."""
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
    """r = G^T mu exactly as two int64 halves (mu = mh*2^25 + ml); overflow excluded by the static bound in __init__."""
    rh = np.zeros(nv, np.int64); rl = np.zeros(nv, np.int64)
    for e in range(rows.shape[0]):
        rh[cols[e]] += vals[e] * mh[rows[e]]; rl[cols[e]] += vals[e] * ml[rows[e]]
    return rh, rl


# ------------------------------------------------------------------------------------------------ problem
class Problem2(bb.Problem):
    def __init__(self, n, T, covers=(), cover_width=None, diag=True, skip_gap=None, skip_rho=0.0, record=None):
        super().__init__(n, T, covers=covers, branch='invol', cover_width=cover_width, domain='basic')
        assert self.centroid
        self.diag = bool(diag)
        self.T2i = math.floor(2 * self.T * D * D)                 # exact
        th = 2 * self.T * D * D; self.thr_num, self.thr_den = th.numerator, th.denominator
        self.skip_gap = skip_gap; self.skip_rho = skip_rho; self.record = record
        self._build_lp()

    def roots(self):
        out = super().roots()
        self.cons_table = [[]]
        return out

    # -------------------------------------------------------------------------------- fixed LP structure
    def _build_lp(self):
        n, m = self.n, self.m; P = len(self.pairs); nv = self.nv
        R, C, V = [], [], []
        def add(r, c, v): R.append(r); C.append(c); V.append(v)
        # area rows: row 2t (s=+1), 2t+1 (s=-1):  tau - s * 2A(w) <= 0   (2A = sum tws * w[tw])
        for t in range(m):
            for si, s in enumerate((1, -1)):
                add(2 * t + si, 0, 1)
                for q in range(6): add(2 * t + si, 1 + 2 * n + int(self.tw[t, q]), -s * int(self.tws[q]))
        base = 2 * m
        mcc_y = np.zeros((4, P), np.int64); mcc_x = np.zeros((4, P), np.int64); mcc_r = np.zeros((4, P), np.int64)
        for blk, cw in enumerate((-1, -1, 1, 1)):
            for p in range(P):
                r = base + blk * P + p; a, b = self.pairs[p]
                add(r, 1 + 2 * n + p, cw)
                mcc_y[blk, p] = len(V); add(r, 1 + n + b, 0)       # coefficient of y_b (filled per node)
                mcc_x[blk, p] = len(V); add(r, 1 + a, 0)           # coefficient of x_a
                mcc_r[blk, p] = r
        base2 = base + 4 * P
        for i in range(n - 1):
            add(base2 + i, 1 + i, D); add(base2 + i, 2 + i, -D)
        cr = base2 + n - 1
        for i in range(n): add(cr, 1 + i, D)
        nr = cr + 1
        self.nr = nr; self.mcc_y, self.mcc_x, self.mcc_r = mcc_y, mcc_x, mcc_r
        self.G_r = np.array(R, np.int64); self.G_c = np.array(C, np.int64); self.G_v = np.array(V, np.int64)
        self.H = np.zeros(nr, np.int64); self.H[cr] = D * D * n // 2
        assert D * D * n % 2 == 0
        # static overflow bound for k_matvec: |coef| <= D, nnz per column <= cnt -> |sum| < cnt * D * 2^25 < 2^63
        cnt = np.bincount(self.G_c, minlength=nv).max()
        assert cnt * D * (1 << 25) < (1 << 62)
        self.clo = np.zeros(nv, np.int64); self.chi = np.zeros(nv, np.int64); self.chi[0] = 2 * D * D
        # float model in natural units (column scale cs as exp 103)
        cs = self.colscale; self.cs = cs
        fl = self.G_v.astype(float) / D ** 2 / cs[self.G_c]
        # initial McCormick coefficients: unit box values (any values work; they are overwritten per node)
        lo0 = np.zeros(2 * n, np.int64); hi0 = np.full(2 * n, D, np.int64)
        k_fill(lo0, hi0, n, self.pa, self.pb, self.G_v, self.H, mcc_y, mcc_x, mcc_r, self.clo, self.chi)
        fl = self.G_v.astype(float) / D ** 2 / cs[self.G_c]
        import scipy.sparse as sp
        A = sp.csc_matrix((fl, (self.G_r, self.G_c)), shape=(nr, nv))
        lp = highspy.HighsLp(); lp.num_col_ = nv; lp.num_row_ = nr
        cost = np.zeros(nv); cost[0] = -1.0
        lp.col_cost_ = cost; lp.col_lower_ = self.clo * cs; lp.col_upper_ = self.chi * cs
        self.row_up = self.H.astype(float) / D ** 2
        self.row_up[:2 * m] = INF                                    # all area rows off initially
        lp.row_lower_ = np.full(nr, -INF); lp.row_upper_ = self.row_up.copy()
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr.astype(np.int32); lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data
        hs = self.highs; hs.clearModel(); hs.passModel(lp)
        self.cur_v = self.G_v.copy()                                 # integer coefficients currently in the model
        self.cur_up = self.row_up.copy()
        self.colidx = np.arange(nv, dtype=np.int32); self.allrows = np.arange(nr, dtype=np.int32)
        self.var_e = np.concatenate([mcc_y.ravel(), mcc_x.ravel()])   # entries that vary per node
        self.lp_fail = 0; self.lp_iters = 0

    # -------------------------------------------------------------------------------- P2 (hot-started)
    def lp_bound2(self, lo, hi, sig, force=False):
        n, m = self.n, self.m; hs = self.highs; cs = self.cs
        k_fill(lo, hi, n, self.pa, self.pb, self.G_v, self.H, self.mcc_y, self.mcc_x, self.mcc_r, self.clo, self.chi)
        # coefficient updates (only changed entries)
        ve = self.var_e; ch = ve[self.G_v[ve] != self.cur_v[ve]]
        for e in ch.tolist():
            c = int(self.G_c[e]); hs.changeCoeff(int(self.G_r[e]), c, float(self.G_v[e]) / D ** 2 / cs[c])
        self.cur_v[ch] = self.G_v[ch]
        # row uppers: area rows on/off, McCormick rhs
        up = self.H.astype(float) / D ** 2
        act = np.zeros(2 * m, bool); act[0::2] = sig == 1; act[1::2] = sig == -1
        up[:2 * m] = np.where(act, 0.0, INF)
        cr = np.nonzero(up != self.cur_up)[0].astype(np.int32)
        if len(cr): hs.changeRowsBounds(len(cr), cr, np.full(len(cr), -INF), up[cr]); self.cur_up[cr] = up[cr]
        hs.changeColsBounds(self.nv, self.colidx, self.clo * cs, self.chi * cs)
        hs.run()
        if hs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            self.lp_fail += 1; hs.clearSolver(); return False, None
        val = -hs.getInfo().objective_function_value
        if not force and val > self.T2f * (1 + 1e-7) + 1e-12: return False, val
        lam = np.maximum(-np.array(hs.getSolution().row_dual), 0.0)
        lam[:2 * m][~act] = 0.0                       # SOUNDNESS: only rows valid on this box may be used
        lmax = lam.max()
        if not (lmax > 0): return False, val
        S = 49 - max(0, int(math.ceil(math.log2(lmax))))
        mu = np.floor(np.ldexp(lam, S)).astype(np.int64)
        assert mu.min() >= 0 and mu.max() < (1 << 50)
        rh, rl = k_matvec(self.G_r, self.G_c, self.G_v, mu >> 25, mu & ((1 << 25) - 1), self.nv)
        rhl = rh.tolist(); rll = rl.tolist()
        s = rhl[0] * (1 << 25) + rll[0]
        if s <= 0: return False, val
        Hl = self.H.tolist(); mul = mu.tolist()
        N = 0
        for i in np.nonzero(mu)[0].tolist(): N += mul[i] * Hl[i]
        cl = self.clo.tolist(); chh = self.chi.tolist()
        for u in range(1, self.nv):
            ru = rhl[u] * (1 << 25) + rll[u]
            if ru: N += max(-ru * cl[u], -ru * chh[u])
        ok = N * self.thr_den <= s * self.thr_num             # N/s <= 2 T D^2
        if force: self._last_bound = Fr(N, s) / (D * D)        # certified bound on min sigma*2A (tests only)
        if self.audit and ok and self.rng.random() < self.audit:
            self.audit_count += 1
            if not self.audit2(lo, hi, sig, mu): raise RuntimeError('AUDIT MISMATCH')
        return ok, val

    def audit2(self, lo, hi, sig, mu):
        """Independent Fraction re-derivation of the aggregated inequality from the row definitions (natural units),
        rows in the fixed order of _build_lp; rows of undetermined orientation must carry mu = 0."""
        n = self.n; F = lambda v: Fr(int(v), D)
        x = lambda i: ('p', i); y = lambda i: ('p', n + i); w = lambda a, b: ('w', a, b)
        rows = []
        for t in range(self.m):
            i, j, k = (int(v) for v in self.tri[t])
            for sg in (1, -1):
                terms = {('tau',): Fr(1)}
                for (a, b), c in (((i, j), 1), ((i, k), -1), ((j, k), 1), ((j, i), -1), ((k, i), 1), ((k, j), -1)):
                    terms[w(a, b)] = terms.get(w(a, b), 0) - sg * c
                rows.append((terms, Fr(0), int(sig[t]) == sg))
        for blk in range(4):
            for (a, b) in self.pairs:
                xl, xh, yl, yh = F(lo[a]), F(hi[a]), F(lo[n + b]), F(hi[n + b])
                if blk == 0: rows.append(({w(a, b): Fr(-1), y(b): xl, x(a): yl}, xl * yl, True))
                if blk == 1: rows.append(({w(a, b): Fr(-1), y(b): xh, x(a): yh}, xh * yh, True))
                if blk == 2: rows.append(({w(a, b): Fr(1), y(b): -xh, x(a): -yl}, -xh * yl, True))
                if blk == 3: rows.append(({w(a, b): Fr(1), y(b): -xl, x(a): -yh}, -xl * yh, True))
        for i in range(n - 1): rows.append(({x(i): Fr(1), x(i + 1): Fr(-1)}, Fr(0), True))
        rows.append(({x(i): Fr(1) for i in range(n)}, Fr(n, 2), True))
        assert len(rows) == len(mu)
        agg = {}; rhs = Fr(0)
        for (terms, h, valid), mm in zip(rows, mu):
            if mm == 0: continue
            if not valid: return False
            mm = int(mm)
            for v, c in terms.items(): agg[v] = agg.get(v, 0) + mm * c
            rhs += mm * h
        s = agg.pop(('tau',), 0)
        if s <= 0: return False
        for v, c in agg.items():
            if c == 0: continue
            if v[0] == 'p': vl, vh = F(lo[v[1]]), F(hi[v[1]])
            else:
                a, b = v[1], v[2]; P_ = [F(lo[a]) * F(lo[n + b]), F(lo[a]) * F(hi[n + b]), F(hi[a]) * F(lo[n + b]), F(hi[a]) * F(hi[n + b])]
                vl, vh = min(P_), max(P_)
            rhs += max(-c * vl, -c * vh)
        return rhs / s <= 2 * self.T

    # -------------------------------------------------------------------------------- one node
    def process(self, lo, hi, pred, stats):
        """Returns (kind, children).  kind in p0, p1, cover, p2, fail, split.  Modifies lo/hi in place (P0)."""
        n = self.n
        if not k_propagate(lo, hi, n, True, self.diag, D): return 'p0', None
        dead, sig = k_ranges(lo, hi, self.tri, n, self.T2i)
        if dead: return 'p1', None
        wmax = int((hi - lo).max())
        if self.covers and wmax <= self.cover_width and self.covered(lo, hi): return 'cover', None
        if self.skip_gap is not None and pred is not None and pred > self.skip_gap:
            stats['lpskip'] = stats.get('lpskip', 0) + 1; gch = pred * self.skip_rho
        else:
            stats['lp'] += 1
            ok, val = self.lp_bound2(lo, hi, sig)
            if self.record is not None and pred is not None and val is not None:
                self.record.append((pred, val / self.T2f - 1, ok))
            if ok: return 'p2', None
            gch = None if val is None else (val / self.T2f - 1)
        k = k_choose(lo, hi, sig, self.tri, n)
        if k < 0: return 'fail', None
        mid = (int(lo[k]) + int(hi[k])) // 2
        l2, h2 = lo.copy(), hi.copy(); h2[k] = mid
        l3, h3 = lo.copy(), hi.copy(); l3[k] = mid
        f = (mid - int(lo[k])) / max(1, int(hi[k] - lo[k]))
        return 'split', ((l2, h2, f), (l3, h3, 1 - f), gch)

    def solve2(self, lo, hi, stats, node_budget=None, fails=None, depth=0, weight=1.0, pred=None, max_fail=None):
        """DFS; stack items (lo, hi, depth, weight, pred).  Returns the unexplored remainder."""
        stack = [(np.array(lo, np.int64), np.array(hi, np.int64), depth, weight, pred)]
        stats.setdefault('closed_w', 0.0); nodes0 = stats['nodes']
        while stack:
            if node_budget is not None and stats['nodes'] - nodes0 >= node_budget: return stack
            lo, hi, dep, wt, pr = stack.pop()
            stats['nodes'] += 1
            if dep > stats['maxdepth']: stats['maxdepth'] = dep
            kind, ch = self.process(lo, hi, pr, stats)
            if kind == 'split':
                (l2, h2, f2), (l3, h3, f3), g = ch
                stack.append((l3, h3, dep + 1, wt * f3, g)); stack.append((l2, h2, dep + 1, wt * f2, g))
                continue
            stats[kind] += 1; stats['closed_w'] += wt
            if kind == 'fail' and fails is not None:
                fails.append((lo.tolist(), hi.tolist(), 0))
                if max_fail is not None and len(fails) >= max_fail: return stack + [None]     # aborted
        return []


def new_stats():
    s = bb.new_stats(); s['lpskip'] = 0; return s


def make_problem(n, mode, eps=1e-3, **kw):
    """Same thresholds as exp 103 run_par.make_problem."""
    import json
    C = json.load(open(os.path.join(REPO, 'certificates', 'exp103', 'local_certs.json')))
    tl = C[str(n)][0]['t_lower']; tlow = Fr(int(tl.split('e')[0]), 10 ** int(tl.split('e-')[1]))
    if mode == 'eps':
        T = np.nextafter(float(tlow * (1 + Fr(eps))), 1.0); covers = []
        assert Fr(float(T)) >= tlow * (1 + Fr(eps))
    elif mode == 'neg':                          # negative control: T below t*, no covers -> must fail
        T = np.nextafter(float(tlow * (1 - Fr(eps))), 0.0); covers = []
    else:
        T = np.nextafter(float(tlow * (1 - Fr(1, 10 ** 9))), 0.0); covers = bb.load_covers(n)
        assert covers and Fr(float(T)) < tlow
    return Problem2(n, Fr(float(T)), covers=covers, cover_width=D // 1000, **kw), tlow
