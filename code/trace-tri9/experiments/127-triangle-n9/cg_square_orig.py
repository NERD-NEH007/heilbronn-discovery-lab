"""Exp 126: cell-graph proof method (after Panorias, "An exact computer-assisted proof for ten points
in Heilbronn's triangle problem", Zenodo 24186293 v1.1, 2026), re-implemented independently from the
mathematics in the paper (no code from external/ was opened or run).  Square, n = 9.

Statement proved by a complete run:  no configuration in family F_j has Phi > T  (Phi = min triangle area),
for a rational T.  With the normalization (NOTES.md) this gives Delta_9 <= T.

Exactness: every closing step uses integer arithmetic (numpy int64 with static overflow bounds, or Python
ints).  Floats are used only by the LP that PROPOSES positive-combination weights; the weights are rounded to
integers and re-checked exactly.

Coordinates are integers in units 1/Q, Q = G * 2^K.  A box is R[p] = [xlo, xhi, ylo, yhi] for each point p.
Support points have one coordinate fixed (lo = hi); each support parameter belongs to exactly one point, so the
parameter box is a product of point rectangles and Lemma 7.1 (multi-affine determinant, extremes at vertices)
applies directly.
"""
import itertools, math, time
from fractions import Fraction as Fr
import numpy as np
import numba as nb

N = 9
TRI = np.array(list(itertools.combinations(range(N), 3)), dtype=np.int64)
TSTAR = (9 * math.sqrt(65) - 55) / 320


def T_above(eps):
    """rational T with T > t*, T ~ t*(1+eps); exact check (320T+55)^2 > 81*65."""
    T = Fr(math.ceil(TSTAR * (1 + eps) * 10 ** 13), 10 ** 13)
    assert 320 * T + 55 > 0 and (320 * T + 55) ** 2 > 81 * 65
    return T


def T_below(rel):
    T = Fr(math.floor(TSTAR * (1 - rel) * 10 ** 13), 10 ** 13)
    assert (320 * T + 55) ** 2 < 81 * 65
    return T


DELTA7_UP = Fr(7, 80)   # Delta_7 = 0.083859.. (Zeng-Chen 2011; re-proved by our exp 103/110); weakened as Panorias


def gap_l0(T):
    """Lemma 4.1 (n = 9 form): two points on one side at distance l and Phi = s > 0 put the other 7 points in a
    1 x (1 - 2s/l) rectangle, so s <= D7 (1 - 2s/l), s <= D7 l/(l + 2 D7).  s > T therefore needs
    l > 2 D7 T/(D7 - T).  Returned exactly (Fraction)."""
    D7 = DELTA7_UP
    assert T < D7
    return 2 * D7 * T / (D7 - T)


# ------------------------------------------------------------------------------------------------ kernels
@nb.njit(cache=True)
def tri_range(R, i, j, k):
    """exact min/max of D = det(Pj-Pi, Pk-Pi) over the product of the three rectangles (vertex enumeration;
    D is affine in (xk, yk) once Pi, Pj fixed, so those two are optimised in closed form)."""
    lo = 1 << 62
    hi = -(1 << 62)
    for a in range(2):
        xi = R[i, a]
        for b in range(2):
            yi = R[i, 2 + b]
            for c in range(2):
                xj = R[j, c]
                for d in range(2):
                    yj = R[j, 2 + d]
                    p = xj - xi            # coefficient of yk
                    q = -(yj - yi)         # coefficient of xk
                    r = -p * yi - q * xi
                    if p >= 0:
                        pmin = p * R[k, 2]; pmax = p * R[k, 3]
                    else:
                        pmin = p * R[k, 3]; pmax = p * R[k, 2]
                    if q >= 0:
                        qmin = q * R[k, 0]; qmax = q * R[k, 1]
                    else:
                        qmin = q * R[k, 1]; qmax = q * R[k, 0]
                    v = r + pmin + qmin
                    if v < lo: lo = v
                    v = r + pmax + qmax
                    if v > hi: hi = v
    return lo, hi


@nb.njit(cache=True)
def all_ranges(R, TRI):
    m = TRI.shape[0]
    LO = np.empty(m, np.int64); HI = np.empty(m, np.int64)
    for t in range(m):
        l, h = tri_range(R, TRI[t, 0], TRI[t, 1], TRI[t, 2])
        LO[t] = l; HI[t] = h
    return LO, HI


@nb.njit(cache=True)
def propagate(R, ORD):
    """orders R[p1].c1 + g <= R[p2].c2 (c = 0 x, 1 y).  Shrinks endpoints to necessary values; False if empty."""
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
    for p in range(R.shape[0]):
        if R[p, 0] > R[p, 1] or R[p, 2] > R[p, 3]:
            return False
    return True


@nb.njit(cache=True)
def choose_split(R, LO, HI, TRI, rq):
    """branching heuristic (not part of the proof).  Score per coordinate = sum over triples of
    width*|dD/dv| / max(excess, 1) where excess is how far the triple is from a T-closure."""
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
        # dD/dxi = yj - yk, dD/dyi = xk - xj (cyclic)
        for (a, b, c) in ((i, j, k), (j, k, i), (k, i, j)):
            gx = max(abs(R[b, 3] - R[c, 2]), abs(R[b, 2] - R[c, 3]))
            gy = max(abs(R[c, 1] - R[b, 0]), abs(R[c, 0] - R[b, 1]))
            score[2 * a] += (R[a, 1] - R[a, 0]) * gx * w8
            score[2 * a + 1] += (R[a, 3] - R[a, 2]) * gy * w8
    best = -1; bs = 0.0
    for v in range(2 * n):
        p = v // 2; c = v % 2
        if R[p, 2 * c + 1] - R[p, 2 * c] >= 2 and score[v] > bs:
            bs = score[v]; best = v
    return best


@nb.njit(cache=True)
def cell_excluded(S, G, cs, rq):
    """S: (k,4) support rectangles; returns (G*G,) bool: cell excluded by a two-support witness
    (|D(S1,S2,P)| <= rho for all S1, S2, P in their boxes)."""
    k = S.shape[0]
    out = np.zeros(G * G, np.bool_)
    R = np.empty((3, 4), np.int64)
    for cx in range(G):
        for cy in range(G):
            R[2, 0] = cx * cs; R[2, 1] = (cx + 1) * cs; R[2, 2] = cy * cs; R[2, 3] = (cy + 1) * cs
            done = False
            for a in range(k):
                if done: break
                for b in range(a + 1, k):
                    for z in range(4):
                        R[0, z] = S[a, z]; R[1, z] = S[b, z]
                    lo, hi = tri_range(R, 0, 1, 2)
                    if lo >= -rq and hi <= rq:
                        done = True; break
            out[cx * G + cy] = done
    return out


@nb.njit(cache=True)
def edge_matrix(S, cells, G, cs, rq):
    """adjacency among retained cells: edge removed if one support witness gives |D(S,C1,C2)| <= rho
    for all points of S's box and both cells.  Same cell: no edge (two free points in one cell give |D| <= 2h <= rho)."""
    m = cells.shape[0]; k = S.shape[0]
    A = np.zeros((m, m), np.bool_)
    R = np.empty((3, 4), np.int64)
    for u in range(m):
        cu = cells[u]
        for v in range(u + 1, m):
            cv = cells[v]
            R[1, 0] = (cu // G) * cs; R[1, 1] = (cu // G + 1) * cs; R[1, 2] = (cu % G) * cs; R[1, 3] = (cu % G + 1) * cs
            R[2, 0] = (cv // G) * cs; R[2, 1] = (cv // G + 1) * cs; R[2, 2] = (cv % G) * cs; R[2, 3] = (cv % G + 1) * cs
            ok = True
            for a in range(k):
                for z in range(4):
                    R[0, z] = S[a, z]
                lo, hi = tri_range(R, 0, 1, 2)
                if lo >= -rq and hi <= rq:
                    ok = False; break
            A[u, v] = ok; A[v, u] = ok
    return A


@nb.njit(cache=True)
def three_cell_blocked(CB, sel, ns, v, rq):
    """True if some triple (sel[a], sel[b], v) of cells has |D| <= rho for all points of the three cells
    (checked three-free-point obstruction, paper sec. 7.5)."""
    R = np.empty((3, 4), np.int64)
    for z in range(4):
        R[2, z] = CB[v, z]
    for a in range(ns):
        for z in range(4):
            R[0, z] = CB[sel[a], z]
        for b in range(a + 1, ns):
            for z in range(4):
                R[1, z] = CB[sel[b], z]
            lo, hi = tri_range(R, 0, 1, 2)
            if lo >= -rq and hi <= rq:
                return True
    return False


# ------------------------------------------------------------------------------------------------ families
class Family:
    """supports: list of (fx, fy) with fx/fy a number in {0,1} (fixed) or None (parameter).  Orders:
    (p1, c1, p2, c2, gap_units).  Lower bounds for single parameters: (p, c, lo_units)."""

    def __init__(self, name, supports, orders, lows, Q):
        self.name = name; self.k = len(supports); self.m = N - self.k
        self.sup = supports; self.Q = Q
        self.ORD = np.array(orders, dtype=np.int64).reshape(-1, 5)
        self.lows = lows

    def root(self):
        Q = self.Q
        S = np.zeros((self.k, 4), np.int64)
        for p, (fx, fy) in enumerate(self.sup):
            S[p, 0:2] = (0, Q) if fx is None else (fx * Q, fx * Q)
            S[p, 2:4] = (0, Q) if fy is None else (fy * Q, fy * Q)
        for (p, c, lo) in self.lows:
            S[p, 2 * c] = max(S[p, 2 * c], lo)
        return S


def make_family(name, Q, l0u):
    # l0u = floor(l0 * Q): weaker than the exact gap, hence a valid necessary condition
    if name == 'F6':   # L0=(0,a) L1=(0,b) B0=(c,0) B1=(d,0) R=(1,e) T=(f,1); a<=e<=b, c<=f<=d, b-a>=l0, d-c>=l0
        sup = [(0, None), (0, None), (None, 0), (None, 0), (1, None), (None, 1)]
        ords = [(0, 1, 4, 1, 0), (4, 1, 1, 1, 0), (2, 0, 5, 0, 0), (5, 0, 3, 0, 0), (0, 1, 1, 1, l0u), (2, 0, 3, 0, l0u)]
        return Family(name, sup, ords, [], Q)
    if name == 'F5':   # O=(0,0) L1=(0,b) B1=(d,0) R=(1,e) T=(f,1); e<=b, f<=d, b>=l0, d>=l0
        sup = [(0, 0), (0, None), (None, 0), (1, None), (None, 1)]
        ords = [(3, 1, 1, 1, 0), (4, 0, 2, 0, 0)]
        return Family(name, sup, ords, [(1, 1, l0u), (2, 0, l0u)], Q)
    if name == 'F4':   # O=(0,0) B=(1,0) L=(0,b) T=(f,1); b >= l0
        sup = [(0, 0), (1, 0), (0, None), (None, 1)]
        return Family(name, sup, [], [(2, 1, l0u)], Q)
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------ P2
TERMS = [(+1, 0, 1), (-1, 0, 2), (-1, 1, 0), (+1, 1, 2), (+1, 2, 0), (-1, 2, 1)]   # D = sum sg * x_a * y_b


def p2_try(R, LO, HI, rq, rho, Q, lp_tol=-1e-9):
    """positive integer combination certificate (Panorias sec. 7.4; our P2 idea).  Returns True only after an
    exact integer check.  Variables s_v in [-1,1]: coordinate = (C_v + W_v s_v)/2."""
    from scipy.optimize import linprog
    detmask = (LO >= -rq) | (HI <= rq)
    idx = np.nonzero(detmask)[0]
    if len(idx) < 2:
        return False, -1
    sig = np.where(LO[idx] >= -rq, 1, -1).astype(np.int64)
    n = R.shape[0]
    C = np.empty(2 * n, np.int64); W = np.empty(2 * n, np.int64)
    C[0::2] = R[:, 0] + R[:, 1]; W[0::2] = R[:, 1] - R[:, 0]
    C[1::2] = R[:, 2] + R[:, 3]; W[1::2] = R[:, 3] - R[:, 2]
    m = len(idx)
    K = np.zeros(m, np.int64); A = np.zeros((m, 2 * n), np.int64)
    tri = TRI[idx]
    for sg, a, b in TERMS:
        pa = tri[:, a]; pb = tri[:, b]
        xa = 2 * pa; yb = 2 * pb + 1
        s = sig * sg
        K += s * C[xa] * C[yb] + np.abs(W[xa] * W[yb])
        np.add.at(A, (np.arange(m), xa), s * W[xa] * C[yb])
        np.add.at(A, (np.arange(m), yb), s * C[xa] * W[yb])
    # 4 sigma D_i <= K_i + A_i . s ; need sum lam (K + A.s) <= 4 rho Q^2 sum lam   for contradiction
    R4 = float(4 * rho * Q * Q)
    sc = 1.0 / (Q * Q)
    nv = 2 * n
    cols = [v for v in range(nv) if W[v] > 0]
    nc = len(cols)
    # vars: lam (m), t (nc)
    c = np.concatenate([(K - R4) * sc, np.ones(nc)])
    Af = A[:, cols].T.astype(float) * sc          # (nc, m)
    Aub = np.block([[Af, -np.eye(nc)], [-Af, -np.eye(nc)]])
    bub = np.zeros(2 * nc)
    Aeq = np.concatenate([np.ones(m), np.zeros(nc)])[None, :]
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[1.0], bounds=[(0, None)] * (m + nc), method='highs')
    if res.status != 0:
        return False, -1
    lam = res.x[:m]
    if res.fun > lp_tol:
        # branching hint (heuristic): coordinate with the largest share of the LP bound excess
        contrib = np.zeros(nv)
        contrib[cols] = np.abs(lam @ A[:, cols].astype(float)) * sc
        for sg, a, b in TERMS:
            q = np.abs(W[2 * tri[:, a]] * W[2 * tri[:, b] + 1]).astype(float) * sc * lam / 2
            np.add.at(contrib, 2 * tri[:, a], q); np.add.at(contrib, 2 * tri[:, b] + 1, q)
        contrib[W < 2] = -1
        return False, int(np.argmax(contrib)) if contrib.max() > 0 else -1
    lam_i = [int(round(x * 2 ** 30)) for x in lam]
    if sum(lam_i) <= 0 or min(lam_i) < 0:
        return False, -1
    # exact check in Python integers
    Wsum = sum(lam_i)
    Ctot = sum(l * int(K[i]) for i, l in enumerate(lam_i) if l)
    Atot = 0
    for v in cols:
        Atot += abs(sum(l * int(A[i, v]) for i, l in enumerate(lam_i) if l))
    return (Ctot + Atot) * rho.denominator <= 4 * rho.numerator * Q * Q * Wsum, -1


# ------------------------------------------------------------------------------------------------ cover (exp 44)
def make_cover_check(images, radius, Q):
    """images: list of exact point lists (Fractions); box closed if it lies in the sup-ball (radius) of some
    image under some bijection of labels (greedy nearest assignment, then exact check)."""
    fimgs = [np.array([[float(x), float(y)] for x, y in im]) for im in images]

    def covered(R):
        cx = (R[:, 0] + R[:, 1]) / (2 * Q); cy = (R[:, 2] + R[:, 3]) / (2 * Q)
        c = np.stack([cx, cy], 1)
        for E, F in zip(images, fimgs):
            d = np.abs(c[:, None, :] - F[None, :, :]).max(2)
            j = d.argmin(1)
            if len(set(j.tolist())) < len(j) or d[np.arange(len(j)), j].max() > float(radius):
                continue
            ok = True
            for i in range(len(j)):
                px, py = E[j[i]]
                for lo, hi, p in ((R[i, 0], R[i, 1], px), (R[i, 2], R[i, 3], py)):
                    if max(abs(Fr(int(lo), Q) - p), abs(Fr(int(hi), Q) - p)) > radius:
                        ok = False; break
                if not ok: break
            if ok: return True
        return False
    return covered


# ------------------------------------------------------------------------------------------------ local tree
def local_solve(R0, fam, rq, rho, Q, budget, stats, lp_w, covered=None, verbose=0, lp_d=0):
    """exhaustive interval tree on a full box (supports + free points in their cells).  Returns True if closed."""
    stack = [(R0.copy(), 0)]
    nodes = 0
    while stack:
        R, dep = stack.pop()
        nodes += 1
        if nodes > budget:
            stats['fail_nodes'] += nodes
            return False
        if not propagate(R, fam.ORD):
            stats['E'] += 1; continue
        LO, HI = all_ranges(R, TRI)
        if np.any((LO >= -rq) & (HI <= rq)):
            stats['T'] += 1; continue
        wmax = max(int((R[:, 1] - R[:, 0]).max()), int((R[:, 3] - R[:, 2]).max()))
        v = -1
        if wmax <= lp_w and dep >= lp_d:
            stats['lp_calls'] += 1
            ok, v = p2_try(R, LO, HI, rq, rho, Q)
            if ok:
                stats['P2'] += 1; continue
            if covered is not None and covered(R):
                stats['cover'] += 1; continue
        if v < 0:
            v = choose_split(R, LO, HI, TRI, rq)
        if v < 0:
            stats['fail_nodes'] += nodes
            return False
        p, c = divmod(v, 2)
        lo, hi = R[p, 2 * c], R[p, 2 * c + 1]
        mid = (lo + hi) // 2
        R1 = R.copy(); R1[p, 2 * c + 1] = mid
        R2 = R.copy(); R2[p, 2 * c] = mid          # closed halves share the endpoint
        stack.append((R1, dep + 1)); stack.append((R2, dep + 1))
        stats['splits'] += 1
        if verbose and nodes % verbose == 0:
            print('   local', nodes, {k: stats[k] for k in ('T', 'P2', 'lp_calls')}, 'stack', len(stack), flush=True)
    stats['local_nodes'] += nodes
    return True


# ------------------------------------------------------------------------------------------------ cliques
def k_cliques(adj, k, cap=None, CB=None, rq=0, stats=None):
    """all k-cliques (sorted vertex tuples) of a boolean adjacency matrix; bitset include/exclude recursion."""
    m = adj.shape[0]
    nb_ = [0] * m
    for u in range(m):
        row = np.nonzero(adj[u, u + 1:])[0] + u + 1
        b = 0
        for v in row.tolist():
            b |= 1 << v
        nb_[u] = b
    out = []
    selarr = np.zeros(k, np.int64)

    def rec(sel, cand, need):
        if need == 0:
            out.append(tuple(sel))
            if cap is not None and len(out) > cap:
                raise OverflowError
            return
        while cand:
            if bin(cand).count('1') < need:
                return
            low = cand & -cand
            v = low.bit_length() - 1
            cand ^= low
            if CB is not None and len(sel) >= 2:
                for z, u in enumerate(sel): selarr[z] = u
                if three_cell_blocked(CB, selarr, len(sel), v, rq):
                    if stats is not None: stats['tri3'] += 1
                    continue
            sel.append(v)
            rec(sel, cand & nb_[v], need - 1)
            sel.pop()
    try:
        rec([], (1 << m) - 1, k)
    except OverflowError:
        return None
    return out


# ------------------------------------------------------------------------------------------------ outer leaf
def solve_outer(S, fam, G, rq, rho, Q, cfg, stats, covered=None):
    """one support box: cell graph, clique enumeration, local trees.  Returns 'closed', or 'split' if the
    clique count or a local tree exceeded its budget (caller splits the support box)."""
    S = S.copy()
    if not propagate(S, fam.ORD):
        stats['outer_E'] += 1; return 'closed'
    k = fam.k
    # support-only triangles
    for t in itertools.combinations(range(k), 3):
        lo, hi = tri_range(S, t[0], t[1], t[2])
        if lo >= -rq and hi <= rq:
            stats['outer_T'] += 1; return 'closed'
    if int((S[:, 1::2] - S[:, 0::2]).max()) > cfg.get('max_sw', Q):
        return 'split'
    cs = Q // G
    excl = cell_excluded(S, G, cs, rq)
    cells = np.nonzero(~excl)[0].astype(np.int64)
    stats['cells_excl'] += int(excl.sum())
    if len(cells) < fam.m:
        stats['outer_deficit'] += 1; return 'closed'
    adj = edge_matrix(S, cells, G, cs, rq)
    stats['edges_missing'] += int(len(cells) * (len(cells) - 1) // 2 - adj.sum() // 2)
    # cheap clique-count bound before enumeration
    deg = adj.sum(1)
    if fam.m >= 2 and int(deg.sum()) // 2 > cfg['max_edges']:
        return 'split'
    CB = np.empty((len(cells), 4), np.int64)
    CB[:, 0] = (cells // G) * cs; CB[:, 1] = CB[:, 0] + cs; CB[:, 2] = (cells % G) * cs; CB[:, 3] = CB[:, 2] + cs
    cl = k_cliques(adj, fam.m, cfg['max_cliques'], CB if fam.m >= 3 else None, rq, stats)
    if cl is None:
        return 'split'
    stats['cliques'] += len(cl)
    for c in cl:
        R = np.zeros((N, 4), np.int64)
        R[:k] = S
        for i, u in enumerate(c):
            cc = int(cells[u])
            R[k + i] = ((cc // G) * cs, (cc // G + 1) * cs, (cc % G) * cs, (cc % G + 1) * cs)
        if not local_solve(R, fam, rq, rho, Q, cfg['local_budget'], stats, cfg['lp_w'], covered, cfg.get('verbose', 0), cfg.get('lp_d', 0)):
            stats['local_fail'] += 1
            return 'split'
    return 'closed'


def new_stats():
    return {k: 0 for k in ('outer_leaves', 'outer_splits', 'outer_E', 'outer_T', 'outer_deficit', 'cells_excl',
                           'edges_missing', 'cliques', 'local_nodes', 'T', 'E', 'P2', 'lp_calls', 'cover', 'splits',
                           'fail_nodes', 'local_fail', 'unresolved', 'tri3')}


def outer_tree(fam, G, rq, rho, Q, cfg, stats, S0=None, covered=None, min_w=None, t_end=None):
    """outer partition of the support box; closed halves at every split.  Returns list of unresolved boxes."""
    stack = [fam.root() if S0 is None else S0]
    unresolved = []
    min_w = min_w if min_w is not None else Q // 4096
    while stack:
        S = stack.pop()
        if t_end is not None and time.time() > t_end:
            unresolved.append(S); unresolved.extend(stack); stats['unresolved'] += 1 + len(stack); break
        r = solve_outer(S, fam, G, rq, rho, Q, cfg, stats, covered)
        if r == 'closed':
            stats['outer_leaves'] += 1; continue
        w = S[:, 1::2] - S[:, 0::2]          # widths: cols (x, y)
        p, c = np.unravel_index(int(np.argmax(w)), w.shape)
        if w[p, c] <= min_w:
            unresolved.append(S); stats['unresolved'] += 1; continue
        lo, hi = S[p, 2 * c], S[p, 2 * c + 1]; mid = (lo + hi) // 2
        S1 = S.copy(); S1[p, 2 * c + 1] = mid
        S2 = S.copy(); S2[p, 2 * c] = mid
        stack += [S1, S2]; stats['outer_splits'] += 1
    return unresolved
