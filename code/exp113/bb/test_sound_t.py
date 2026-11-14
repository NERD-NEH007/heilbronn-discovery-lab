"""Randomised soundness test of bbt (not part of the proof; a check of the implementation).  python test_sound_t.py n [seed]
Boxes near the exp 105 optimum and its 6 S3 images (x-sorted), random hypotenuse label c:
 (0) P0: no domain point is removed.  Domain samples are exact integers on the grid 1/(K D) (K = 16): x sorted,
     x_i + y_i <= 1, y_c := 1 - x_c (equality), A >= B >= C; all checks in int64.
 (1) P1: int64 corner ranges equal an independent Fraction computation (sampled triples); a dead box has no sample with
     all |A| > T; samples with all |A| > T respect every determined sign.
 (2) P2: no sample beats the certified bound; the Fraction audit agrees with the integer certificate on every box.
 (3) cover: on random tiny boxes around S3 images, every 'covered' box has all sampled points within R of x~ after g^-1.
"""
import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import numpy as np
import bbt
D = bbt.D; K = 16
n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
X0 = np.array(json.load(open(os.path.join(bbt.POOL, f'pool_tri_n{n}.json')))[0]['points'])
flo, fhi = bbt.fstar_bounds(n)
P = bbt.ProblemT(n, Fr(float(np.float64(0.8 * float(flo)))))
T2 = 2 * float(P.T)
I, J, Kk = P.tri.T
viol = dict(p0=0, p1=0, sign=0, p2=0, audit=0, cover=0); tested = dict(p0=0, p1=0, p2=0, p2_farkas=0, cover=0)
gaps = []


def domain_int(S, c):
    """S: (N, 2n) int64 on grid K D.  Exact domain test (with y_c already set to KD - x_c)."""
    x, y = S[:, :n], S[:, n:]; KD = K * D
    ok = (np.diff(x, axis=1) >= 0).all(1) & (x >= 0).all(1) & (y >= 0).all(1) & ((x + y) <= KD).all(1)
    ok &= x[:, c] + y[:, c] == KD
    A = x.sum(1); B = y.sum(1); C = n * KD - A - B
    return ok & (A >= B) & (B >= C)


for trial in range(600):
    g = bbt.PERMS[trial % 6]
    Y = np.array([[float(v) for v in bbt.s3_apply(g, (Fr(p[0]), Fr(p[1])))] for p in X0])
    Y = Y[np.argsort(Y[:, 0])]; x0 = np.concatenate([Y[:, 0], Y[:, 1]])
    c = int(np.argmax(Y[:, 0] + Y[:, 1])) if trial % 3 else int(rng.integers(n))
    w = int(D * 10 ** rng.uniform(-4, -0.5))
    cen = np.clip(((x0 + rng.normal(0, 10 ** rng.uniform(-4, -1.3), 2 * n)) * D).astype(np.int64), 0, D)
    lo = np.clip(cen - rng.integers(0, w + 1, 2 * n), 0, D).astype(np.int64); hi = np.clip(lo + w, 0, D).astype(np.int64)
    # (0) P0 on exact integer samples of the ORIGINAL box
    S = (lo * K)[None, :] + (rng.random((20000, 2 * n)) * ((hi - lo) * K)[None, :]).astype(np.int64)
    S[:, n + c] = K * D - S[:, c]
    S = S[(S[:, n + c] >= lo[n + c] * K) & (S[:, n + c] <= hi[n + c] * K)]
    Sd = S[domain_int(S, c)] if len(S) else S
    l2, h2 = lo.copy(), hi.copy()
    alive = bbt.k_propagate(l2, h2, n, c, D); tested['p0'] += 1
    if not alive:
        if len(Sd): viol['p0'] += 1; print('P0 declared empty, but domain samples exist', trial)
        continue
    if len(Sd) and not ((Sd >= l2 * K) & (Sd <= h2 * K)).all(): viol['p0'] += 1; print('P0 removed domain samples', trial)
    lo, hi = l2, h2
    dead, sig = bbt.k_ranges(lo, hi, P.tri, n, P.T2i); tested['p1'] += 1
    for t in rng.integers(0, P.m, 3):
        i, j, k = P.tri[t]; vals = []
        for ci in itertools.product((0, 1), repeat=2):
            for cj in itertools.product((0, 1), repeat=2):
                for ck in itertools.product((0, 1), repeat=2):
                    pts = [(Fr(int((lo, hi)[cq[0]][q]), D), Fr(int((lo, hi)[cq[1]][n + q]), D)) for q, cq in ((i, ci), (j, cj), (k, ck))]
                    (x1, y1), (x2, y2), (x3, y3) = pts
                    vals.append((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        mn, mx = min(vals), max(vals)
        if max(mx, -mn) <= 2 * P.T and not dead: viol['p1'] += 1; print('P1 missed dead triple', trial)
        if not dead:
            s_ = 1 if mn >= -2 * P.T else (-1 if mx <= 2 * P.T else 0)
            if s_ != sig[t]: viol['p1'] += 1; print('P1 sign rule mismatch', trial)
    # float samples in the propagated box (domain up to float rounding) for P1 / P2 checks
    Sf = lo / D + rng.random((6000, 2 * n)) * (hi - lo) / D
    Sf[:, n + c] = 1 - Sf[:, c]
    x, y = Sf[:, :n], Sf[:, n:]
    dom = (np.diff(x, axis=1) >= 0).all(1) & ((x + y) <= 1 + 1e-15).all(1) & (y[:, c] >= lo[n + c] / D) & (y[:, c] <= hi[n + c] / D)
    dom &= (x.sum(1) >= y.sum(1)) & (y.sum(1) >= n - x.sum(1) - y.sum(1))
    Sf = Sf[dom]
    if len(Sf) == 0: continue
    x, y = Sf[:, :n], Sf[:, n:]
    A2 = (x[:, J] - x[:, I]) * (y[:, Kk] - y[:, I]) - (x[:, Kk] - x[:, I]) * (y[:, J] - y[:, I])
    big = (np.abs(A2) > T2 * (1 + 1e-12)).all(1)
    if dead:
        if big.any(): viol['p1'] += 1; print('P1 dead box has a sample with all |A| > T', trial)
        continue
    det = sig != 0
    if big.any() and (np.sign(A2[big][:, det]) != sig[det]).any(): viol['sign'] += 1; print('SIGN', trial)
    if not det.any(): continue
    P._last_bound = None
    ok, val = P.lp_bound2(lo, hi, sig, c, force=True)
    if P._last_bound is None: continue
    tested['p2'] += 1
    m = (sig[det] * A2[:, det]).min(1)
    if (m > float(P._last_bound) + 1e-12).any(): viol['p2'] += 1; print('P2 BOUND VIOLATED', trial)
    gaps.append(float(P._last_bound) - m.max())
    if P.audit2(lo, hi, sig, c, P._last_mu) != ok: viol['audit'] += 1; print('AUDIT MISMATCH', trial)

# (3) cover test (exact-mode covers if the certificate exists)
covs = bbt.load_covers(n)
if covs:
    Pc = bbt.ProblemT(n, P.T, covers=covs, cover_width=D // 1000); cv = covs[0]; R = float(cv['R'])
    for trial in range(300):
        gi = trial % 6; E = np.array(cv['fimgs'][gi]); perm = np.argsort(E[:, 0], kind='stable'); E = E[perm]
        w = int(D * R * 10 ** rng.uniform(-0.3, 0.6))
        cen = (np.concatenate([E[:, 0], E[:, 1]]) * D).astype(np.int64) + rng.integers(-w, w + 1, 2 * n)
        lo = np.clip(cen - w // 2, 0, D); hi = np.clip(lo + w // 2, 0, D)
        if not Pc.covered(lo, hi): continue
        tested['cover'] += 1
        Sf = lo / D + rng.random((2000, 2 * n)) * (hi - lo) / D
        g = bbt.PERMS[gi]; X = np.array([[float(a), float(b)] for a, b in cv['pts']])
        best = 0.0
        for s in Sf[:20]:                          # worst sample of: best symmetry of (max over points of nearest distance)
            ds = np.inf
            for gg in bbt.PERMS:
                pts = np.array([[float(v) for v in bbt.s3_inverse_apply(gg, (s[i], s[n + i]))] for i in range(n)])
                d = np.abs(pts[:, None, :] - X[None, :, :]).max(2)
                ds = min(ds, d.min(1).max())
            best = max(best, ds)
        if best > R * (1 + 1e-9): viol['cover'] += 1; print('COVER: sampled point outside the ball', trial, best, R)
print(f'n={n}: tested {tested}; violations {viol}; P2 bound - best sample (2A units): '
      f'median {np.median(gaps) if gaps else float("nan"):.3g}, min {min(gaps) if gaps else float("nan"):.3g}')
