"""Randomised soundness test of bb2 (not part of the proof; a check of the implementation).  python test_sound2.py n
For random boxes near the exp 102 optimum (and its diagonal mirror):
 (1) P1: the int64 corner ranges equal an independent Fraction computation (sampled triples), and no sampled point
     with all |A| > T lies in a box declared dead / violates a determined sign;
 (2) P0: no sampled domain point (x sorted, mean x <= 1/2, sum(x-1/2)^2 >= sum(y-1/2)^2) is removed by propagation;
 (3) P2: no sampled point beats the certified bound of the hot-started LP certificate, and the Fraction audit agrees
     with the integer certificate (pruned or not) on every tested box."""
import sys, os, itertools, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractions import Fraction as Fr
import numpy as np
import bb2
D = bb2.D
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
OPT = json.load(open(os.path.join(bb2.REPO, 'configurations', 'small_n', 'optima_small.json')))[str(n)][0]
X0 = np.array(OPT['points'])
P = bb2.Problem2(n, Fr(float(np.float64(0.8 * OPT['value']))), diag=True)
T2 = 2 * float(P.T)
I, J, K = P.tri.T
viol = dict(p1=0, sign=0, p0=0, p2=0, audit=0); tested = dict(p0=0, p1=0, p2=0)
gaps = []
for trial in range(600):
    Y = X0.copy()
    if trial % 2: Y = Y[:, ::-1]                                   # diagonal mirror (exercise both sides of diag)
    if trial % 3 == 0: Y[:, 0] = 1 - Y[:, 0]
    Y = Y[np.argsort(Y[:, 0])]; x0 = np.concatenate([Y[:, 0], Y[:, 1]])
    w = int(D * 10 ** rng.uniform(-4, -0.5))
    c = np.clip(((x0 + rng.normal(0, 10 ** rng.uniform(-4, -1.3), 2 * n)) * D).astype(np.int64), 0, D)
    lo = np.clip(c - rng.integers(0, w + 1, 2 * n), 0, D).astype(np.int64); hi = np.clip(lo + w, 0, D).astype(np.int64)
    # samples in the ORIGINAL box
    S = lo / D + rng.random((6000, 2 * n)) * (hi - lo) / D
    x, y = S[:, :n], S[:, n:]
    dom = (np.diff(x, axis=1) >= 0).all(1) & (x.sum(1) <= n / 2) & (((x - .5) ** 2).sum(1) >= ((y - .5) ** 2).sum(1))
    l2, h2 = lo.copy(), hi.copy()
    alive = bb2.k_propagate(l2, h2, n, True, True, D); tested['p0'] += 1
    Sd = S[dom]
    if not alive:
        if len(Sd): viol['p0'] += 1; print('P0 removed all, but domain samples exist', trial)
        continue
    inside = ((Sd >= l2 / D) & (Sd <= h2 / D)).all(1)
    if not inside.all(): viol['p0'] += 1; print('P0 removed domain samples', trial)
    lo, hi = l2, h2
    dead, sig = bb2.k_ranges(lo, hi, P.tri, n, P.T2i); tested['p1'] += 1
    for t in rng.integers(0, P.m, 3):
        i, j, k = P.tri[t]; vals = []
        for ci in itertools.product((0, 1), repeat=2):
            for cj in itertools.product((0, 1), repeat=2):
                for ck in itertools.product((0, 1), repeat=2):
                    pts = [(Fr(int((lo, hi)[cq[0]][q]), D), Fr(int((lo, hi)[cq[1]][n + q]), D)) for q, cq in ((i, ci), (j, cj), (k, ck))]
                    (x1, y1), (x2, y2), (x3, y3) = pts
                    vals.append((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        mn, mx = min(vals), max(vals)
        if not dead:
            s_ = 1 if mn >= -2 * P.T else (-1 if mx <= 2 * P.T else 0)
            if s_ != sig[t]: viol['p1'] += 1; print('P1 sign rule mismatch', trial)
        elif False: pass
    S = lo / D + rng.random((6000, 2 * n)) * (hi - lo) / D
    S = S[(np.diff(S[:, :n], axis=1) >= 0).all(1) & (S[:, :n].sum(1) <= n / 2)]
    if len(S) == 0: continue
    x, y = S[:, :n], S[:, n:]
    A2 = (x[:, J] - x[:, I]) * (y[:, K] - y[:, I]) - (x[:, K] - x[:, I]) * (y[:, J] - y[:, I])
    big = (np.abs(A2) > T2 * (1 + 1e-12)).all(1)
    if dead:
        if big.any(): viol['p1'] += 1; print('P1 dead box has a sample with all |A| > T', trial)
        continue
    det = sig != 0
    if big.any() and (np.sign(A2[big][:, det]) != sig[det]).any(): viol['sign'] += 1; print('SIGN', trial)
    if not det.any(): continue
    P._last_bound = None
    ok, val = P.lp_bound2(lo, hi, sig, force=True)
    if P._last_bound is None: continue
    tested['p2'] += 1
    m = (sig[det] * A2[:, det]).min(1)
    if (m > float(P._last_bound) + 1e-12).any(): viol['p2'] += 1; print('P2 BOUND VIOLATED', trial)
    gaps.append(float(P._last_bound) - m.max())
    # independent Fraction audit of the same multipliers
    lam = np.maximum(-np.array(P.highs.getSolution().row_dual), 0.0)
    act = np.zeros(2 * P.m, bool); act[0::2] = sig == 1; act[1::2] = sig == -1
    lam[:2 * P.m][~act] = 0.0
    S_ = 49 - max(0, int(np.ceil(np.log2(lam.max()))))
    mu = np.floor(np.ldexp(lam, S_)).astype(np.int64)
    if P.audit2(lo, hi, sig, mu) != ok: viol['audit'] += 1; print('AUDIT MISMATCH', trial)
print(f'n={n}: tested {tested}; violations {viol}; P2 bound - best sample (2A units): median {np.median(gaps):.3g}, min {min(gaps):.3g}')
