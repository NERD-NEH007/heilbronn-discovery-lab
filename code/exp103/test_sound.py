"""Randomised soundness test of the exact pruning bounds (not part of the proof; a check of the implementation).
For random boxes: (1) sampled points never beat the certified LP bound on min over determined triples of sigma*A;
(2) sampled points with all |A| > T respect the determined signs; (3) the integer certificate agrees with an
independent Fraction recomputation of the triple corner ranges.
    python code/exp103/test_sound.py [n]"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hlib
from fractions import Fraction as Fr
import numpy as np
import bb
rng = np.random.default_rng(1)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
import json
OPT = hlib.small_n_optima()[str(n)][0]
X0 = np.array(OPT['points']); X0 = X0[np.argsort(X0[:, 0])]; x0 = np.concatenate([X0[:, 0], X0[:, 1]])
P = bb.Problem(n, Fr(float(np.float64(0.8 * OPT['value']))))
bad = 0; tested = 0; gaps = []
for trial in range(400):
    w = int(bb.D * 10 ** rng.uniform(-4, -0.7))
    c = np.clip(((x0 + rng.normal(0, 10 ** rng.uniform(-4, -1.3), 2 * n)) * bb.D).astype(np.int64), 0, bb.D)
    lo = np.clip(c - rng.integers(0, w + 1, 2 * n), 0, bb.D).astype(np.int64); hi = np.clip(lo + w, 0, bb.D).astype(np.int64)
    if not P.propagate(lo, hi): continue
    A2lo, A2hi = P.triple_ranges(lo, hi)
    # independent exact corner check for a few triples
    for t in rng.integers(0, P.m, 3):
        i, j, k = P.tri[t]
        vals = []
        for ci in itertools.product((0, 1), repeat=2):
            for cj in itertools.product((0, 1), repeat=2):
                for ck in itertools.product((0, 1), repeat=2):
                    pts = []
                    for q, cq in ((i, ci), (j, cj), (k, ck)):
                        pts.append((Fr(int((lo, hi)[cq[0]][q]), bb.D), Fr(int((lo, hi)[cq[1]][n + q]), bb.D)))
                    (x1, y1), (x2, y2), (x3, y3) = pts
                    vals.append((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        assert Fr(A2lo[t]) == min(vals) and Fr(A2hi[t]) == max(vals), 'corner range mismatch'
    T2 = P.T2f
    if (np.maximum(A2hi, -A2lo) <= T2).any(): continue
    sig = np.where(A2lo >= -T2, 1, np.where(A2hi <= T2, -1, 0)).astype(np.int64)
    if not (sig != 0).any(): continue
    ok, val, info = P.lp_bound(lo, hi, sig, force=True)
    if info is None or 'bound' not in info: continue
    tested += 1
    X = lo / bb.D + rng.random((4000, 2 * n)) * (hi - lo) / bb.D
    X = X[(np.diff(X[:, :n], axis=1) >= 0).all(1)]            # the LP includes the ordering x_i <= x_{i+1}
    if len(X) == 0: continue
    x, y = X[:, :n], X[:, n:]
    I, J, K = P.tri.T
    A2 = (x[:, J] - x[:, I]) * (y[:, K] - y[:, I]) - (x[:, K] - x[:, I]) * (y[:, J] - y[:, I])
    det = sig != 0
    m = (sig[det] * A2[:, det]).min(1)
    if (m > 2 * info['bound'] + 1e-12).any(): bad += 1; print('BOUND VIOLATED', trial)
    allbig = (np.abs(A2) > T2).all(1)
    if allbig.any() and ((np.sign(A2[allbig][:, det]) != sig[det]).any()): bad += 1; print('SIGN VIOLATED', trial)
    gaps.append(2 * info['bound'] - m.max())
print(f'n={n}: tested {tested} LP boxes, violations {bad}; LP bound - best sample (2A units): median {np.median(gaps):.3g}, min {min(gaps):.3g}')
