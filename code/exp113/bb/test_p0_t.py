"""Random P0 test (implementation check): random boxes anywhere in [0,1]^(2n) for random c; exact integer domain samples
(grid 1/(16 D), y_c = 1 - x_c) must never be removed, and a box declared empty must contain no sample.
Domain samples are built directly (sorted x, then y_i in [0, 1 - x_i]) and boxes are drawn around them so that many
boxes are non-empty.  python test_p0_t.py n boxes [seed]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import bbt
D = bbt.D; K = 16; KD = K * D
n = int(sys.argv[1]); NB = int(sys.argv[2]); rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
kept = removed = empt_bad = alive_boxes = 0
for b in range(NB):
    c = int(rng.integers(n))
    x = np.sort(rng.integers(0, KD + 1, n)); y = (rng.random(n) * (KD - x)).astype(np.int64); y[c] = KD - x[c]
    p = np.concatenate([x, y]) // K
    w = (D * 10 ** rng.uniform(-3, 0, 2 * n)).astype(np.int64)
    lo = np.clip(p - (rng.random(2 * n) * w).astype(np.int64), 0, D); hi = np.clip(lo + w, 0, D)
    S = (lo * K)[None, :] + (rng.random((300, 2 * n)) * ((hi - lo) * K)[None, :]).astype(np.int64)
    S[:, n + c] = KD - S[:, c]
    xs, ys = S[:, :n], S[:, n:]
    ok = (np.diff(xs, axis=1) >= 0).all(1) & (ys >= 0).all(1) & ((xs + ys) <= KD).all(1)
    ok &= (S[:, n + c] >= lo[n + c] * K) & (S[:, n + c] <= hi[n + c] * K)
    A = xs.sum(1); B = ys.sum(1); C = n * KD - A - B; ok &= (A >= B) & (B >= C)
    Sd = S[ok]
    l2, h2 = lo.copy(), hi.copy(); alive = bbt.k_propagate(l2, h2, n, c, D)
    if not alive:
        empt_bad += len(Sd) > 0; continue
    alive_boxes += 1
    ins = ((Sd >= l2 * K) & (Sd <= h2 * K)).all(1); kept += int(ins.sum()); removed += int((~ins).sum())
print(dict(n=n, boxes=NB, alive=alive_boxes, domain_samples_kept=kept, removed=removed, empty_but_samples=empt_bad))
