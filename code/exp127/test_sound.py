"""randomised implementation tests (not part of the proof).
1. range3/all_ranges vs brute force: random rect (clipped to x+y<=Q) / hypotenuse-segment domains; the exact
   range must contain D at random exact domain samples and be attained at sampled vertices of the domains.
2. propagate never removes a sampled domain point satisfying the constraints.
3. P2 never closes a box containing a sampled point with all |D| > rho (n = 9, random boxes in Omega).
4. cover: sampled points of covered boxes map back within R of x* (exact)."""
import itertools, random, sys, os
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cgt

random.seed(7)
Q = 64


def D(P, i, j, k):
    return (P[j][0] - P[i][0]) * (P[k][1] - P[i][1]) - (P[k][0] - P[i][0]) * (P[j][1] - P[i][1])


def rdom():
    kind = random.random() < 0.3
    a, b = sorted(random.randint(0, Q) for _ in range(2))
    if kind:
        return [a, b, Q - b, Q - a], 1
    while True:
        c, d = sorted(random.randint(0, Q) for _ in range(2))
        if random.random() < 0.2: b = a
        if random.random() < 0.2: d = c
        if a + c <= Q: return [a, min(b, Q - c), c, min(d, Q - a)], 0


def sample(r, k, den=8):
    while True:
        if k == 1:
            s = Fr(random.randint(r[0] * den, r[1] * den), den); return (s, Q - s)
        x = Fr(random.randint(r[0] * den, r[1] * den), den); y = Fr(random.randint(r[2] * den, r[3] * den), den)
        if x + y <= Q: return (x, y)


bad = 0; tight = 0
for _ in range(3000):
    doms = [rdom() for _ in range(3)]
    R = np.array([d[0] for d in doms], np.int64); K = np.array([d[1] for d in doms], np.int64)
    lo, hi = cgt.range3(R.copy(), K, Q)
    VX = np.empty((3, 5), np.int64); VY = np.empty((3, 5), np.int64); NV = np.empty(3, np.int64)
    cgt.build_verts(R, K, Q, VX, VY, NV)
    vals = []
    for vs in itertools.product(*[range(NV[p]) for p in range(3)]):
        P = [(int(VX[p, vs[p]]), int(VY[p, vs[p]])) for p in range(3)]
        assert all(x >= 0 and y >= 0 and x + y <= Q for x, y in P)
        for p in range(3):
            r = R[p]; assert r[0] <= P[p][0] <= r[1] and r[2] <= P[p][1] <= r[3]
        vals.append(D(P, 0, 1, 2))
    if min(vals) != lo or max(vals) != hi: bad += 1
    for s in range(30):
        P = [sample(R[p], K[p]) for p in range(3)]
        v = D(P, 0, 1, 2)
        if not lo <= v <= hi: bad += 1
print('range mismatches/violations:', bad)

# propagate soundness on SSS0 / MMS families (orders, sums, x+y<=Q)
cgt.configure(9)
viol = 0
for name in ('SSS0', 'MMS', 'SSS2', 'MSS0'):
    fam = cgt.make_family(name, 64, 10)
    for _ in range(500):
        S = fam.root()
        for p in range(fam.k):
            for c in range(2):
                if fam.SSPL[p, c]:
                    a, b = sorted(random.randint(S[p, 2 * c], S[p, 2 * c + 1]) for _ in range(2))
                    S[p, 2 * c], S[p, 2 * c + 1] = a, b
            if fam.SK[p] == 1: S[p, 2], S[p, 3] = 64 - S[p, 1], 64 - S[p, 0]
        S2 = S.copy(); ok = cgt.propagate(S2, fam.SK, fam.ORD, fam.SUM, 64)
        for s in range(40):
            P = [sample(S[p], fam.SK[p]) for p in range(fam.k)]
            co = lambda p, c: P[p][c]
            if all(co(a, c1) + g <= co(b, c2) for a, c1, b, c2, g in fam.ORD) and \
               all(co(a, c1) + co(b, c2) <= 64 for a, c1, b, c2 in fam.SUM):
                if not ok or any(not (S2[p, 0] <= P[p][0] <= S2[p, 1] and S2[p, 2] <= P[p][1] <= S2[p, 3]) for p in range(fam.k)):
                    viol += 1
print('propagate violations:', viol)

# P2 random boxes, n = 9
Q = 64
rho = Fr(1, 40)
rq = int(rho * Q * Q)
viol = 0; certs = 0; tried = 0
for it in range(1500):
    P0 = []
    while len(P0) < 9:
        x, y = random.random(), random.random()
        if x + y <= 1: P0.append((x, y))
    w = random.choice([1, 2, 3, 4])
    R = []; K = []
    for idx, (x, y) in enumerate(P0):
        if idx == 0:                               # one hypotenuse point
            s = int(x * Q); R.append([max(0, s - w), min(Q, s + w), 0, 0]); R[-1][2:] = [Q - R[-1][1], Q - R[-1][0]]; K.append(1)
        else:
            a, b = max(0, int(x * Q) - w), min(Q, int(x * Q) + w); c, d = max(0, int(y * Q) - w), min(Q, int(y * Q) + w)
            if a + c > Q: continue
            R.append([a, b, c, d]); K.append(0)
    if len(R) < 9: continue
    R = np.array(R, np.int64); K = np.array(K, np.int64)
    if not cgt.propagate(R, K, np.zeros((0, 5), np.int64), np.zeros((0, 4), np.int64), Q): continue
    LO, HI = cgt.all_ranges(R, K, Q, cgt.TRI)
    if np.any((LO >= -rq) & (HI <= rq)): continue
    tried += 1
    ok = cgt.p2_try(R, K, LO, HI, rq, rho, Q, cgt.TRI)[0]
    if ok:
        certs += 1
        for s in range(400):
            P = [sample(R[p], K[p]) for p in range(9)]
            if min(abs(D(P, *t)) for t in itertools.combinations(range(9), 3)) > rho * Q * Q: viol += 1; break
print('p2 boxes tried', tried, 'certificates:', certs, 'violations:', viol)

# cover: n = 9 certificate, boxes around images, sampled points must map back within the basin radius
import run
P, Rc, tlo, thi = run.load_cert(9)
Qc = 37 * 2 ** 14
cov = cgt.make_cover(P, Rc, Qc)
nc = 0; bad = 0
for g in cgt.s3_maps():
    (a, b), (c, d) = g[0]; o = g[1]
    img = [(a * x + b * y + o[0], c * x + d * y + o[1]) for x, y in P]
    Mi, ci = cgt._inv(g)
    for it in range(40):
        hw = random.choice([1, 4, 16, 32, 64])
        R = np.array([[int(x * Qc) - random.randint(0, hw), int(x * Qc) + random.randint(1, hw),
                       int(y * Qc) - random.randint(0, hw), int(y * Qc) + random.randint(1, hw)] for x, y in img], np.int64)
        R = np.clip(R, 0, Qc); K = np.zeros(9, np.int64)
        perm = list(range(9)); random.shuffle(perm); R = R[perm]
        hit = cov(R, K, want=True)
        if hit:
            (Mi, ci), jj = hit
            nc += 1
            for s in range(50):
                pts = [(Fr(random.randint(R[p, 0], R[p, 1]), Qc), Fr(random.randint(R[p, 2], R[p, 3]), Qc)) for p in range(9)]
                if any(x + y > 1 for x, y in pts): continue
                back = [(Mi[0][0] * x + Mi[0][1] * y + ci[0], Mi[1][0] * x + Mi[1][1] * y + ci[1]) for x, y in pts]
                dist = min(max(max(abs(back[i][0] - P[pi][0]), abs(back[i][1] - P[pi][1])) for i, pi in enumerate(pm))
                           for pm in [jj])
                if dist > Rc: bad += 1
print('cover: covered boxes', nc, 'violations', bad, 'R =', float(Rc))

# P2 cell exclusion: random SSS1 support boxes + one cell; if excluded, no sampled (supports, cell point) may have all
# triples |D| > rho
cgt.configure(9)
Qs = 37 * 2 ** 14; Ts = Fr(43, 1568) * Fr(1001, 1000); rhos = 2 * Ts; rqs = int(rhos * Qs * Qs); cs = Qs // 37
fam = cgt.make_family('SSS1', Qs, int(0.22 * Qs))
excl = 0; viol = 0; tried = 0
for it in range(600):
    S = fam.root()
    for p in range(fam.k):
        for c in range(2):
            if fam.SSPL[p, c]:
                w = Qs // random.choice([64, 128, 256])
                a = random.randint(S[p, 2 * c], S[p, 2 * c + 1] - w); S[p, 2 * c], S[p, 2 * c + 1] = a, a + w
        if fam.SK[p] == 1: S[p, 2], S[p, 3] = Qs - S[p, 1], Qs - S[p, 0]
    if not cgt.propagate(S, fam.SK, fam.ORD, fam.SUM, Qs): continue
    cx = random.randint(0, 36); cy = random.randint(0, 36 - cx)
    CB = np.array([[cx * cs, (cx + 1) * cs, cy * cs, (cy + 1) * cs]], np.int64)
    tried += 1
    CB2, _ = cgt.p2_cell_prune(S, fam, CB, np.zeros((1, 1), bool), Qs, rqs, rhos, 8, {})
    if len(CB2) == 0:
        excl += 1
        Kp = list(fam.SK) + [0]; Rp = np.vstack([S, CB])
        for s_ in range(300):
            P = []
            for p in range(fam.k + 1):
                while True:
                    if Kp[p] == 1:
                        v = Fr(random.randint(Rp[p, 0], Rp[p, 1])); q = (v, Qs - v)
                    else:
                        q = (Fr(random.randint(Rp[p, 0], Rp[p, 1])), Fr(random.randint(Rp[p, 2], Rp[p, 3])))
                    if q[0] + q[1] <= Qs: break
                P.append(q)
            if min(abs(D(P, *t3)) for t3 in itertools.combinations(range(fam.k + 1), 3)) > rhos * Qs * Qs: viol += 1; break
print('cell-P2: boxes', tried, 'excluded', excl, 'violations', viol)
