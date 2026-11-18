"""negative test (not part of the proof): T = t_lo (1 - 1e-6) < t*, no cover.  No box containing a labelled
family image of the optimum may be closed by the E, T or P2 rules.  python test_neg.py n FAM"""
import random, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cgt, run


def images_in_family(P, fam_name):
    out = []
    for (M, o) in cgt.s3_maps():
        I = [(M[0][0] * x + M[0][1] * y + o[0], M[1][0] * x + M[1][1] * y + o[1]) for x, y in P]
        on = lambda f: [i for i, p in enumerate(I) if f(p)]
        Bs = sorted(on(lambda p: p[1] == 0 and p[0] != 0), key=lambda i: I[i][0])
        Ls = sorted(on(lambda p: p[0] == 0 and p[1] != 0), key=lambda i: I[i][1])
        Hs = sorted(on(lambda p: p[0] + p[1] == 1 and p[1] != 0 and p[0] != 0), key=lambda i: I[i][0])
        Os = on(lambda p: p == (0, 0))
        if fam_name == 'SSS0' and len(Bs) == len(Ls) == len(Hs) == 2 and not Os:
            sup = [Bs[0], Bs[1], Ls[0], Ls[1], Hs[0], Hs[1]]
        elif fam_name == 'SSS1' and Os and len(Bs) == len(Ls) == 1 and len(Hs) == 2:
            sup = [Os[0], Bs[0], Ls[0], Hs[0], Hs[1]]
        else:
            continue
        out.append([I[i] for i in sup] + [I[i] for i in range(len(P)) if i not in sup])
    return out


n, fam_name = int(sys.argv[1]), sys.argv[2]
fam, T, rho, rq, Q, cfg, cov, l0 = run.setup(n, fam_name, 'neg', 1e-6, 37 if n == 9 else 30)
P, Rc, tlo, thi = run.load_cert(n)
ims = images_in_family(P, fam_name)
random.seed(3); bad = 0; tested = 0; skipped = 0
def satisfies(I):
    co = lambda p, c: I[p][c] * Q
    return all(co(a, c1) + g <= co(b, c2) for a, c1, b, c2, g in fam.ORD) and all(co(a, c1) + co(b, c2) <= Q for a, c1, b, c2 in fam.SUM)
ims = [I for I in ims if satisfies(I)]
for I in ims:
    for d in range(4, 16):
        for rep in range(3):
            h = Q >> d
            R = np.zeros((n, 4), np.int64)
            for p, (x, y) in enumerate(I):
                for c, v in enumerate((x, y)):
                    u = int(v * Q)
                    fixed = p < fam.k and not fam.SSPL[p, c]
                    if fixed and fam.SK[p] == 0:
                        lo = hi = u
                    else:
                        lo = max(0, u - random.randint(0, h)); hi = min(Q, u + 1 + random.randint(0, h))
                    R[p, 2 * c] = lo; R[p, 2 * c + 1] = hi
                if p < fam.k and fam.SK[p] == 1:
                    R[p, 2], R[p, 3] = Q - R[p, 1], Q - R[p, 0]
            S = R[:fam.k].copy()
            if not cgt.propagate(S, fam.SK, fam.ORD, fam.SUM, Q):
                skipped += 1; continue          # image violates this family's symmetry-breaking rows
            Rp = R.copy()
            if not cgt.propagate(Rp, fam.K, fam.ORD, fam.SUM, Q):
                print('E closed', d); bad += 1; continue              # E rule closed a box containing the optimum
            LO, HI = cgt.all_ranges(Rp, fam.K, Q, cgt.TRI); tested += 1
            tr = np.any((LO >= -rq) & (HI <= rq))
            if tr or cgt.p2_try(Rp, fam.K, LO, HI, rq, rho, Q, cgt.TRI)[0]:
                bad += 1; print('closed d', d, 'T-rule' if tr else 'P2', cgt.TRI[np.nonzero((LO >= -rq) & (HI <= rq))[0]].tolist() if tr else '')
# P2 cell exclusion (CELL_P2 flag) must keep every cell holding a free point of an optimum image
G = 37 if n == 9 else 30; cs = Q // G; cbad = 0; ctested = 0
for I in ims:
    for d in range(4, 16):
        h = Q >> d
        S = np.zeros((fam.k, 4), np.int64)
        for p, (x, y) in enumerate(I[:fam.k]):
            for c, v in enumerate((x, y)):
                u = int(v * Q)
                if not fam.SSPL[p, c] and fam.SK[p] == 0: lo = hi = u
                else: lo = max(0, u - random.randint(0, h)); hi = min(Q, u + 1 + random.randint(0, h))
                S[p, 2 * c] = lo; S[p, 2 * c + 1] = hi
            if fam.SK[p] == 1: S[p, 2], S[p, 3] = Q - S[p, 1], Q - S[p, 0]
        if not cgt.propagate(S, fam.SK, fam.ORD, fam.SUM, Q): cbad += 1; continue
        for (x, y) in I[fam.k:]:
            for cx in {int(x * G), min(G - 1, int(x * G))}:
                cy = int(y * G)
                if cx + cy > G - 1: cx = G - 1 - cy
                CB = np.array([[cx * cs, (cx + 1) * cs, cy * cs, (cy + 1) * cs]], np.int64)
                c0, c1, c2, c3 = (int(v) for v in CB[0]); assert c0 <= x * Q <= c1 and c2 <= y * Q <= c3
                st = {}
                CB2, _ = cgt.p2_cell_prune(S, fam, CB, np.zeros((1, 1), bool), Q, rq, rho, 8, st)
                ctested += 1; cbad += len(CB2) == 0
print(f'cell-P2 negative control: {ctested} protected cells tested, wrongly excluded {cbad}')
print(f'n={n} {fam_name}: images {len(ims)}, boxes tested {tested} (skipped by symmetry rows {skipped}), wrongly closed {bad}')
