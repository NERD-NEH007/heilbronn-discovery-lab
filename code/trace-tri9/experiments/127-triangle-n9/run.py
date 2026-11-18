"""driver:  python run.py n FAM MODE VAL [G] [workers] [minutes] [depth] [max_sw_div]
MODE eps:   T = t_up (1 + VAL)            claim: no config of FAM has Phi > T
MODE exact: T = t_lo (1 - VAL), cover on   claim: every config of FAM with Phi > T lies in a basin ball
MODE neg:   T = t_lo (1 - VAL), no cover   (negative control: must NOT close)"""
import json, math, os, sys, time, multiprocessing as mp
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cgt

CERT9 = os.path.join(HERE, 'cert', 'local_certs_tri_n9.json')
CERT8 = os.path.join(HERE, '..', '113-triangle-n8', 'cert', 'local_certs_tri.json')


def load_cert(n):
    c = json.load(open(CERT9 if n == 9 else CERT8))[str(n)][0]
    assert c['certified'] and c['mode'] == 'first-order'
    P = [(Fr(x), Fr(y)) for x, y in c['points_exact']]
    rb = Fr(c['extra']['basin_radius_exact']); er = Fr(c['extra']['box_rho_exact'])
    R = rb * Fr(999999, 1000000) - er
    assert R > 0
    if n == 9:
        tlo = thi = Fr(c['t_exact'])
    else:
        tlo, thi = Fr(c['t_lower']), Fr(c['t_upper'])
    return P, R, tlo, thi


# Delta_{n-2} upper bounds used by the gap lemma (raw areas in Omega)
DELTA_UP = {9: Fr(7, 144),                       # n = 7: Zeng-Chen; re-proved exp 113 (exact mode)
            8: Fr(1, 16) * Fr(1001, 1000)}       # n = 6: Yang-Zhang-Zeng 1/16; exp 113 eps run gives this bound


def gap_l0(T, n):
    """two points on one side with parameter distance d: the other n - 2 points lie in a homothetic copy of
    Omega of ratio 1 - 2T/d, so T < Delta_{n-2} (1 - 2T/d)^2, i.e. d > 2T / (1 - sqrt(T/Delta)).  Returns a
    rational lower bound for that threshold (sqrt rounded down)."""
    r = T / DELTA_UP[n]
    assert r < 1
    s_lo = Fr(math.isqrt(math.floor(r * 10 ** 40)), 10 ** 20)
    assert s_lo * s_lo <= r
    return 2 * T / (1 - s_lo)


def setup(n, fam_name, mode, val, G, swdiv=8):
    cgt.configure(n)
    K = 14
    Q = G * 2 ** K
    P, Rcov, tlo, thi = load_cert(n)
    v = Fr(val).limit_denominator(10 ** 12)
    T = thi * (1 + v) if mode == 'eps' else tlo * (1 - v)
    rho = 2 * T
    rq = (rho * Q * Q).numerator // (rho * Q * Q).denominator
    cs = Q // G
    assert 2 * cs * Q <= rho * Q * Q, 'cell bound 2h <= rho fails'
    assert 2 * Q * Q < 2 ** 62
    l0 = gap_l0(T, n)
    l0u = (l0 * Q).numerator // (l0 * Q).denominator
    fam = cgt.make_family(fam_name, Q, l0u)
    cfg = dict(max_cliques=200000, local_budget=200000, lp_w=Q, max_sw=Q // swdiv, lp_d=3,
               cell_p2=int(os.environ.get('CELL_P2', '0')))     # P2 cell exclusion cap per graph (0 = off)
    cov = cgt.make_cover(P, Rcov, Q) if mode == 'exact' else None
    return fam, T, rho, rq, Q, cfg, cov, l0


_GRAPH = {}
TASK_SECS = 120      # a parameter-family task returns its open boxes after this; the driver resubmits them


def task(args):
    n, fam_name, mode, val, G, swdiv, S0, first, t_end = args
    fam, T, rho, rq, Q, cfg, cov, _ = setup(n, fam_name, mode, val, G, swdiv)
    st = cgt.new_stats()
    t0 = time.time()
    if first is None:
        unres = [u.tolist() for u in cgt.outer_tree(fam, G, rq, rho, Q, cfg, st, S0=np.array(S0, np.int64), covered=cov,
                                                     t_end=min(t_end, time.time() + TASK_SECS))]
        st['_leftover'] = [u.tolist() for u in st.get('_leftover', [])]
    else:
        key = (n, fam_name, mode, val, G)
        if key not in _GRAPH:
            _GRAPH[key] = cgt.cell_graph(fam.root(), fam, G, Q, rq, cgt.new_stats(), rho, cfg['cell_p2'])
        r = cgt.solve_outer(fam.root(), fam, G, rq, rho, Q, cfg, st, cov, first=first, graph=_GRAPH[key])
        unres = [] if st['unresolved_cliques'] == 0 else [['first', first, st['unresolved_cliques']]]
        st['outer_leaves'] += 1
    st['secs'] = time.time() - t0
    return st, unres


def presplit(fam, depth, Q):
    boxes = [fam.root()]
    for _ in range(depth):
        nxt = []
        for S in boxes:
            w = (S[:, 1::2] - S[:, 0::2]) * fam.SSPL
            p, c = np.unravel_index(int(np.argmax(w)), w.shape)
            lo, hi = S[p, 2 * c], S[p, 2 * c + 1]; mid = (lo + hi) // 2
            if mid <= lo: nxt.append(S); continue
            S1 = S.copy(); S1[p, 2 * c + 1] = mid; S2 = S.copy(); S2[p, 2 * c] = mid
            for X in (S1, S2):
                if cgt.propagate(X, fam.SK, fam.ORD, fam.SUM, Q): nxt.append(X)
        boxes = nxt
    return boxes


def main(argv):
    n, fam_name, mode, val = int(argv[0]), argv[1], argv[2], float(argv[3])
    G = int(argv[4]) if len(argv) > 4 else (37 if n == 9 else 30)
    W = int(argv[5]) if len(argv) > 5 else 14
    minutes = float(argv[6]) if len(argv) > 6 else 60
    depth = int(argv[7]) if len(argv) > 7 else 6
    swdiv = int(argv[8]) if len(argv) > 8 else 8
    fam, T, rho, rq, Q, cfg, cov, l0 = setup(n, fam_name, mode, val, G, swdiv)
    P, Rcov, tlo, thi = load_cert(n)
    print(f'n={n} {fam_name} mode={mode} val={val} T={float(T):.15f} T/t-1={float(T / tlo) - 1:.3e} G={G} Q={Q} '
          f'l0={float(l0):.4f} cell_p2={cfg["cell_p2"]} params={fam.nparam} free={fam.m} cover_R={float(Rcov):.4e}', flush=True)
    t_end = time.time() + 60 * minutes
    if fam.nparam:
        boxes = presplit(fam, depth, Q)
        tasks = [(n, fam_name, mode, val, G, swdiv, S.tolist(), None, t_end) for S in boxes]
    else:
        CB, adj = cgt.cell_graph(fam.root(), fam, G, Q, rq, cgt.new_stats(), rho, cfg['cell_p2'])
        print(f'  fixed supports: {len(CB)} cells kept, {int(adj.sum()) // 2} edges', flush=True)
        tasks = [(n, fam_name, mode, val, G, swdiv, None, u, t_end) for u in range(len(CB))]
    t0 = time.time()
    tot = cgt.new_stats(); tot['secs'] = 0.0; unres = []; rnd = 0; ntasks = 0
    with mp.Pool(W) as pool:
        while tasks:
            rnd += 1; ntasks += len(tasks); nxt = []
            for i, (st, un) in enumerate(pool.imap_unordered(task, tasks)):
                lo = st.pop('_leftover', [])
                for k, v in st.items(): tot[k] += v
                unres += un
                nxt += lo
                if (i + 1) % max(1, len(tasks) // 5) == 0:
                    print(f'  round {rnd}: {i + 1}/{len(tasks)} roots {time.time() - t0:.0f}s leaves={tot["outer_leaves"]} '
                          f'cliques={tot["cliques"]} local={tot["local_nodes"]} P2={tot["P2"]} cover={tot["cover"]} '
                          f'unres={len(unres)} requeued={len(nxt)}', flush=True)
            if time.time() > t_end:
                unres += [S if isinstance(S, list) else S.tolist() for S in nxt]; tot['unresolved'] += len(nxt); break
            tasks = [(n, fam_name, mode, val, G, swdiv, S, None, t_end) for S in nxt]
    tot['wall'] = time.time() - t0; tot['roots'] = ntasks; tot['rounds'] = rnd
    print(json.dumps(tot), flush=True)
    out = os.path.join(HERE, 'results'); os.makedirs(out, exist_ok=True)
    tag = f'n{n}_{fam_name}_{mode}_{val:g}_G{G}' + (f'_cp{cfg["cell_p2"]}' if cfg['cell_p2'] else '')
    json.dump(dict(stats=tot, T=str(T), unresolved=unres, l0=str(l0)), open(os.path.join(out, tag + '.json'), 'w'))
    if unres:
        verdict = f'{len(unres)} unresolved'
    elif mode == 'eps':
        verdict = 'PROVED: no config in family with Phi > T'
    elif mode == 'exact':
        verdict = f'PROVED: every config with Phi > T is in a basin ball (cover used {tot["cover"]}x)'
    else:
        verdict = 'NEGATIVE CONTROL CLOSED -- UNSOUND?'
    print('VERDICT:', verdict, flush=True)
    return tot, unres


if __name__ == '__main__':
    main(sys.argv[1:])
