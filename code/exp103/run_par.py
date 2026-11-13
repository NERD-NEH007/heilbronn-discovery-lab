"""Parallel, checkpointed driver for bb.py.

    python run_par.py N MODE [--eps E] [--workers 4] [--budget 2000] [--tag TAG]

MODE eps   : T = t*(1+eps) rounded up to a double; no covers.  Claim: max min-area <= T.
MODE exact : T = t_lower*(1-1e-9) rounded down; certified basin covers on.  Claim: every config with min-area > T is
             covered by a certified basin ball (=> max = t*, unique up to symmetry).

State (output/exp103/run_{tag}/): open.pkl (list of open boxes with their weight), stats.json, progress.log, fails.jsonl.
Restarting the same command resumes from open.pkl.  Each task = one open box; a worker explores it depth-first for
at most `budget` nodes and returns the unexplored remainder, which goes back into the queue.
Progress = sum of weights of closed boxes (a box at depth d below a root has weight 2^-d / #roots): the fraction of
the search tree's "volume" that is finished (not a time estimate, but its rate gives one).
"""
import os, sys, json, time, pickle, argparse
from fractions import Fraction as Fr
import numpy as np
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import bb

_P = None


DOMAIN = os.environ.get('DOM', 'walls')


def make_problem(n, mode, eps):
    C = json.load(open(os.path.join(REPO, 'certificates', 'exp103', 'local_certs.json')))
    tl = C[str(n)][0]['t_lower']; tlow = Fr(int(tl.split('e')[0]), 10 ** int(tl.split('e-')[1]))
    if mode == 'eps':
        T = np.nextafter(float(tlow * (1 + Fr(eps))), 1.0); covers = []
        assert Fr(float(T)) >= tlow * (1 + Fr(eps))
    else:
        T = np.nextafter(float(tlow * (1 - Fr(1, 10 ** 9))), 0.0); covers = bb.load_covers(n)
        assert covers and Fr(float(T)) < tlow
    return bb.Problem(n, Fr(float(T)), covers=covers, branch='invol', cover_width=bb.D // 1000, domain=DOMAIN), tlow


def init(n, mode, eps):
    global _P
    _P, _ = make_problem(n, mode, eps); _P.roots()        # builds cons_table
    _P.audit = float(os.environ.get('AUDIT', '0')); _P.rng = np.random.default_rng(os.getpid())


def work(task):
    lo, hi, depth, wgt, cid = task
    st = bb.new_stats(); st['closed_w'] = 0.0; fails = []; a0 = _P.audit_count
    rest = _P.solve(np.array(lo, np.int64), np.array(hi, np.int64), st, node_budget=_BUDGET, fails=fails, max_fail=10 ** 9,
                    depth=depth, weight=wgt, cid=cid)
    rest = [(l.tolist(), h.tolist(), d, w, c) for (l, h, d, w, c) in rest]
    st['audited'] = _P.audit_count - a0
    return st, rest, fails


_BUDGET = 2000


def main():
    global _BUDGET
    ap = argparse.ArgumentParser()
    ap.add_argument('n', type=int); ap.add_argument('mode'); ap.add_argument('--eps', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--budget', type=int, default=2000)
    ap.add_argument('--tag', default=None); ap.add_argument('--hours', type=float, default=1e9)
    a = ap.parse_args(); _BUDGET = a.budget
    tag = a.tag or f'n{a.n}_{a.mode}' + (f'_{a.eps:g}' if a.mode == 'eps' else '')
    d = os.path.join(REPO, 'output', 'exp103', f'run_{tag}'); os.makedirs(d, exist_ok=True)
    P, tlow = make_problem(a.n, a.mode, a.eps)
    sp_, op_, lp_ = os.path.join(d, 'stats.json'), os.path.join(d, 'open.pkl'), os.path.join(d, 'progress.log')
    if os.path.exists(op_):
        opn = pickle.load(open(op_, 'rb')); S = json.load(open(sp_))
    else:
        roots = P.roots(); w0 = 1.0 / len(roots)
        opn = [(l.tolist(), h.tolist(), 0, w0, c) for l, h, c in roots][::-1]
        S = dict(n=a.n, mode=a.mode, eps=a.eps, T=float(P.T), T_exact=str(P.T), t_lower=str(tlow), elapsed=0.0,
                 **bb.new_stats(), closed_w=0.0, tasks=0)
    log = open(lp_, 'a'); flog = open(os.path.join(d, 'fails.jsonl'), 'a')
    print(f'start {tag}: open {len(opn)}, closed_w {S["closed_w"]:.6g}, workers {a.workers}', file=log, flush=True)
    t0 = time.time(); el0 = S['elapsed']; last_ck = time.time(); last_log = 0
    keys = list(bb.new_stats().keys())
    with mp.Pool(a.workers, initializer=init, initargs=(a.n, a.mode, a.eps)) as pool:
        pending = {}
        def submit():
            while opn and len(pending) < 4 * a.workers:
                t = opn.pop(); pending[id(t)] = (t, pool.apply_async(work, (t,)))
        submit()
        while pending:
            done = [k for k, (t, r) in pending.items() if r.ready()]
            if not done: time.sleep(0.05); continue
            for k in done:
                t, r = pending.pop(k); st, rest, fails = r.get()
                for kk in keys:
                    S[kk] = max(S[kk], st[kk]) if kk == 'maxdepth' else S[kk] + st[kk]
                S['closed_w'] += st['closed_w']; S['tasks'] += 1; S['audited'] = S.get('audited', 0) + st.get('audited', 0)
                opn.extend(rest)
                for f in fails: flog.write(json.dumps(f) + '\n')
                if fails: flog.flush()
            submit()
            now = time.time(); S['elapsed'] = el0 + now - t0
            if now - last_log > 60:
                last_log = now
                rate = (S['closed_w']) / max(S['elapsed'], 1)
                eta = (1 - S['closed_w']) / rate / 3600 if rate > 0 else float('inf')
                print(f"{time.strftime('%H:%M:%S')} el {S['elapsed']/3600:.2f}h nodes {S['nodes']} open {len(opn)+len(pending)} "
                      f"closed_w {S['closed_w']:.6f} p1 {S['p1']} p2 {S['p2']} cover {S['cover']} fail {S['fail']} "
                      f"depth {S['maxdepth']} audited {S.get('audited', 0)} naive_eta {eta:.2f}h", file=log, flush=True)
            if now - last_ck > 300 or S['elapsed'] - el0 > a.hours * 3600:
                last_ck = now
                snap = opn + [t for (t, r) in pending.values()]          # pending tasks are re-run on resume
                pickle.dump(snap, open(op_ + '.tmp', 'wb')); os.replace(op_ + '.tmp', op_)
                json.dump(S, open(sp_, 'w'), indent=1)
                if S['elapsed'] - el0 > a.hours * 3600:
                    print('time limit; checkpoint written', file=log, flush=True); pool.terminate(); return
    S['done'] = True; S['elapsed'] = el0 + time.time() - t0
    json.dump(S, open(sp_, 'w'), indent=1); pickle.dump([], open(op_, 'wb'))
    print(f"DONE {tag}: {json.dumps(S)}", file=log, flush=True); print(json.dumps(S))


if __name__ == '__main__':
    main()
