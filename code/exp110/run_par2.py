"""Parallel, checkpointed driver for bb2.py (adapted from exp 103 run_par.py).

    python run_par2.py N MODE [--workers W] [--budget 2000] [--tag TAG] [--skip G] [--nodiag] [--est-nodes E]

MODE exact: T = t_lower (1 - 1e-9) rounded down, certified basin covers on (as exp 103).  MODE eps: T = t*(1+eps).
State in output/exp110/run_<tag>/:
  state.pkl     {'S': stats, 'open': open boxes}  (single atomic file; the run resumes from it)
  stats.json    human-readable copy of the stats
  progress.log  one line per minute: nodes, open boxes, closed width fraction, rate, ETA
  fails.jsonl   unprunable boxes at the grid floor (the proof needs zero)
  pid.txt       PID of the running master process
  STOP          create this file to make the run checkpoint and exit cleanly
Restarting the same command resumes.  Closed weight = sum over closed boxes of 2^-depth / #roots (exact bisection
fractions): the fraction of the search tree's 'volume' that is finished.
"""
import os, sys, json, time, pickle, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import multiprocessing as mp

_P = None
_BUDGET = 2000


def init(n, mode, eps, diag, skip, audit):
    global _P
    import bb2
    _P, _ = bb2.make_problem(n, mode, eps, diag=diag, skip_gap=skip)
    _P.audit = audit; _P.rng = np.random.default_rng(os.getpid())


def work(task):
    import bb2
    lo, hi, depth, wgt, pred = task
    st = bb2.new_stats(); st['closed_w'] = 0.0; fails = []; a0 = _P.audit_count; f0 = _P.lp_fail
    rest = _P.solve2(lo, hi, st, node_budget=_BUDGET, fails=fails, depth=depth, weight=wgt, pred=pred)
    rest = [(l.tolist(), h.tolist(), d, w, p) for (l, h, d, w, p) in rest]
    st['audited'] = _P.audit_count - a0; st['lp_fail'] = _P.lp_fail - f0
    return st, rest, fails


def work_b(args):
    global _BUDGET
    _BUDGET = args[1]
    return work(args[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('n', type=int); ap.add_argument('mode'); ap.add_argument('--eps', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument('--budget', type=int, default=2000); ap.add_argument('--tag', default=None)
    ap.add_argument('--skip', type=float, default=None); ap.add_argument('--nodiag', action='store_true')
    ap.add_argument('--audit', type=float, default=0.0); ap.add_argument('--est-nodes', type=float, default=0.0)
    a = ap.parse_args()
    import bb2
    if os.name == 'nt':        # ask Windows not to idle-sleep while this process runs (released automatically at exit)
        try:
            import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception: pass
    tag = a.tag or f'n{a.n}_{a.mode}'
    d = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'output', 'exp110', f'run_{tag}'); os.makedirs(d, exist_ok=True)
    stp, sjp, lgp, stopf = (os.path.join(d, f) for f in ('state.pkl', 'stats.json', 'progress.log', 'STOP'))
    if os.path.exists(stopf): os.remove(stopf)
    open(os.path.join(d, 'pid.txt'), 'w').write(str(os.getpid()))
    P, tlow = bb2.make_problem(a.n, a.mode, a.eps, diag=not a.nodiag, skip_gap=a.skip)
    keys = list(bb2.new_stats().keys()) + ['lp_fail', 'audited']
    if os.path.exists(stp):
        st_ = pickle.load(open(stp, 'rb')); S = st_['S']; opn = st_['open']
        assert S['T_exact'] == str(P.T) and S['diag'] == (not a.nodiag), 'state belongs to a different problem'
    else:
        roots = P.roots(); w0 = 1.0 / len(roots)
        opn = [(l.tolist(), h.tolist(), 0, w0, None) for l, h, c in roots][::-1]
        S = dict(n=a.n, mode=a.mode, eps=a.eps, T=float(P.T), T_exact=str(P.T), t_lower=str(tlow), diag=not a.nodiag,
                 skip=a.skip, elapsed=0.0, closed_w=0.0, tasks=0, done=False, **{k: 0 for k in keys})
    log = open(lgp, 'a'); flog = open(os.path.join(d, 'fails.jsonl'), 'a')
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start {tag} pid {os.getpid()}: open {len(opn)}, nodes {S['nodes']}, "
          f"closed_w {S['closed_w']:.6g}, workers {a.workers}, skip {a.skip}, diag {not a.nodiag}", file=log, flush=True)

    def checkpoint(snap):
        pickle.dump(dict(S=S, open=snap), open(stp + '.tmp', 'wb')); os.replace(stp + '.tmp', stp)
        json.dump(S, open(sjp + '.tmp', 'w'), indent=1); os.replace(sjp + '.tmp', sjp)

    t0 = time.time(); el0 = S['elapsed']; n0 = S['nodes']; last_ck = time.time(); last_log = 0
    hist = [(S['elapsed'], S['nodes'], S['closed_w'])]
    with mp.Pool(a.workers, initializer=init, initargs=(a.n, a.mode, a.eps, not a.nodiag, a.skip, a.audit)) as pool:
        pending = {}
        def submit():
            while opn and len(pending) < 3 * a.workers:
                t = opn.pop(); pending[id(t)] = (t, pool.apply_async(work_b, ((t, a.budget),)))
        submit()
        stopping = False
        while pending:
            done = [k for k, (t, r) in pending.items() if r.ready()]
            if not done: time.sleep(0.05); continue
            for k in done:
                t, r = pending.pop(k); st, rest, fails = r.get()
                for kk in keys:
                    S[kk] = max(S[kk], st.get(kk, 0)) if kk == 'maxdepth' else S[kk] + st.get(kk, 0)
                S['closed_w'] += st['closed_w']; S['tasks'] += 1
                opn.extend(rest)
                for f in fails: flog.write(json.dumps(f) + '\n')
                if fails: flog.flush()
            submit()
            now = time.time(); S['elapsed'] = el0 + now - t0
            if now - last_log > 60:
                last_log = now; hist.append((S['elapsed'], S['nodes'], S['closed_w']))
                h0 = next((h for h in hist if h[0] >= S['elapsed'] - 3600), hist[0])      # 1-hour window
                dt = max(S['elapsed'] - h0[0], 1.0)
                nrate = (S['nodes'] - h0[1]) / dt; wrate = (S['closed_w'] - h0[2]) / dt
                eta_w = (1 - S['closed_w']) / wrate / 3600 if wrate > 0 else float('inf')
                eta_n = (a.est_nodes - S['nodes']) / nrate / 3600 if (a.est_nodes and nrate > 0) else float('nan')
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} el {S['elapsed']/3600:.2f}h nodes {S['nodes']} open {len(opn)+len(pending)} "
                      f"closed_w {S['closed_w']:.6f} rate {nrate:.0f}/s p1 {S['p1']} p2 {S['p2']} lpskip {S['lpskip']} cover {S['cover']} "
                      f"fail {S['fail']} depth {S['maxdepth']} lp_fail {S['lp_fail']} audited {S['audited']} "
                      f"eta_width {eta_w:.1f}h eta_nodes {eta_n:.1f}h", file=log, flush=True)
            if os.path.exists(stopf): stopping = True
            if now - last_ck > 300 or stopping:
                last_ck = now
                checkpoint(opn + [t for (t, r) in pending.values()])         # pending tasks are re-run on resume
                if stopping:
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} STOP file seen; checkpoint written, exiting", file=log, flush=True)
                    pool.terminate(); return
    S['done'] = True; S['elapsed'] = el0 + time.time() - t0
    checkpoint([])
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} DONE {tag}: {json.dumps(S)}", file=log, flush=True); print(json.dumps(S))


if __name__ == '__main__':
    main()
