"""Independent check of a finished n = 9 branch-and-bound run from its saved files (peer review, 2026-11-13).

Reads stats.json, progress.log and fails.jsonl of the run and the exp 44 certificate, and checks exactly:
  1. the run finished (done = true; the last log line is DONE with the same statistics);
  2. no failures: fail = 0 and fails.jsonl is empty;
  3. node accounting: every node is either a leaf (closed by P0, P1, P2 or the cover, or a failure) or an internal
     node with exactly two children, so for a forest of R full binary trees  leaves = internal + R.  With
     leaves = p0 + p1 + p2 + cover + fail and internal = nodes - leaves this must give R = 36 = C(9, 2), the number
     of root boxes (exp 103 bb.py roots(), domain 'basic': x_0 = 0, x_8 = 1, y_b = 0, y_t = 1 for each b < t);
  4. the threshold: 0 < T <= t_lower (1 - 1e-9) < t_lower <= t*, with T the exact double used by the run and
     t_lower from the local certificate, so every closed box has maximum below the certified optimum;
  5. coverage tally: closed_w within 1e-9 of 1 (a float tally of closed weight; check 3 is the exact one).
Usage:  python verify_run.py RUN_DIR CERT_JSON [--roots 36]"""
import sys, os, json
from fractions import Fraction as Fr

def main(run, cert_path, roots=36):
    S = json.load(open(os.path.join(run, 'stats.json')))
    ok = True
    def check(cond, msg):
        nonlocal ok
        print(('PASS ' if cond else 'FAIL ') + msg); ok &= bool(cond)
    check(S.get('done') is True, 'run finished (done = true)')
    last = [l for l in open(os.path.join(run, 'progress.log'), encoding='utf-8') if l.strip()][-1]
    check(' DONE ' in last and json.loads(last.split(': ', 1)[1]) == S, 'last log line is DONE with identical statistics')
    fails = open(os.path.join(run, 'fails.jsonl')).read().strip()
    check(S['fail'] == 0 and fails == '', 'no failures (fail = 0, fails.jsonl empty)')
    leaves = S['p0'] + S['p1'] + S['p2'] + S['cover'] + S['fail']
    internal = S['nodes'] - leaves
    check(leaves - internal == roots,
          f'forest identity: leaves {leaves:,} - internal {internal:,} = {leaves - internal} = number of roots {roots}')
    C = json.load(open(cert_path))[str(S['n'])][0]
    tl = Fr(int(C['t_lower'].split('e')[0]), 10 ** 32); assert C['t_lower'].endswith('e-32')
    T = Fr(S['T_exact']); tlr = Fr(S['t_lower'])
    check(tlr == tl, 'run used the certificate\'s t_lower')
    check(0 < T <= tl * (1 - Fr(1, 10 ** 9)) < tl, f'0 < T <= t_lower (1 - 1e-9) < t_lower  (T = {float(T):.16g})')
    check(abs(S['closed_w'] - 1) < 1e-9, f'closed weight tally = {S["closed_w"]!r} (float)')
    print('RUN VERIFIED' if ok else 'RUN NOT VERIFIED'); return ok

if __name__ == '__main__':
    a = sys.argv[1:]; r = 36
    if '--roots' in a: i = a.index('--roots'); r = int(a[i + 1]); del a[i:i + 2]
    sys.exit(0 if main(a[0], a[1], r) else 1)
