"""Smoke test (about a minute): re-derives a few headline numbers of both papers from the files in this repository.

  python code/smoke_test.py

1. exact minimum area and tie counts of our best configurations n = 37, 40, 43, 64, 100 match the stored values;
2. experiment 44 certificate (paper 1, Section 2.3) for n = 40 (second order, symmetry-reduced) and n = 51;
3. experiment 78 independent check for n = 37 (first order) and n = 40 (second order, d = 1, 64 loaded ties);
4. exact form for n = 20 (paper 1, Table 7): the degree-19 polynomial with leading coefficient 3 * 2^42 vanishes at
   the stored 300-digit value, which agrees with the published record value;
5. record n = 22 (paper 2, Theorem 3.1(a)): from the coordinates printed in paper 2, exactly 34 triangles lie within
   relative 1e-12 of the minimum, the same for every tolerance up to 1e-2, and the next triangle is 1.614 % above t.
"""
import os, sys, json, time, subprocess, tempfile
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hlib

T0 = time.time(); FAIL = []
def check(name, ok, info=''):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok: FAIL.append(name)

# 1. exact min area
for n in (37, 40, 43, 64, 100):
    d = json.load(open(os.path.join(hlib.CONF, 'best', f'n{n}.json')))
    m, c = hlib.exact_min_area(d['points'])
    check(f'exact min area n={n}', f'{m.numerator}/{m.denominator}' == d['min_area_exact'] and c['rel_1e-09'] == d['count_rel_1e9']
          and c['rel_1e-06'] == d['count_rel_1e6'] and hlib.inside_unit_square(d['points']),
          f'(t = {float(m):.12e}, n^2 t = {float(m) * n * n:.4f}, ties 1e-9: {c["rel_1e-09"]}, 1e-6: {c["rel_1e-06"]})')
d40 = json.load(open(os.path.join(hlib.CONF, 'best', 'n40.json')))
check('n=40 has 64 critical triangles (paper 2: 64 < 73 = 2n-7)', d40['count_rel_1e9'] == 64 and d40['count_rel_1e6'] == 64)

# 2. exp 44
sys.path.insert(0, os.path.join(HERE, 'exp44')); import certify_interval as CI
want44 = {40: ('second-order', 64, 8, 36), 51: ('second-order', 95, 7, 51)}
for n, (mode, lt, lw, nc) in want44.items():
    o = CI.certify(n, hlib.load_best(n), log=lambda *a, **k: None)
    check(f'exp 44 certificate n={n}', o.get('certified') is True and o.get('mode') == mode and o.get('loaded_ties') == lt
          and o.get('loaded_walls') == lw and o.get('n_core') == nc,
          f"({o.get('mode')}, loaded ties {o.get('loaded_ties')}, walls {o.get('loaded_walls')}, core {o.get('n_core')}, "
          f"radius {o.get('existence_radius'):.2e}, t in [{o.get('t_lower', '')[:14]}..])")

# 3. exp 78
sys.path.insert(0, os.path.join(HERE, 'exp78')); import kkt_check as K
for n, (ties, walls, core, dd) in {37: (68, 7, 37, 0), 40: (64, 8, 36, 1)}.items():
    info = K.certify(n, [tuple(p) for p in hlib.load_best(n)['points']])
    check(f'exp 78 independent check n={n}', info.get('certified') is True and info['ties'] == ties and info['walls'] == walls
          and info['core'] == core and info['d'] == dd,
          f"(ties {info['ties']}, walls {info['walls']}, core {info['core']}, d {info['d']}, radius {info.get('radius', 0):.1e}"
          + (f", eigenvalue {info['second_order_eigs'][0]:.3g})" if info.get('second_order_eigs') else ')'))

# 4. exact form n = 20
import check_exact_forms as X
e20 = [e for e in json.load(open(os.path.join(hlib.CERT, 'exp104', 'exact_forms.json'))) if e['label'] == '20'][0]
ok, msg, p = X.check(e20)
rv = hlib.record_values()[20]
check('exact form n=20', ok and p.degree() == 19 and p.LC() == 3 * 2 ** 42 and abs(float(e20['t_full']) / rv - 1) < 1e-13,
      f'(degree {p.degree()}, LC = 3*2^42: {p.LC() == 3 * 2 ** 42}; {msg}; vs record value {float(e20["t_full"]) / rv - 1:+.1e})')

# 5. record 22 from the printed coordinates
r22 = json.load(open(os.path.join(hlib.CONF, 'records_printed', 'record22_paper2.json')))
A, D2 = hlib.exact_areas2(r22['points']); A.sort(); m = A[0]
cnt = {k: sum(1 for a in A if a <= m * (1 + Fr(1, 10 ** k))) for k in (12, 9, 6, 4, 2)}
nxt = A[cnt[12]] / Fr(m) - 1
check('record 22: 34 critical triangles (< 37 = 2n-7)', set(cnt.values()) == {34} and abs(float(nxt) - 0.01614) < 5e-5
      and abs(float(Fr(m, 2 * D2)) - 0.0107716568096426) < 1e-16,
      f'(t = {float(Fr(m, 2 * D2)):.16f}, counts at 1e-12..1e-2: {sorted(set(cnt.values()))}, next {100 * float(nxt):.3f}% above)')

# 6. n = 9 closed form (paper 3) and exp 110 import
r = subprocess.run([sys.executable, os.path.join(HERE, 'exp110', 'check_closed_form_exact.py')], capture_output=True, text=True, timeout=120)
check('exp 110 exact closed form (Q(sqrt 65))', r.returncode == 0 and 'ALL EXACT CHECKS PASSED' in r.stdout,
      '(' + (r.stdout.strip().splitlines() or ['no output'])[-1][:90] + ')')
ROOT = os.path.dirname(HERE)
r = subprocess.run([sys.executable, os.path.join(HERE, 'exp110', 'verify_run.py'),
                    os.path.join(ROOT, 'certificates', 'exp110', 'results', 'run_n9'),
                    os.path.join(ROOT, 'certificates', 'exp103', 'local_certs.json')], capture_output=True, text=True, timeout=60)
check('exp 110 n = 9 run accounting (verify_run.py)', r.returncode == 0 and 'RUN VERIFIED' in r.stdout)
r = subprocess.run([sys.executable, os.path.join(HERE, 'exp110', 'clean_room', 'test_check_p2.py')], capture_output=True, text=True, timeout=120)
check('clean-room P2 checker: negative tests', r.returncode == 0)
r = subprocess.run([sys.executable, os.path.join(HERE, 'exp110', 'clean_room', 'check_p2.py'),
                    os.path.join(ROOT, 'certificates', 'exp110', 'p2dump', 'n9_r0.jsonl.gz')], capture_output=True, text=True, timeout=120)
check('clean-room P2 checker on n = 9 sample (root 0)', r.returncode == 0 and 'rejected 0' in r.stdout, '(' + r.stdout.strip()[-60:] + ')')
# 7. triangle n = 8 (exp 113): run accounting and exact septic identification
S8 = json.load(open(os.path.join(ROOT, 'certificates', 'exp113', 'run_n8', 'stats.json')))
C8 = json.load(open(os.path.join(HERE, 'exp113', 'cert', 'local_certs_tri.json')))['8'][0]
tl8 = Fr(int(C8['t_lower'].split('e')[0]), 10 ** 32); T8 = Fr(S8['T_exact'])
leaves8 = S8['p0'] + S8['p1'] + S8['p2'] + S8['cover'] + S8['fail']
check('triangle n = 8 run: done, 0 failures, leaves - internal = 64 roots, T <= t_lower(1 - 1e-9)',
      S8['done'] and S8['fail'] == 0 and 2 * leaves8 - S8['nodes'] == 64 and T8 <= tl8 * (1 - Fr(1, 10 ** 9))
      and open(os.path.join(ROOT, 'certificates', 'exp113', 'run_n8', 'fails.jsonl')).read().strip() == '')
r = subprocess.run([sys.executable, os.path.join(HERE, 'exp113', 'cert', 'identify_n8.py')], capture_output=True, text=True, timeout=300)
check('triangle n = 8 value = root of the septic (exact, identify_n8.py)', r.returncode == 0 and "'proved_t_equals_septic_root': True" in r.stdout)
try:
    sys.path.insert(0, os.path.join(HERE, 'exp110')); import bb2
    check('exp 110 bb2 imports', hasattr(bb2, 'Problem2'))
except Exception as ex:
    check('exp 110 bb2 imports', False, repr(ex))

print(f"\n{'ALL PASSED' if not FAIL else 'FAILED: ' + ', '.join(FAIL)} in {time.time() - T0:.0f} s")
sys.exit(1 if FAIL else 0)
