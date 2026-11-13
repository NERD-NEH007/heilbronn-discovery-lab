"""Paper 1 figures from the CURRENT library (exp 108, 2026-11-12).  Based on experiments/report_figures/make_figures.py
and experiments/92-growth-cost/phase.py (neither is edited).  Notation as in the paper: t = minimum triangle area.

  fig_families.pdf   n^2 t of the best per n, 19-64 (records 19-36: configurations/record_values.json; ours: configurations/best),
                     coloured by the exp 85 family holder (families computed on the library of commit 091485f;
                     open markers: best improved since, family label not reassigned).
  fig_transition.pdf four exp 86 observables recomputed on the current bests (pure geometry), 20-64.
  fig_survival.pdf   exp 85 tie survival (library 091485f): current-best pairs within a family / at a handover /
                     unrelated k = 1 pairs.  AUC and Cohen's d printed.
  fig_phase.pdf      exp 92 phase diagram recomputed on the current bests (convex layers K, centre gap, symmetry);
                     family brackets from exp 85 (library 091485f).
  python code/make_paper_figures.py   (writes output/figures/*.pdf; prints every number used in captions)
Inputs: configurations/best (n >= 37), configurations/record_values.json, code/figure_data/ (exp 85, 86, 92 tables).
Geometry of the records 20-36 is recomputed from records/n{n}.txt if you downloaded them; otherwise the
precomputed scalar observables in figure_data/observables_records.json are used."""
import os, sys, json, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import hlib
FD = os.path.join(HERE, 'figure_data'); OUTD = os.path.join(hlib.REPO, 'output', 'figures'); os.makedirs(OUTD, exist_ok=True)
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150,
                     'pdf.fonttype': 42})
INK, MUTED, GRID, BLUE, ORANGE, RED = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#eb6834', '#c0392b'
def save(fig, name):
    fig.savefig(os.path.join(OUTD, f'{name}.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------------- data: current bests ----------------
RV = hlib.record_values()
def best_entry(n):
    if n <= 36:
        try: P = np.array(hlib.load_record(n)['points'], float)
        except FileNotFoundError: P = None
        return float(RV[n]), P
    e = hlib.load_best(n)
    return float(e['value']), np.array(e['points'], float)
BEST = {n: best_entry(n) for n in range(19, 65)}
trend = lambda n: 2.26 + 0.95 * math.log(n)

# ---------------- fig 1: families ----------------
phy = json.load(open(os.path.join(FD, 'phylogeny.json')))
holder = phy['holder']; famrow = {r['fam']: r for r in phy['rows']}
ns = list(range(19, 65))
_V85 = json.load(open(os.path.join(FD, 'values_exp85_library.json')))['values']
def val_at(commit, n):                          # best value per n in the library used by exp 85 (commit 091485f)
    return _V85[str(n)]
changed = [n for n in ns if n > 36 and abs(BEST[n][0] / val_at('091485f', n) - 1) > 1e-6]
fams = []; [fams.append(holder[str(n)]) for n in ns if holder[str(n)] not in fams]
cmap = plt.get_cmap('tab20'); col = {f: cmap(i % 20) for i, f in enumerate(fams)}
fig, ax = plt.subplots(figsize=(7.0, 3.2))
xx = np.linspace(19, 64, 200); ax.plot(xx, [trend(n) for n in xx], color='0.55', lw=1, ls='--', zorder=1)
for n in ns[1:]:
    if holder[str(n)] != holder[str(n - 1)]: ax.axvline(n - 0.5, color='0.88', lw=0.7, zorder=0)
for n in ns:
    f = holder[str(n)]; q = BEST[n][0] * n * n; open_ = n in changed
    ax.scatter(n, q, s=26, marker='s' if n <= 36 else 'o', zorder=3, lw=0.9 if open_ else 0.3,
               facecolor='white' if open_ else col[f], edgecolor=col[f] if open_ else 'k')
for f in fams:
    m = [n for n in ns if holder[str(n)] == f]
    if len(m) >= 3: ax.text(np.mean(m), max(BEST[n][0] * n * n for n in m) + 0.05, f, ha='center', fontsize=6.5, color=col[f])
ax.axvline(42.5, color=RED, lw=1, alpha=0.6); ax.text(42.8, 6.2, '$n = 43$', color=RED, fontsize=7, va='top')
ax.set_xlabel('$n$'); ax.set_ylabel('$n^2 t$'); ax.set_xlim(18.3, 64.7)
ax.legend(handles=[Line2D([], [], color='0.55', ls='--', lw=1, label='trend $2.26 + 0.95\\ln n$'),
                   Line2D([], [], marker='s', ls='', color='0.4', mec='k', mew=0.3, ms=5, label='published record'),
                   Line2D([], [], marker='o', ls='', color='0.4', mec='k', mew=0.3, ms=5, label='ours (family as of exp 85)'),
                   Line2D([], [], marker='o', ls='', mfc='white', mec='0.4', mew=0.9, ms=5, label='ours, improved since (family not reassigned)')],
          loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=6.5, ncol=2)
save(fig, 'fig_families')
q = {n: BEST[n][0] * n * n for n in ns}
print('fig_families: n^2 t range', round(min(q.values()), 3), round(max(q.values()), 3), 'max at', max(q, key=q.get),
      '; improved since exp 85:', changed)

# ---------------- geometry observables (exp 86 / 92 definitions) ----------------
C = np.array([.5, .5])
MAPS = {'R90': [[0, -1], [1, 0]], 'R180': [[-1, 0], [0, -1]], 'R270': [[0, 1], [-1, 0]], 'MX': [[-1, 0], [0, 1]],
        'MY': [[1, 0], [0, -1]], 'MD': [[0, 1], [1, 0]], 'MA': [[0, -1], [-1, 0]]}
def sym_rms(P):
    out = {}
    for k, M in MAPS.items():
        Q = (P - C) @ np.array(M, float).T + C; D = np.hypot(*(P[:, None, :] - Q[None, :, :]).transpose(2, 0, 1))
        r, c = linear_sum_assignment(D ** 2); out[k] = float(np.sqrt((D[r, c] ** 2).mean()))
    return out
def group(P, tol=0.02):                       # exp 92: RMS * sqrt(n) < tol
    e = sym_rms(P); s = {k for k, v in e.items() if v * math.sqrt(len(P)) < tol}
    if {'R90', 'MX'} <= s or {'R90', 'MD'} <= s: return 'D4'
    if 'R90' in s: return 'C4'
    if {'MX', 'MY'} <= s or {'MD', 'MA'} <= s: return 'D2'
    if 'R180' in s: return 'C2'
    if s & {'MX', 'MY'}: return 'm|'
    if s & {'MD', 'MA'}: return 'm/'
    return 'C1'
def layers(P):
    n = len(P); lab = -np.ones(n, int); left = np.arange(n); L = 0
    while len(left) > 0:
        if len(left) <= 2: lab[left] = L; break
        try: h = ConvexHull(P[left]).vertices
        except Exception: lab[left] = L; break
        lab[left[h]] = L; left = np.delete(left, h); L += 1
    return lab
OBS = {}
_PRE = os.path.join(FD, 'observables_records.json')
PRE = {int(k): v for k, v in json.load(open(_PRE)).items()} if os.path.exists(_PRE) else {}
for n in range(20, 65):
    P = BEST[n][1]
    if P is None: OBS[n] = PRE[n]; continue
    D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1)); np.fill_diagonal(D, np.inf)
    on = sum(int((np.abs(P[:, e] - v) < 1e-9).sum()) for e, v in ((0, 0.), (0, 1.), (1, 0.), (1, 1.)))
    OBS[n] = dict(nbr16=float((D < 1.6 / np.sqrt(n)).sum(1).mean()), gap=float(2 * np.hypot(*(P - C).T).min() * np.sqrt(n)),
                  wall_pts=on, sym_err=min(sym_rms(P).values()), K=int(layers(P).max() + 1), sym=group(P))

# ---------------- fig 2: transition observables ----------------
old = {}
for l in open(os.path.join(FD, 'observables.jsonl')):
    r = json.loads(l)
    if r.get('rank', 0) == 0: old.setdefault(r['n'], r)
diff86 = [n for n in range(20, 65) if abs(old[n]['q'] - BEST[n][0] * n * n) > 1e-6]
ns2 = list(range(20, 65))
obs = [('nbr16', 'neighbours\nwithin $1.6/\\sqrt{n}$'), ('gap', 'central hole\n$2\\delta\\sqrt{n}$'), ('wall_pts', 'points on\nthe walls'),
       ('sym_err', 'symmetry\nerror')]
fig, axs = plt.subplots(len(obs), 1, figsize=(6.4, 5.4), sharex=True)
for ax, (k, lab) in zip(axs, obs):
    y = [OBS[n][k] for n in ns2]
    if k == 'sym_err': ax.semilogy(ns2, np.maximum(y, 1e-17), 'o-', ms=2.8, lw=0.8, color='k')
    else: ax.plot(ns2, y, 'o-', ms=2.8, lw=0.8, color='k')
    ax.axvline(42.5, color=RED, lw=1, alpha=0.6); ax.set_ylabel(lab, fontsize=7)
axs[-1].set_xlabel('$n$')
save(fig, 'fig_transition')
mean = lambda k, a, b: np.mean([OBS[n][k] for n in range(a, b + 1)])
print('fig_transition: exp86 rows differ from current bests at', diff86)
for k, _ in obs[:3]:
    print(f'  {k}: mean 20-42 {mean(k, 20, 42):.3f}, 43-64 {mean(k, 43, 64):.3f}; exp86 20-42 '
          f'{np.mean([old[n][k if k != "gap" else "centre_gap"] for n in range(20, 43)]):.3f}, 43-64 '
          f'{np.mean([old[n][k if k != "gap" else "centre_gap"] for n in range(43, 65)]):.3f}')
print('  gap (current):', {n: round(OBS[n]['gap'], 2) for n in ns2})

# ---------------- fig 3: survival (exp 85 data) ----------------
S = [json.loads(l) for l in open(os.path.join(FD, 'pairs_survival.jsonl'))]
groups = [('within a family', [s['frac_kept'] for s in S if s['cat'] == 'cur_within'], BLUE),
          ('family handover', [s['frac_kept'] for s in S if s['cat'] == 'cur_handover'], RED),
          ('unrelated ($k = 1$, different families)', [s['frac_kept'] for s in S if s['cat'] == 'k1_cross'], '0.55')]
fig, ax = plt.subplots(figsize=(6.4, 2.6)); bins = np.linspace(0, 1, 26)
for lab, v, c in groups:
    ax.hist(v, bins=bins, density=True, histtype='step', lw=1.6, color=c, label=f'{lab}: median {100 * np.median(v):.0f}% ($N = {len(v)}$)')
ax.set_xlabel('fraction of minimal triangles that survive one growth step'); ax.set_ylabel('density')
ax.legend(frameon=False, fontsize=7)
save(fig, 'fig_survival')
w, h, u = (np.array(g[1]) for g in groups)
auc = lambda a, b: float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
dco = lambda a, b: (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
bw = np.array([s['frac_kept'] for s in S if s['cat'] == 'best_within']); bh = np.array([s['frac_kept'] for s in S if s['cat'] == 'best_handover'])
de = np.array([s['frac_kept'] for s in S if s['cat'].startswith('descent')])
print(f'fig_survival: current-best within/handover/unrelated medians {np.median(w):.3f} {np.median(h):.3f} {np.median(u):.3f}; '
      f'AUC {auc(w, h):.3f} d {dco(w, h):.2f}; best-pair medians {np.median(bw):.3f} {np.median(bh):.3f} AUC {auc(bw, bh):.3f} '
      f'd {dco(bw, bh):.2f}; descents median {np.median(de):.3f}')

# ---------------- fig 4: phase diagram ----------------
txt = open(os.path.join(FD, 'phylogeny.txt')).read()
import re
cur = dict((int(a), b) for a, b in re.findall(r'(\d+):(\S+)', [l for l in txt.splitlines() if l.startswith('current')][0]))
rows = [dict(n=n, K=OBS[n]['K'], gap=OBS[n]['gap'], topo='empty' if OBS[n]['gap'] >= 1.1 else 'filled', sym=OBS[n]['sym'],
             fam=cur.get(n, '?')) for n in range(20, 65)]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.6], hspace=0.08))
for a in (ax, ax2): a.spines['left'].set_color(MUTED); a.spines['bottom'].set_color(MUTED); a.tick_params(colors=MUTED)
for r in rows:
    c = BLUE if r['topo'] == 'empty' else ORANGE
    ax.add_patch(plt.Rectangle((r['n'] - 0.45, r['K'] - 0.42), 0.9, 0.84, color=c, alpha=0.85 if r['topo'] == 'empty' else 0.75, lw=0))
    if r['sym'] != 'C1': ax.text(r['n'], r['K'] + 0.5, r['sym'], ha='center', va='bottom', fontsize=6, color=INK, rotation=90)
fb = []
for r in rows:
    if fb and fb[-1][0] == r['fam']: fb[-1][2] = r['n']
    else: fb.append([r['fam'], r['n'], r['n']])
for f, a, b in fb:
    ax.plot([a - 0.4, b + 0.4], [2.35, 2.35], color=MUTED, lw=1.1)
    ax.text((a + b) / 2, 2.27, f if f.startswith('F') else '·', ha='center', va='top', fontsize=5.5 if f.startswith('F') else 8, color=MUTED)
ax.text(19.0, 2.35, 'family\n(exp 85)', ha='right', va='center', fontsize=6, color=MUTED)
Kmax = max(r['K'] for r in rows)
ax.set_ylim(1.9, Kmax + 1.7); ax.set_yticks(range(3, Kmax + 1)); ax.set_ylabel('convex layers $K$')
ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.axvline(42.5, color=MUTED, lw=0.8, ls=(0, (3, 3))); ax.text(42.7, Kmax + 1.6, '$n = 43$', fontsize=7, color=MUTED, va='top')
ax.legend(handles=[Patch(color=BLUE, label='empty centre ($2\\delta\\sqrt{n} \\geq 1.1$)'), Patch(color=ORANGE, alpha=0.75, label='filled centre')],
          loc='upper left', frameon=False, fontsize=7)
g = np.array([r['gap'] for r in rows]); E = np.array([r['topo'] == 'empty' for r in rows]); nn = np.array([r['n'] for r in rows])
ax2.plot(nn, g, color=MUTED, lw=1.1, zorder=1)
ax2.scatter(nn[E], g[E], s=20, color=BLUE, edgecolor='white', lw=1, zorder=2); ax2.scatter(nn[~E], g[~E], s=20, color=ORANGE, edgecolor='white', lw=1, zorder=2)
ax2.axhline(1.1, color=MUTED, lw=0.8, ls=(0, (3, 3)))
ax2.set_ylabel('$2\\delta\\sqrt{n}$'); ax2.set_xlabel('$n$'); ax2.grid(axis='y', color=GRID, lw=0.6)
ax2.set_xticks(range(20, 65, 4)); ax2.set_xlim(19, 65)
save(fig, 'fig_phase')
topo = ''.join('E' if r['topo'] == 'empty' else 'f' for r in rows)
sw = [rows[i]['n'] for i in range(1, len(rows)) if rows[i]['topo'] != rows[i - 1]['topo']]
print('fig_phase: topology 20..64', topo, '; switches at', sw)
print('  K:', {r['n']: r['K'] for r in rows})
print('  symmetric:', {r['n']: r['sym'] for r in rows if r['sym'] != 'C1'})
old92 = {r['n']: r for r in json.load(open(os.path.join(FD, 'phase.json')))}
if '--dump-record-observables' in sys.argv:
    json.dump({n: OBS[n] for n in range(20, 37)}, open(_PRE, 'w'), indent=1); print('wrote', _PRE)
print('  differs from exp 92 (K/topo/sym):', [n for n in range(20, 65) if (old92[n]['K'], old92[n]['topo'], old92[n]['sym']) != (OBS[n]['K'], 'empty' if OBS[n]['gap'] >= 1.1 else 'filled', OBS[n]['sym'])])
