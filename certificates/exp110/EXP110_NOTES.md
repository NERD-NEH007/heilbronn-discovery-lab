# Experiment 110: fast rigorous branch-and-bound for n = 9 (2026-11-12)

Goal: make the exp 103 rigorous B&B fast enough for n = 9 and run it. exp 103 is imported unchanged
(`bb.py` for D, symmetries, covers, root boxes; `results/local_certs.json` for the exp 44 certificates).
Statement, thresholds and claims are exactly those of exp 103 (exact mode: T = t_lower(1 - 1e-9) rounded down,
every surviving box covered by the exp 44 basin ball => max min-area = t*, maximiser unique up to D4).

## Files
| file | content |
|---|---|
| `bb2.py` | `Problem2(bb.Problem)`: numba kernels (P0, P1, branching, certificate matvec), hot-started HiGHS P2, diag reduction, LP skipping, Fraction audit `audit2` |
| `run_small2.py` | serial driver (validation) |
| `run_par2.py` | parallel checkpointed driver (single atomic `state.pkl`, STOP file, per-minute progress with ETA) |
| `probe2.py` | Knuth random-probe tree-size estimate (estimates only) |
| `test_sound2.py` | randomised implementation test (not part of the proof) |
| `launch_n9.ps1`, `RESUME.md` | detached n = 9 run: launch / resume / check / stop |
| `results/val/`, `results/run_n8_exact/`, `results/probe/`, `results/run_n9/` | outputs |

Package installed: `numba` 0.68.1 (compiles the per-node kernels).

## Changes and why each keeps the proof exact
(a) **Hot-started LP.** One persistent HiGHS model with a fixed structure: columns tau, x, y, all n(n-1) products w;
rows: two area rows per triple (tau - s 2A(w) <= 0 for s = +1 and s = -1), 4 McCormick rows per pair, ordering rows,
mean(x) <= 1/2. Per node only the McCormick coefficients/rhs that changed (`changeCoeff`), the row uppers (area row
"on" = upper 0 iff the triple's orientation is determined as s on the box, else +inf) and the column bounds are
updated; HiGHS keeps the basis (about 20 simplex iterations per node). Soundness: the LP still only proposes duals.
The certificate zeroes the multiplier of every switched-off area row before rounding, so only rows valid on the box
enter mu^T(row); McCormick/ordering/centroid rows are valid for every point of the box/domain. The integer
certificate (rows scaled by D^2, multipliers floor(lambda 2^S) < 2^50, int64 matvec in two 25-bit halves with a
static overflow bound asserted at construction, final sums in Python integers, comparison N q <= s p with
2T D^2 = p/q exactly) is otherwise identical to exp 103.

(b) **Compiled node code** (numba): P0 propagation, P1 corner ranges, the branching rule, McCormick fill and the
certificate matvec. P1 is now computed in **int64** at grid units (2A D^2 = (xj-xi)(yk-yi)-(xk-xi)(yj-yi) with
|.| < 2^50), compared with floor(2T D^2) — exact without the float-exactness argument of exp 103.

(c) **Diagonal symmetry** (new reduction): wlog Q_x := sum (x_i - 1/2)^2 >= Q_y := sum (y_i - 1/2)^2.
*Lemma.* If some configuration has min area > T, one exists that touches all four walls and satisfies, with
labels sorted by x: x_0 = 0, x_{n-1} = 1, y_b = 0, y_t = 1, b < t (b, t smallest labels on bottom/top wall),
mean(x) <= 1/2, and Q_x >= Q_y.
*Proof.* Stretch to touch all four walls (exp 103; areas only grow). Touching all four walls is invariant under D4,
and every element of D4 preserves all areas. Apply the diagonal swap (x, y) -> (y, x) if Q_x < Q_y; it exchanges
Q_x and Q_y. Then apply x -> 1 - x if mean(x) > 1/2: this preserves Q_x, Q_y (each term (x-1/2)^2 is invariant) and
the y's. Then apply y -> 1 - y if b > t: this swaps the bottom and top walls, preserves Q_x, Q_y and all x's (so
mean(x) and the x-sorted labelling are unchanged). Each later step preserves the conditions established before it,
so all hold; finally relabel by x. Box [] The covers are tested against all 8 images g(x*), so the exact-mode
conclusion (any surviving configuration is near some g(x*)) is unaffected.
*Use (exact integers, P0):* with a_i = 2x_i - D, b_i = 2y_i - D in grid units, A_max = sum max over the box of
a_i^2, B_min = sum min of b_i^2: prune if A_max < B_min; tighten y_i to |b_i| <= sqrt(A_max - B_min + Bmin_i) and
x_i away from 1/2 when |a_i| >= sqrt(B_min - A_max + Amax_i) excludes one side; all square roots via exact isqrt with
outward rounding (new bounds only ever enlarge the kept set). Effect: -10 % nodes at n = 6, -0.6 % at n = 7,
-3 % at n = 9 (paired probes): the hard mid-size boxes rarely decide the sign of Q_x - Q_y.

(d) **LP skipping** (`--skip 1.2`): if the last solved ancestor (the parent) had LP value > 2T (1 + 1.2), the node
skips the LP and is split; its children always solve their LP (no two skips in a row). A skip never closes a box,
so soundness is untouched (it can only cost extra nodes). Calibration: at n = 9, 36 % of LPs have parent gap > 1.2
and only 7 % of P2 prunes come from them; paired probes (same seed): nodes 2.30e8 -> 2.31e8, time per node -21 %.
(skip 0.5: +12 % nodes; rejected.)

## Validation (all with the new code; zero failures everywhere)
| run | claim | nodes (exp 103) | P1 / P2 / covers | time |
|---|---|---|---|---|
| n = 5 eps 1e-3, no diag, audit 100 % | max <= t*(1+1e-3) | 1 120 (1 120) | 251 / 314 identical to exp 103; 314/314 audits ok | 2.0 s |
| n = 6 eps 1e-3, no diag / diag | max <= t*(1+1e-3) | 9 973 (9 973) / 8 979 | 2 115 / 2 803 identical / 1 932 / 2 489 | 4.3 s / 3.2 s (exp 103: 15-19 s) |
| n = 7 exact, no diag | max = t*(7), unique up to D4 | 857 707 (1.17 M, older build without mean(x) cut) | 130 022 / 294 510 / 24 | 599 s, 1 core (*) |
| n = 7 exact, diag | same | 852 699 | 129 289 / 292 058 / 16 | 600 s (*) |
| n = 7 exact, diag + skip 1.2 / 0.5 | same | 853 071 / 857 097 | covers 16 / 16 | 590 / 589 s (*) |
| n = 8 exact, diag + skip 1.2, 14 workers, audit 0.2 % | max = t*(8), unique up to D4 | 4 252 598 (4 256 822) | 740 009 / 1 368 375 / 64 ; 2 701 audits ok; lp_fail 0 | **538 s wall** (exp 103: 4 204 s on 3 workers) |

(*) four n = 7 runs and six probes shared the machine, so the single-core times are inflated (~0.70 ms/node).
Fewer covers with diag: covers are the D4 images of the optimum, the diag cut removes half of them.

- `test_sound2.py` n = 7, 8, 9 (600 boxes each, near the optimum and its mirrored images): P0 / P1 / sign / P2-bound /
  audit violations all 0 (210 / 270 / 310 LP boxes certified; LP bound - best sample median 0.014-0.019, min 2e-5).
- Random P0 test (20 000 sorted-x boxes, n = 9, diag active on 3 862): 0 of 5.6 M domain samples removed; no
  domain sample in any box declared empty.
- Negative control n = 5, T = t*(1 - 1e-3), no covers: **fails as it must** (diag: 5 failures after 509 nodes;
  no diag: after 438 nodes), at grid-floor boxes next to the optimum (`results/val/n5_negative_control*.txt`).
  (A first attempt hung because `run_small2.py` only checked the failure limit between roots; `solve2` now takes
  `max_fail`. Driver-only change, not used by the n = 9 run, which logs every failure and never aborts.)

## Speed
- Single process, n = 9 (same 4 000 nodes, identical prune counts): 1.62 ms/node (exp 103) -> 0.58 ms/node
  (a + b; HiGHS run 1.4 -> 0.7 ms per LP), then -21 % per node with (d): about **3.4x per thread**.
- Parallel: n = 9 throughput 860 nodes/s (exp 103, 3 workers) -> ~6 100 nodes/s (14 workers on 8 cores / 16 threads);
  n = 8 wall time 4 204 s -> 538 s.

## n = 9 estimate and run
Knuth probes (4 000 each, diag): 3.47e8 +- 0.37e8, 3.09e8 +- 0.35e8, 2.50e8 +- 0.31e8 -> about 3.0e8 nodes
(heavy-tailed; exp 103 got 2.5-4.2e8). At ~6 100 nodes/s: **about 14 h (range 11-19 h)**.
Launched 2026-11-12 20:31 as a detached hidden process (`launch_n9.ps1`, Start-Process), state in
`results/run_n9/`; see `RESUME.md`. A first launch through WMI `Win32_Process.Create` died after 5 s (its pool
workers then failed with DuplicateHandle access denied); it had no checkpoint and was discarded.
The proof of n = 9 is complete only when the run reports DONE with fail = 0 and closed_w = 1.

## Result (2026-11-13 12:31): n = 9 PROVED
DONE after 57 636 s (16.0 h): 358 594 738 nodes, `fail 0`, `lp_fail 0`, `closed_w` = 1 (float tally 1 + 1e-13),
`fails.jsonl` empty, stderr empty, 21 286 Fraction audits of P2 certificates all agreed, 32 boxes closed by the
exp 44 basin ball (D4 images of the optimum, halved by the diag cut). Prune split: P0 2.5e6, P1 7.10e7 (corner,
exact int64), P2 1.06e8 (LP, exact dual check), 6.3e6 LP skips.
Statement (as exp 103, exact mode): every configuration of 9 points in the unit square has minimum triangle area
<= t* = (9 sqrt 65 - 55)/320 = 0.054875999170896..., with equality only at the known configuration up to the
symmetries of the square. Computer-assisted: HiGHS only proposes duals; every discard is decided in exact integer
or rational arithmetic; the reductions (x-sort, mean(x) <= 1/2, y-mirror, Sigma(x-1/2)^2 >= Sigma(y-1/2)^2) are
proved above; the basin-ball uniqueness is the exp 44 interval certificate. Not yet independently reviewed.
Earlier work: Chen-Xu-Zeng (bounds), Sudermann-Merx and Monji-Modir-Kocuk (MINLP, optimal within solver tolerances).
