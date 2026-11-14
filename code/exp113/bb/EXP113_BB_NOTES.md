# Experiment 113 / bb: exact branch-and-bound for the Heilbronn problem in the triangle (2026-11-13)

Port of the exp 110 B&B (`110-n9-proof/bb2.py`, which proved the square n = 9) and the parts of exp 103 `bb.py` it
inherits. Self-contained copy: exp 103 / 110 are not imported and were not modified.

Frame: Omega = {x >= 0, y >= 0, x + y <= 1} (area 1/2); f(P) = min over triples of |area|; site value = 2f.
Known f*: n = 5: (3 - 2 sqrt 2)/2; n = 6: 1/16; n = 7: 7/144 (Zeng-Chen 2019); n = 8: 0.0338946066270986 (numerical;
septic conjectured; certified local optimum in `../cert/local_certs_tri.json`).

## Files
| file | content |
|---|---|
| `bbt.py` | `ProblemT`: numba kernels (P0 triangle propagation, P1, branching, McCormick fill, certificate matvec), hot-started HiGHS P2 with exact integer certificate (+ Farkas case), Fraction audit `audit2`, S3 cover, thresholds |
| `run_small_t.py` | serial driver (validation) |
| `run_par_t.py` | parallel checkpointed driver (copy of run_par2; tasks carry c; `--max-hours`) |
| `probe_t.py` | Knuth random-probe tree-size estimate (estimates only) |
| `test_sound_t.py` | randomised implementation test near the optimum and its 6 S3 images (P0 / P1 / sign / P2 bound / audit / cover) |
| `test_p0_t.py` | random P0 test with exact integer domain samples |
| `results/` | outputs (`val_*`, `run_n7_exact/`, `run_n8_exact_sample/`, `probe_*`, `test_*`) |

## Reductions (proved before use)
Notation: A = sum x_i, B = sum y_i, C = sum (1 - x_i - y_i) (barycentric sums w.r.t. the vertices X = (1,0), Y = (0,1),
O = (0,0)).

**Lemma W (walls).** Let P in Omega^n with f(P) > 0. There is P' in Omega^n with f(P') = lambda f(P), lambda >= 1,
having a point on each of the three sides; lambda = 1 iff P already touches all three sides (then P' = P).
*Proof.* (i) Left side: let delta = min x_i. If delta > 0: delta < 1 (else all points are (1,0) and f = 0). The affine
map a(x, y) = ((x - delta)/(1 - delta), y/(1 - delta)) sends {x >= delta} cap Omega into Omega
(x' >= 0, y' >= 0, x' + y' = (x + y - delta)/(1 - delta) <= 1), has determinant 1/(1 - delta)^2 > 1, so multiplies every
area by that factor; the point(s) with x = delta land on x = 0. Points with y = 0 keep y' = 0 and points with x + y = 1
keep x' + y' = 1, so the map preserves contacts with the other two sides. (ii) Bottom side: the same with x and y
exchanged, (x, y) -> (x/(1 - delta), (y - delta)/(1 - delta)), delta = min y_i; preserves contacts with x = 0 and x+y = 1.
(iii) Hypotenuse: s = max (x_i + y_i) > 0 (else all points at O); if s < 1 the homothety p -> p/s maps P into Omega,
multiplies areas by 1/s^2 > 1, puts a point on x + y = 1 and keeps x = 0, y = 0 contacts. Applying (i), (ii), (iii) in
this order, each step keeps the contacts made before it. []

**Lemma S (S3).** The 6 affine maps of Omega onto itself (vertex permutations) permute the barycentric coordinates
(x, y, 1-x-y) of every point, hence permute (A, B, C); they have |det| = 1, so preserve all areas and f, and they permute
the three sides, so preserve "touches all three sides". So wlog A >= B >= C.

**Lemma N (normalisation).** For every P in Omega^n with f(P) > 0 there is P' = relabel(g(stretch(P))), g in S3, with
f(P') = lambda f(P), lambda >= 1, such that: x_0 <= ... <= x_{n-1}; x_0 = 0; y_b = 0 for some b; x_c + y_c = 1 for some c;
A >= B >= C; and x_i + y_i <= 1, x_i, y_i >= 0 for all i. If lambda = 1, P' is an S3 image of P, relabelled.
*Proof.* Lemma W, then Lemma S, then sort labels by x (relabelling preserves everything). The point on x = 0 has minimal
x, so x_0 = 0. []
The root boxes impose x_0 = 0 (hi[0] = 0), y_b = 0 (hi[n+b] = 0) and the equality x_c + y_c = 1 (enforced in P0 and as
LP rows), one root per (b, c) in {0..n-1}^2 (n^2 roots; b = c and b = 0, c = 0 allowed; overlaps only cost time).
Dropped from the square: x_{n-1} = 1, mean(x) <= 1/2, y-mirror, diagonal Q_x >= Q_y (no proved triangle analogues).

## Tests and why each keeps soundness (changes vs bb2)
Boxes: integer endpoints on the grid D = 2^24 in [0, D]; points inside are arbitrary reals (unchanged).

**P0 (`k_propagate`, exact int64).** Constraints of Lemma N: sorted x; x_i + y_i <= D; x_c + y_c >= D;
sum y <= sum x; sum x + 2 sum y >= nD (B >= C rewritten: n - A - B <= B). New bounds:
hi[x_i] <= D - lo[y_i], hi[y_i] <= D - lo[x_i]; lo[x_c] >= D - hi[y_c], lo[y_c] >= D - hi[x_c];
hi[y_i] <= sum hi_x - sum_{j != i} lo_y; lo[x_i] >= sum lo_y - sum_{j != i} hi_x;
lo[x_i] >= nD - 2 sum hi_y - sum_{j != i} hi_x; lo[y_i] >= floor((nD - sum hi_x - 2 sum_{j != i} hi_y)/2).
Each is implied for every real point of the box satisfying the constraints (the sums are bounded by box endpoints), all
are integers, and the single halving is rounded down for a lower bound (outward), so no domain point is removed;
emptiness is declared only when some lo > hi or a sum test is violated by the box extremes. Magnitudes <= 3nD < 2^30.
(Lemma A.1 of the paper: P0 only raises lo / lowers hi and returns empty iff lo > hi: still true.)

**P1 (`k_ranges`).** Unchanged. The corner range of 2A over the box bounds 2A on box cap Omega, so Lemma "orient" holds.

**P2 (`lp_bound2` / `_exact`).** Same fixed hot-started model; rows (integer form, D^2 x natural rows):
(R1) area rows (both orientations, on iff orientation determined), (R2) 4 McCormick rows per pair, (R3) ordering
D X_i - D X_{i+1} <= 0, **(R4)** D X_i + D Y_i <= D^2 (n rows, always valid on Omega), **(R5)** -D X_i - D Y_i <= -D^2
(n rows, switched on only for i = c), **(R6)** -D sum X + D sum Y <= 0 (A >= B), **(R7)** -D sum X - 2D sum Y <= -nD^2
(B >= C). The mean row of the square is removed. The certificate zeroes the multiplier of every row that is not valid on
the box: switched-off area rows and the R5 rows with i != c (`active_rows`; asserted `not mu[~act].any()`).
*Lemma P2' (replaces Lemma A.4).* B a box, sigma its pattern, mu >= 0 integer, zero on rows not valid on B, r = G^T mu,
s = r_tau, N = mu^T H + sum_{c >= 1} max(-r_c lo_c, -r_c hi_c). If (s > 0 and N <= 2TD^2 s) or (s = 0 and N < 0), no
P in B cap Omega_N (Omega_N = normalised domain with this c) has f(P) > T.
*Proof.* As Lemma A.4: at the intended values every row with mu > 0 holds (area rows by Lemma orient, McCormick since P in
B, R3-R7 since P in Omega_N), so s tau^ <= N. If s > 0 this gives tau^ <= 2TD^2, contradicting f > T. If s = 0 it gives
0 <= N, contradicting N < 0, i.e. B cap Omega_N is empty. []
The s = 0 case is new: about 0.3 % of LPs are infeasible (the linear rows R3-R7 + box are inconsistent although interval
propagation did not notice); HiGHS' dual ray (both signs tried) is then fed to the same exact test. Also new: tau has no
lower bound in the float LP (the certificate never uses tau's bounds; with tau >= 0 HiGHS reported 'infeasible' for
boxes where all determined sigma*2A are forced negative, which are then pruned through the ordinary s > 0 test).
*Overflow (A.5).* Entries |G| <= 2D (R7 has 2D on Y), McCormick entries <= D. Instead of count * D the code computes, per
column, the sum of the coefficient bounds and asserts colsum_max * 2^25 < 2^62. n = 8: X_i columns (inner labels)
4*7 McCormick + 2 ordering + R4 + R5 + R6 + R7 = 34 D; Y columns 28 + 1 + 1 + 1 + 2 = 33 D; colsum_max = 34 D (computed
and printed), so every partial sum is < 34 * 2^24 * 2^25 < 2^55. tau column: 2m = 112 nonzeros of size 1. |H| <= nD^2 = 2^51.
*Audit (`audit2`).* Rebuilt from row definitions in Fractions, including R4-R7 and the s = 0 rule; rejects any nonzero
multiplier on an invalid row.

**Cover (`covered`).** Basin balls of `../cert/local_certs_tri.json` (R = rho_b'(1 - 1e-6) - rho - 2^-60, as exp 103),
tested against all 6 S3 images. The S3 maps are affine but not sup-norm isometries in this frame: with
g^{-1}(p) = M p + v, ||M||_inf (max row sum) is 1 for the identity and the swap (x, y) -> (y, x), and 2 for the four maps
that move the hypotenuse (e.g. (x, y) -> (1 - x - y, y) has M = [[-1, -1], [0, 1]]). These norms are computed exactly and
asserted at import (`OPNORM`). The test does not rely on them: for a proposed g and matching j it maps **the 4 corners of
every point rectangle** back with g^{-1} in exact rationals and requires || g^{-1}(q) - x~_{j(i)} ||_inf <= R. Since
p -> |(g^{-1}p - x~)_k| is convex, the corners bound every point of the rectangle, so every P in the box has
|| g^{-1}(P) - pi(x~) ||_inf <= R and hence || g^{-1}(P) - pi(x*) || <= R + rho < rho_b: the conclusion is drawn in the
frame of the certificate, with no factor 2 lost and none missing. (A box of sup-radius R around g(x~) in the image frame
would only give 2R after mapping back; the code never uses that.) Assumption: the certificate's basin radius is a
sup-norm radius in this frame (as in exp 44 / the cert notes). n = 6 has no basin radius (second-order certificate):
exact mode n = 6 is not available.

**LP skip, branching, driver.** Unchanged (skip never discards). Tasks carry c.

**Thresholds.** eps: T >= f*_upper (1 + eps) (n = 5 uses a rational upper bound of (3 - 2 sqrt 2)/2 via isqrt), claim
max f <= T. neg: T <= f*_lower (1 - eps), no covers. exact: T = t_lower (1 - 1e-9) rounded down to a double, covers on.

## Validation (all zero failures except the negative control, which must fail)
| run | claim | nodes | P0 / P1 / P2 (Farkas) / cover | time |
|---|---|---|---|---|
| n = 5 eps 1e-3, audit 100 % | max f <= f*(1+1e-3) | 32 587 | 75 / 5 422 / 10 809 (96) / 0; 10 809 audits agree; lp_fail 0 | 20 s, 1 core |
| n = 6 eps 1e-3, audit 5 % | max f <= 1/16 (1+1e-3) | 164 550 | 148 / 29 218 / 52 927 (43) / 0; 2 576 audits; lp_fail 0 | 89 s, 1 core |
| n = 7 exact, skip 1.2, audit 0.2 %, 12 workers | max f = 7/144, maximiser unique up to S3 | 3 100 671 | 4 868 / 628 342 / 916 990 (6 797) / 160; 1 824 audits; lp_fail 0; closed_w 1 + 7e-15 | **325 s wall** |
| n = 5 neg (T = f*(1-1e-3)), no covers | must fail | aborted after 5 fails at 6 638 nodes | fails at grid-floor boxes next to the optimum (min area there 0.0857006 > T) | 2.5 s |

- n = 7 exact: `results/run_n7_exact/` (DONE, fail 0, `fails.jsonl` empty). With Lemma N, the local certificate
  (n = 7, first order, basin 3.14e-4) and the argument of the square paper's Theorem 1.1 proof (stretch factor), this
  gives: every 7 points in Omega have f <= 7/144, with equality only at the S3 images of the certified configuration,
  computer-assisted, consistent with Zeng-Chen 2019. The run started a few minutes before the `OPNORM` assertion was added
  to `bbt.py`; the cover test itself (exact corner mapping) was identical.
- `test_sound_t.py` n = 6, 7, 8 (600 boxes each, near the optimum and all 6 S3 images): P0 / P1 / sign / P2-bound /
  audit / cover violations all 0 (n = 7: 276 LP boxes, 66 covered boxes; n = 8: 150 / 31; n = 6: 247 / -).
- `test_p0_t.py` n = 8: 20 000 random boxes around exact integer domain points, 190 688 domain samples kept, 0 removed,
  0 boxes declared empty that contained a sample.

## n = 8 estimate (exact mode: T = t_lower(8)(1 - 1e-9), certified cover, skip 1.2)
Knuth probes (`probe_t.py`, `results/probe_n8_*.txt`); estimates are heavy-tailed (single deep paths weigh 2^depth * 64):
| batch | probes | mean nodes | se |
|---|---|---|---|
| seeds 0 / 1 / 2 | 4 000 each | 1.8e10 (one path of weight 7e13) / 2.2e8 / 3.3e7 | 1.8e10 / 1.4e8 / 8.5e6 |
| seeds 10-21 | 10 000 each | 0.67e8 - 3.1e8 (median of the 12 batch means 1.7e8) | 2e7 - 2e8 each |
Pooled over all 132 000 probes: 6.9e8 (dominated by the one 7e13 path); without seed 0: 1.6e8.
Calibration on n = 7 (same code, 4 000 probes): estimate 2.3e6 vs actual 3.10e6 (probes underestimate by ~1.3x).
Throughput: timed 14-worker sample run (`results/run_n8_exact_sample/`, 10 min, stopped by `--max-hours`, checkpoint kept,
resumable with `python run_par_t.py 8 exact --workers 14 --skip 1.2 --audit 0.002 --tag n8_exact_sample`): 4.13e6 nodes,
**~6 900 nodes/s**, fail 0, lp_fail 0, 3 002 audits ok; closed width only 0.118 (the width tally is front-loaded and is
not a reliable ETA). Single-thread cost 0.53-0.81 ms/node (probe timing, with 3 vs 12 concurrent processes).
**Estimate: about 2e8 nodes (plausible range 1e8 - 7e8, heavy tail), i.e. about 8 h wall with 14 workers
(range 4 - 28 h).** For comparison: square n = 8 needed 4.3e6 nodes; the triangle tree is ~50x larger (n^2 roots instead
of n(n-1)/2, no diagonal / mean reductions, and the n = 6 eps tree is already 16x the square's).

## Soundness doubts / open points
- The cover assumes the certificate's basin radius is a sup-norm radius in this frame (exp 44 convention; cert agent
  should confirm). The test maps corners back exactly, so no operator-norm factor is involved.
- New proof steps vs the reviewed square code: P0 triangle rules, rows R4-R7, the Farkas (s = 0) case and the conditional
  R5 rows. All are covered by Lemma P2' and checked by audit2 / test_sound_t / test_p0_t, but not independently reviewed.
- n = 7 exact: the Zeng-Chen value is reproduced; the equality statement relies on Lemma W's lambda = 1 argument exactly
  as in the square paper.

## Result (2026-11-14 01:32): triangle n = 8 PROVED
DONE after 34 646 s (9.6 h, 14 workers): 243 871 662 nodes, fail 0, lp_fail 1 (box split, harmless), fails.jsonl and
stderr empty, 183 861 rational audits all agreed, 8 boxes closed by the basin cover, 242 117 Farkas (infeasible-box)
prunes. Exact checks: leaves (p0 + p1 + p2 + cover + fail = 121 935 863) minus internal nodes (121 935 799) = 64 = n^2
root boxes; T = 2442363801968369/2^56 <= t_lower (1 - 1e-9) < t_lower with t_lower from cert/local_certs_tri.json
(the stats field "t_lower" holds a looser pair of bounds, not the certificate's). closed_w float tally 1 - 1e-15.
Statement: every 8 points in the triangle {x, y >= 0, x + y <= 1} have min triangle area <= t*, with equality only at
the known configuration up to the 6 symmetries; t* = 2 f* normalized is the unique real root of the site's septic
(Chen-Zeng-Zhou conjecture), identified exactly in cert/identify_n8.py. Computer-assisted, AI-assisted, not yet
independently checked. Prior: Sudermann-Merx arXiv:2607.15021 (optimal within solver tolerances).
