# Exp 127: cell-graph proof for Heilbronn's problem with 9 points in a triangle (2026-11-14) -- proof in progress

Frame: Ω = {x ≥ 0, y ≥ 0, x + y ≤ 1}, with area 1/2. Φ(P) = min |area| over triples (raw). The site's
normalized value is 2Φ. Vertices are O = (0,0), X = (1,0), Y = (0,1). Sides:
- B: y = 0, parameter x, from O to X;
- L: x = 0, parameter y, from O to Y;
- H: x + y = 1, parameter x, from Y (x = 0) to X (x = 1).

The midpoints are mB = (1/2,0), mL = (0,1/2), mH = (1/2,1/2).

The code is our own: `cgt.py` is adapted from `126-cellgraph/cg.py` (the original is kept unmodified as
`cg_square_orig.py`), and `cert/certify_tri.py` is copied from exp 113. **Nothing in `external/` was opened or run.**

## 1. Normalization lemma (proved here before use)

**Lemma N1 (existence).** Let K be a convex polygon with nonempty interior. Then a triangle of minimum area containing K
exists.

*Proof.*
1. Fix one enclosing triangle with area A0. Restrict to the enclosing triangles of area ≤ A0.
2. K contains a disk of radius r > 0. Any such triangle contains that disk, so its inradius is ≥ r.
3. Area = inradius · semiperimeter, so the perimeter is ≤ 2A0/r.
4. The triangle contains K, so its vertices lie within distance 2A0/r of K.
5. The vertex triples therefore range over a compact set. Containing K is a closed condition, and area is continuous,
   so the minimum is attained. A nondegenerate limit is guaranteed because area ≥ πr² > 0. □

**Lemma N2 (midpoint contact; Klamkin–Chakerian 1982, O'Rourke et al. 1986).** Let T be a locally area-minimal triangle
enclosing a convex polygon K. Then the midpoint of every side of T lies in K.

*Proof.*
1. Let s = [A,B] be a side on the line ℓ, and let V be the opposite vertex. If K ∩ ℓ = ∅, translating ℓ towards V
   decreases the area, so K ∩ ℓ = [u, v] is a nonempty segment of s.
2. Suppose the midpoint m ∉ [u, v]. Without loss of generality [u, v] lies in the half [A, m), so |vA| < |vB|.
3. Rotate ℓ about v by a small angle θ > 0, in the sense that moves the A-part outward and the B-part inward. Call the
   result ℓ_θ.
4. K is still on the correct side of ℓ_θ:
   - points of K ∩ ℓ lie in [u, v], on the outward-moving part or at the pivot v;
   - the vertices of K off ℓ are at distance ≥ δ > 0 from ℓ, which survives small θ;
   - so K, the hull of its vertices, is still inside.
5. The new triangle T_θ (sides AV and BV unchanged, ℓ replaced by ℓ_θ) still contains K. Its area is
   area(T) + area(vAA') − area(vBB'), where A', B' are the new vertices on the lines VA, VB.
6. The sine rule gives area(vAA') = ½|vA|² sin θ sin α / sin(α − θ) = ½|vA|²θ + O(θ²), with α the angle of T at A, and
   the same for B.
7. So d area/dθ at 0⁺ equals ½(|vA|² − |vB|²) < 0. This contradicts local minimality. □

(O'Rourke et al. also prove that some side is flush with a hull edge. **We do not use this**: the family MMM below, which
it would exclude, is searched anyway.)

**Lemma N3 (normalization).** Let P be 9 points in a triangle Δ with Φ(P) > 0, and let K = conv P. Let T be a
minimum-area triangle enclosing K (Lemma N1), and let A be the affine map with A(T) = Ω.
- Φ(AP)/area(Ω) = Φ(P)/area(T) ≥ Φ(P)/area(Δ), because area(T) ≤ area(Δ) (Δ encloses K).
- Ω is a minimum enclosing triangle of A(P), since affine maps preserve area ratios. So by N2, every midpoint
  m_s ∈ conv(AP) ∩ s = conv(AP ∩ s).
- Therefore Δ9(triangle) = sup Φ over the set 𝒩 of P ⊂ Ω with m_s ∈ conv(P ∩ s) for all three sides s.
- If P attains the supremum, then area(T) = area(Δ). □

**Contact families (exhaustive case analysis).** Let P ∈ 𝒩 with Φ(P) > 0.
- A side contains ≤ 2 points of P, since 3 collinear points give Φ = 0. Every side contains ≥ 1 point, its contact.
- A point lies on two sides only at a vertex, and never on three.
- So each side is of one of two types:
  - **M**: exactly one point of P lies on s, and it is the midpoint;
  - **S**: exactly two points lie on s, with parameters u ≤ 1/2 ≤ v (closed; a point exactly at the midpoint is allowed).
    S sides are the flush sides.
- A vertex point lies on both adjacent sides, so both must be of type S, and it is their contact in the half next to
  that vertex.

The patterns are (side types, set of occupied vertices between two S sides). The S3 group acts on them. The 6 affine
maps of Ω (vertex permutations, |det| = 1) preserve Φ and 𝒩, and permute the sides, midpoints and vertices. The 8
orbits, with the representative used, are:

| family | supports | params | free | dim |
|---|---|---|---|---|
| MMM | mB, mL, mH | 0 | 6 | 12 |
| MMS | B1 ∈ [0,½], B2 ∈ [½,1] on B; mL, mH | 2 | 5 | 12 |
| MSS0 | B1, B2, L1 ∈ [0,½], L2 ∈ [½,1]; mH | 4 | 4 | 12 |
| MSS1 | O, B2, L2; mH | 2 | 5 | 12 |
| SSS0 | B1, B2, L1, L2, H1 (x ∈ [0,½]), H2 (x ∈ [½,1]) | 6 | 3 | 12 |
| SSS1 | O, B2, L2, H1, H2 | 4 | 4 | 12 |
| SSS2 | O, X, L2, H1 | 2 | 5 | 12 |
| SSS3 | O, X, Y | 0 | 6 | 12 |

(For n = 8 the same supports apply with one free point fewer.) The free points range over Ω. Dropping "free points
are not on the sides" is a relaxation, and so are the closed half-ranges. Coincident contacts are exactly the vertex
cases. The "two contacts are one point" case is the midpoint case M.

**S3 symmetry breaking** (WLOG rows; each uses only that the pattern's stabilizer maps the family into itself):
- **SSS0.** The six "vertex distances" B1.x, 1−B2.x, L1.y, 1−L2.y, H1.x, 1−H2.x are permuted transitively by S3, since
  affine maps preserve ratios along sides. WLOG the minimum is B1.x: rows B1.x ≤ L1.y, B1.x ≤ H1.x,
  B1.x + B2.x ≤ 1, B1.x + L2.y ≤ 1, B1.x + H2.x ≤ 1.
- **MMS.** Use the reflection O↔X: B1.x + B2.x ≤ 1.
- **MSS0.** Use the mirror x↔y: B1.x ≤ L1.y.
- **MSS1 and SSS1.** Use the mirror: L2.y ≤ B2.x.
- **SSS2.** Use the reflection O↔X, which swaps L and H: H1.x + L2.y ≤ 1.

**Gap lemma (triangle form).** Take two points on one side with parameter distance d, say on side B, with the other
points w.
1. Each other point satisfies area = d·y_w/2 > T. (On H: area = d(1 − x_w − y_w)/2, the same form.)
2. So the other n − 2 points lie in {y > 2T/d} ∩ Ω, a homothetic copy of Ω with ratio 1 − 2T/d.
3. Hence T < Δ_{n−2}(1 − 2T/d)², which gives d > 2T/(1 − √(T/Δ_{n−2})).
4. The code uses a rational lower bound, with the square root rounded down.

The bounds used:
- **n = 9:** Δ7 = 7/144 (Zeng–Chen; re-proved by our exp 113 in exact mode). This gives l0 = 0.22090 at T = t*(1 + 10⁻³) (eps mode) and
  l0 ≈ 0.220351 at T = t*(1 − 10⁻⁹) (exact mode).
- **n = 8:** Δ6 ≤ (1/16)(1 + 10⁻³) (our exp 113 eps run; the literature value is 1/16). This gives l0 = 0.2574 at T = t8(1 + 10⁻³).

The gap lemma is imposed on S sides without a vertex point. With a vertex point the other contact is ≥ ½ away anyway.

**Three vertices (SSS3).** A triangulation bound gives 13 triangles of total area ½, so Φ ≤ 1/26 = 0.0385. That is
above t* = 0.0274, so it does **not** exclude the case. SSS3 is searched directly instead; it is cheap.

## 2. Port of the cell graph (`cgt.py`)

**Point domains.**
- Kind 0 is a rectangle ∩ {x + y ≤ Q}. This covers free points, fixed points and points on the legs.
- Kind 1 is a hypotenuse segment, parameter x, with y = Q − x.
- D is affine in each point, so its exact range over a product of convex polygons is attained at products of vertices.
  `build_verts` produces the exact integer vertices: the clip of a rectangle by x + y = Q has integer vertices, and
  there are ≤ 5 per domain.
- `propagate` handles three kinds of rows:
  - ORD rows: coord + g ≤ coord;
  - SUM rows: coord + coord ≤ Q;
  - x + y ≤ Q (kind 0), and y = Q − x (kind 1).

**Cells.** Cells are squares of side h = 1/G with cx + cy ≤ G − 1. These cover Ω, since cells with cx + cy ≥ G meet Ω
in at most a point that also lies in a kept cell. Cells are clipped to Ω through the kind-0 vertex rule.

**One free point per cell (re-derived for Ω).** Let p1, p2 be in one cell and p3 ∈ Ω.
- Then |D| = |(p2 − p1) × (p3 − p1)| ≤ h(|Δx| + |Δy|) ≤ 2h, because p1, p3 ∈ Ω ⊂ [0,1]².
- So 2h ≤ ρ = 2T is needed, i.e. G ≥ 1/T.
- This gives G = 37 for n = 9 (1/T = 36.5) and G = 30 for n = 8 (1/T = 29.5). It is asserted in `run.setup`.

**Witnesses.** Cell removal, edge removal and 3-cell pruning work as in exp 126. They use exact vertex ranges, with
supports of kind 0 or 1.

**P2 (positive combination).** Each coordinate is written as (C + W s_var)/2, where s_var ∈ [−1,1]. A hypotenuse point
uses one variable for both coordinates (y = (2Q − C_x − W_x s)/2).
- The bilinear terms x_a·y_b (a ≠ b, so distinct variables) are bounded by |W_a W_b|.
- The LP proposes λ. The final inequality is checked in Python integers, as in exp 126.
- The box ignores the clipping x + y ≤ Q, which is a relaxation.

**Cover (exact mode).**
- The cover is the balls of the first-order local certificate (`cert/`), of radius R = ρ_b(1 − 10⁻⁶) − (centre error).
- They are tested against all 6 S3 images. For each image, the labelling is *proposed* by a float nearest-centre
  assignment (it must be injective) and then verified exactly. Not all injections are enumerated, so a missed labelling
  can only cost time, never soundness.
- Every vertex of every point domain is mapped back by g⁻¹ in exact integers/Fractions and must lie within R of the
  matched certificate point. By convexity, the box then maps into the ball **in the certificate's own frame**, so no
  operator-norm factor (2 for four of the maps) is lost or missed.

**Parallelism.**
- Families with parameters are presplit into up to 2^depth outer boxes. Boxes emptied by propagation are dropped.
- The 0-parameter families (MMM, SSS3) are split by the smallest clique vertex, with no clique cap.
- Caveat: fixed-support tasks do not check the wall-clock deadline inside clique traversal. Only parameter-family outer
  trees stop at `t_end`.

## 3. Local certificate, n = 9 (`cert/`)
`python cert/certify_tri.py --sizes 9 --out local_certs_tri_n9.json` uses the exp 113 certifier unchanged; only the
paths of the copy differ. The input is `105-shape-universality/results/pool_tri_n9.json[0]` (site value 0.0548469).

**Result: certified, first order.**
- Core: 9 points, no rattlers. There are 13 loaded ties and 6 loaded wall contacts, i.e. 19 = 2·9 + 1 independent
  constraints with positive multipliers.
- The detected symmetry is C3. The reduced system has 5 unknowns and rank 5, so **d = 0** leftover directions.
- The centre is **exact rational**: (7/8,0), (0,1/8), (1/8,7/8), (0,25/28), (3/28,0), (25/28,3/28), (3/7,11/28),
  (11/28,5/28), (5/28,3/7).
- **t\* = 43/1568** exactly (normalized 43/784 = 0.0548469).
- Sup-norm basin radius ρ_b = 7.5691e-5 (exact rational in `cert/local_certs_tri_n9.json`). The smallest y lower bound
  is 0.0069.

x\* lies in family SSS0, with no vertex or midpoint points: B {3/28, 7/8}, L {1/8, 25/28}, H x ∈ {1/8, 25/28}. Its
three images under the SSS0 symmetry rows are covered. The cover radius is R = ρ_b(1 − 10⁻⁶) − 2·10⁻⁴⁰.

## 4. Validation

**Tests.** None of these tests is part of the proof.
- `test_sound.py`:
  - vertex ranges vs brute force on 3000 random rect/clipped/segment triples: 0 mismatches;
  - propagate on 4 families × 500 boxes: 0 sample points removed;
  - P2 on 796 random 9-point boxes with a hypotenuse point: 54 certificates, 0 violations;
  - cover on 240 boxes around the 6 images of x\*: 182 covered, 0 sampled points farther than R after exact
    back-mapping with the cover's own (g, labelling);
  - P2 cell exclusion on 295 random SSS1 support boxes + cell: 280 excluded, 0 violations.
- `test_neg.py` (T = t(1 − 10⁻⁶), no cover; boxes of half-width 2⁻⁴ … 2⁻¹⁵ around the family images of the optimum
  that satisfy the symmetry rows):
  - n = 8 SSS1: 36 boxes, 0 wrongly closed (E/T/P2); 36 protected cells, 0 wrongly excluded by cell P2;
  - n = 9 SSS0: 108 boxes, 0 wrongly closed; 108 protected cells, 0 wrongly excluded.
  - A first version of the test wrongly reported closures. Those came from images that violate the symmetry rows; such
    images are now filtered exactly.
- **Family-level negative control:** n = 8 SSS1 in `neg` mode (T = t8(1 − 10⁻⁶), no cover) does **not** close (13
  unresolved). One unresolved box contains x\* (width 2⁻⁸ … 2⁻⁹ in the support parameters). The others are boxes
  around it that were cut off by the 15-minute deadline.

**Triangle n = 8 (known value t8, proved in exp 113 by B&B in 9.6 h × 14 = 134 core-h).** G = 30, Q = 30·2¹⁴. These
runs were without cell P2, except the extra SSS1 eps comparison.

| family | eps (t8(1+10⁻³)) local nodes / core-h | exact (t8(1−10⁻⁹), cover) local nodes / core-h / cover |
|---|---|---|
| MMM | 298 / 0.004 | 298 / 0.004 / 0 |
| MMS | 7.8 k / 0.006 | 7.8 k / 0.008 / 0 |
| MSS0 | 0.54 M / 0.087 | 0.53 M / 0.062 / 0 |
| MSS1 | 25 k / 0.008 | 25 k / 0.010 / 0 |
| SSS0 | 1.61 M (+0.8 M fail) / 0.60 | 1.62 M (+0.8 M) / 0.86 / 0 |
| SSS1 | 1.23 M / 0.13 (cell P2: 0.89 M / 0.096) | 1.21 M (+0.2 M) / 0.24 / **1** |
| SSS2 | 31 k / 0.008 | 31 k / 0.010 / 0 |
| SSS3 | 988 / 0.011 | 992 / 0.006 / 0 |
| **total** | 3.4 M / 0.85 core-h | 3.4 M / **1.2 core-h**, all 8 closed |

The exact-mode n = 8 run reproduces exp 113 (value t8 = septic root, maximizer affinely equivalent to x8\*) at about
1/100 of the CPU. Wall time is dominated by one serial SSS0 root (25–40 min), and the
total is ≈ 1 h of 14 workers.

## Result (2026-11-15): triangle n = 9 value PROVED (computer-assisted, not yet independently reviewed)
Exact mode, T = (43/1568)(1 − 1e-9) rounded (T = 0.027423469360332), G = 37, run with batch.sh (cell_p2 = 0, so the P2 cell
exclusion was NOT used in exact mode; eps mode used cell_p2 = 8):
- all 8 families report "PROVED: every config with Phi > T is in a basin ball" and 0 unresolved;
- covers were needed only in SSS0 (18 cover closures logged); SSS0 alone took 1 677 outer leaves, 27.9 M cliques, 138 M
  local nodes and 4.0 h wall;
- eps mode (T = t*(1 + 1e-3)): all 8 families closed. Validation: triangle n = 8 exact, all 8 families closed; the
  negative control (n = 8 SSS1, T 1e-6 below the optimum, no cover) left 13 unresolved, as it must.
Statement: max over 9 points in a triangle of (min triangle area)/(triangle area) = 43/784, attained by the known
configuration; every maximizer lies in the certified basin of an S3 image of it. Pending the minimum-enclosing-triangle
uniqueness check, the classification is up to affine maps of |det| = 1 (see the exp 126 review).
Caveats: AI-assisted; the results section has not been reviewed externally; no replayable per-leaf certificates.
