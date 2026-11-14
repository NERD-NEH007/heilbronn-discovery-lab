# 113 / cert: local-optimality certificates in the triangle (2026-11-13)

**Result.** The rank-0 triangle optima of exp 105 are certified strict local maxima for n = 6, 7 and 8, and their
values are identified exactly.
- n = 7 and n = 8 are certified at first order, with a proven basin radius.
- n = 6 is certified at second order, with no radius.
- n = 7: t\* = 7/144 exactly (site value 7/72).
- n = 8: t\* is **exactly** the real root of the Chen–Zeng–Zhou septic (site value 2t\*).

Frame: Ω = {x ≥ 0, y ≥ 0, x + y ≤ 1}, with area 1/2. t = min |area| (not normalised); the site's normalised value is 2t.

Files:
- `certify_tri.py`: the certifier, copied from `44-interval-certificate/certify_interval.py` and modified as listed below.
  `python certify_tri.py` writes `local_certs_tri.json` and `run.log`.
- `identify_n8.py`: the exact identification of t\* at n = 8 with the septic. It writes `identify_n8.json` and
  `identify_n8.log`, and adds `t_exact` to the n = 8 entry of `local_certs_tri.json`. Run it **after** certify_tri.py.
- `local_certs_tri.json`: same layout and fields as `103-global-proof/results/local_certs.json`, keyed by n, each a list with one
  entry. Additional fields: `t_exact`, `frame`, and `extra`. `extra` holds the exact radii as rational strings, the exact tie
  list, the exact wall contacts and diagnostics.

Inputs: `105-shape-universality/results/pool_tri_n{6,7,8}.json[0]`. These are the same configurations as rank 0 of
`best_tri.jsonl`. Nothing outside this directory was modified.

## What changed vs exp 44, and why each step stays exact

1. **Walls.** In exp 44 a wall was "coordinate = 0 or 1". Here a wall is an affine function g_w(p) = a_w·p + b_w ≥ 0:
   - L: a = (1, 0), b = 0
   - B: a = (0, 1), b = 0
   - H: a = (−1, −1), b = 1

   A contact (j, w) means g_w(p_j) = 0, and its stress row is [a_w at the columns of point j, 0]. The H row has two nonzero
   entries; its coefficients are integers.
   - **Uniform-decrease lemma (paper3 B.5).** It only needs s_w = a_w·d_j ≥ 0 for a feasible displacement. This holds because
     g_w(x\* + d) = a_w·d_j (g_w is affine and vanishes at x\*).
   - **Quadratic remainder.** The bound 4δ² is unchanged.
   - **Cap t̃/10.** The bound ‖∇A_T‖₁ ≤ 3 still holds, since Ω ⊂ [0,1]².
   - **‖M(x\*) − M(x̃)‖∞ ≤ 6ρ.** Unchanged, because wall rows are constant.
2. **Symmetries.** The 8 square isometries are replaced by the 6 affine symmetries of Ω, g(p) = Mp + c. These are the vertex
   permutations of (0,0), (1,0), (0,1), and M and c are integer matrices. They are **not isometries**, but |det M| = 1. That is all
   the exact steps use: |A_{g(T)}(g x)| = |A_T(x)| identically, which gives orbit closure in `close_exact` and one equation per
   tie orbit. Each g maps walls to walls exactly: g_w ∘ g⁻¹ = g_{w'}.
3. **Parametrisation (`build_param`).** The rows are written as k·p = h in general form:
   - stabiliser rows: (M − I)p = −c;
   - wall rows on orbit images: (a_w M)·p_i = −b_w − a_w·c.

   Everything is exact rational, and exact equivariance is re-checked.
4. **Inside Ω.** Exp 44 checked "free coordinate in (ρ, 1 − ρ)". Here, for every point and wall, g_w(p_j(u)) is affine in u. Either
   it is identically 0 (an exact contact), or g_w(p̃_j) > ‖a_w‖₁ρ is checked exactly. ρ bounds every coordinate's movement.
5. **Non-tied triangles.** All C(n,3) triangles are now checked **exactly**: |A(x̃)| − 4ρ − 4ρ² > t̃ + r. Exp 44 used a float
   shortcut for margins above 1e-6; that shortcut is removed. Rattler nudging is not ported, and there are no rattlers at
   n = 6, 7, 8.
6. **Second-order wall multipliers.** At each contact point j, Σ_w μ_w a_w = −Σ_T y_T σ_T ∇_j A_T is solved exactly (1 or 2 walls).
   The rigorous componentwise error is propagated by ‖A_j⁻¹‖∞.
   - **Lift from the reduced to the full KKT system, for non-orthogonal G.** The covector ω = Σ y_T σ_T ∇A_T is G-invariant.
     The reduced stationarity says ω vanishes on S = V^G ∩ {contacts fixed}. Write ω|V^G = Σ c_n n|V^G and average c over G (the
     normals permute exactly). Then ω − Σ c̄_n n is invariant and vanishes on V^G, so it is 0, because (V\*)^G ≅ (V^G)\* for a
     finite group.
   - This is a short proof, **not machine-checked** (as in exp 44). The full stationarity residual at the centre is computed in
     exact arithmetic and is 0.
7. **Exact centres.** If the polished point rounds (limit_denominator 1e6) to rationals with F = 0 exactly, that point is used as
   the centre. Then Y = 0, and the unique zero in the Krawczyk ball is the centre itself, so `t_exact` is recorded. This happens
   at n = 7. At n = 6 the dyadic polish is already exact.
8. **Large uniqueness ball.** `contraction` also tests the radius 1e-6 (`uniqueness_radius_1e6`), which passes at all n. It is used
   by identify_n8.
9. **Output rounding.** In the JSON, `basin_radius` and `y_min_lower` are doubles rounded **down** from the exact rationals, and
   `existence_radius`, `box_rho` and `Minv_norm_upper` are rounded **up**. The exact rationals are in `extra`.

Unchanged from exp 44: the float stage (ties 1e-6, contacts 1e-8, relative-interior stress LP), the Newton polish on the 2^-110
grid, the Krawczyk test, closure by the 4-point identity and polynomial identities, the vertex stress, the β/C bound, and the
SOSC computation.

## Results

| n | sym | loaded ties + walls vs 2n+1 | mode | existence r | basin radius ρ_b (sup norm) | t enclosure | exact t\* (site 2t\*) |
|---|---|---|---|---|---|---|---|
| 6 | D3 (all 6) | 6 + 6 vs 13 | second-order (SOSC q = −2) | 1e-40 | none | 1/16 ± 1e-40 | 1/16 (1/8) |
| 7 | C3 | 9 + 6 = 15 | first-order | 1e-40 | 3.1404217136961446e-4 | 7/144 ± 1e-40 | 7/144 (7/72) |
| 8 | C1 | 11 + 6 = 17 | first-order | 6.99e-32 | 2.77542779402078e-4 | [3389460662709860617940540248694e-32, …248709e-32] | septic root (0.06778921325419721235881…) |

Further details:
- y_min lower bounds: 0.0625 (n = 6), 0.0238 (n = 7), 0.0246 (n = 8).
- Smallest margin of a non-tied triangle: 0.031 (n = 6), 0.024 (n = 7), 0.0020 (n = 8).
- n = 6 points: (1/4, 0), (0, 3/4), (3/4, 1/4), (0, 1/4), (1/4, 3/4), (3/4, 0).
- n = 7 points: (1/4, 3/4), (3/4, 0), (0, 1/4), (1/6, 0), (0, 5/6), (5/6, 1/6), (1/3, 1/3).

**n = 8 septic** (from math.tejstead.com/heilbronn/triangle/8/; the page states the polynomial is for the unit right triangle,
with A twice its root, which matches t here):

    p(t) = 1216512 t^7 + 1022976 t^6 − 2745456 t^5 + 2218730 t^4 + 32206304 t^3 − 19148615 t^2 + 3156103 t − 86233

identify_n8.py proves, in exact arithmetic (Fractions and sympy QQ polynomials):
1. **The septic.** p is irreducible over Q and has exactly 1 real root. p changes sign on the certified enclosure, and a Sturm
   count gives exactly 1 root in it.
2. **The coordinates.** Each of the 10 parameters (and t) is c_l(θ) ∈ Q(θ), a polynomial of degree ≤ 6 with denominators around
   6e30. These were *proposed* by PSLQ on a 1e-400 exact Newton polish. Every reduced tie equation σ_k A_{T_k}(p(c(θ))) − θ is 0
   mod p, by exact polynomial remainder.
3. **Identification with x\*.** The resulting point lies within 6.4e-30 of the certificate centre. This is a rigorous bound over
   the enclosure of θ\*, and it lies inside the 1e-6 ball where the zero of F is unique. Hence it **is** x\*, and t\* = θ\*
   **exactly**.

This proves the Chen–Zeng–Zhou value for this local optimum. It is not a proof of global optimality.

Caution: a first PSLQ run at 210 digits returned spurious relations, and the exact mod-p check rejected them. At 420 digits the
relations are genuine. The exact check is what makes the identification valid; PSLQ only proposes.

## GAPs and caveats

- **Cover test with non-isometric symmetries (important for the B&B port).** ρ_b is a sup-norm radius around x\* itself. For an
  image g(x̃), the requirement is ‖M_g⁻¹‖∞ · R + ρ < ρ_b (instead of R + ρ < ρ_b), where ‖M_g⁻¹‖∞ is the norm of the inverse map's
  linear part. ‖M_g⁻¹‖∞ = 1 for id and the mirror (x, y) ↦ (y, x). It is 2 for rot, rot2, mirror_X and mirror_Y. Alternatively,
  use R ≤ (ρ_b − ρ)/2 for all images.
- **n = 6 is second order: no basin radius.** It is therefore not directly usable by a B&B cover. It is an isolated strict local
  max (SOSC), not a family.
- **The n = 6 lift** from the symmetric reduced KKT system to the full one is a short argument (change 6 above), not
  machine-verified.
- **Trust base:** this script (~550 lines, not formally verified), Python Fractions and big ints, numpy object-int matmul, and, for
  the n = 8 identification only, sympy's exact QQ polynomial routines (`factor_list`, `count_roots`/Sturm, `rem`). mpmath is used
  only to propose relations.
- **Sanity check, not proof.** Random feasible perturbations (scratch script) never reach t\* at radii 1e-6, 1e-4 and ρ_b/2.
  Few samples were feasible, because wall points rarely stay inside Ω under a box perturbation.
- **Not global optimality**, at any n.
