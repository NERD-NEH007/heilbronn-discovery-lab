# Exp 112 — rattler repair of the 2n−7 counterexamples (Sudermann-Merx's reply, 2026-11-13)

His point: at record 22 the two free points (rattlers) can be pushed until each touches critical
triangles; that "repairs" Conjecture 1 (read as: *some* optimal configuration has >= 2n−7).

Method (repair.py): for each rattler-like point (<= 1 tie), others fixed, enumerate all vertices of
its feasible region (pairwise intersections of the lines |area(p,a,b)| = t and the walls), keep the
feasible one with most ties; repeat. Min area unchanged by construction. exact_check.py re-places
the moved points exactly (rational intersection of two tie lines) and verifies the exact minimum is
unchanged.

| n | ties before | after | 2n−7 | moves |
|---|---|---|---|---|
| 21 (record) | 37 | 37 | 35 | none needed |
| 22 (record) | 34 | **38** | 37 | 2 rattlers x 2 ties; exact check: min unchanged, 4 new exact ties |
| 29 (record) | 49 exact | 51 | 51 | 1 rattler |
| 35 (record) | 62 exact | 63 | 63 | 1 rattler (+1) |
| 36 (record) | 72 | 76 | 65 | — |
| 39 (ours) | 69 | 71 | 71 | 1 rattler |
| 40 (ours) | 64 | **72** | 73 | 4 rattlers x 2; whole-square search (RAD=2) same |

So no record is a counterexample under the existential reading. The note's headline claim fails.

Mechanism for what survives: generically a rattler at a vertex touches 2 constraints, so the repaired
count is at most (tight) 2·core + 1 − w − d + 2·r = 2n + 1 − w − d. It is < 2n − 7 iff w + d > 8,
i.e. 8 loaded walls and d >= 1 (a second-order maximum). Record 22 has w = 6, so it repairs; our
40, 81, 89 have w = 8, d = 1, so they don't (81/89 runs pending). None of these is a record or known optimal.

Full sweep of records 19–36 (published float coordinates, ties within rel 1e-9): only 22 (34), 29 (49), 35 (62)
are below 2n − 7; all three are repaired above. 5–18 satisfy the bound as stored (exp 102, Sudermann-Merx Table 10).
So every record 5 ≤ n ≤ 36 satisfies the existential reading of Conjecture 1.
