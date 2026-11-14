# Heilbronn discovery lab

Papers, data and verification code on the Heilbronn triangle problem in the unit square and the triangle
(place n points in the region to maximise the minimum triangle area t). Author of all four papers: Ethan Hendren.

1. **A computer-assisted proof that the known nine-point Heilbronn configuration in the unit square is optimal.**
   (`papers/heilbronn_n9.pdf`; code `code/exp110/`, run logs `certificates/exp110/`.)
   Result: H(9) = (9√65 − 55)/320, attained only by the known configuration up to the symmetries of the square.
   **The code was written with AI assistance (Claude, Anthropic), and the proof has not yet been independently
   checked. Independent verification is welcome.**
2. **A computer-assisted proof of optimality for Heilbronn's problem with eight points in a triangle.**
   (`papers/heilbronn_tri8.pdf`; code `code/exp113/`, run logs `certificates/exp113/`.)
   Result: in the triangle {x, y ≥ 0, x + y ≤ 1}, 8 points have a triangle of area at most θ, the unique real root of
   the septic conjectured by Chen, Zeng and Zhou (normalised value 2θ ≈ 0.0677892), attained only by the known
   configuration up to the 6 symmetries. **AI-assisted, not yet independently checked; verification welcome.**
3. The structure of best-known Heilbronn configurations in the unit square: certificates, geometric families and a
   crossover at n = 43. arXiv: TBD. (`papers/heilbronn_structure.pdf`)
4. Critical triangles in best-known Heilbronn configurations: rattlers, walls and flat directions.
   arXiv: TBD. (`papers/critical_triangles_note.pdf`)

In the table below, P1 and P2 are papers 3 and 4 of this list (the structure paper and the critical-triangles note)
and P3 is the n = 9 proof. The triangle proof (exp 113) is checked by `code/smoke_test.py` (run accounting and the exact septic identification).

Quick check (about 10 seconds; must print `ALL PASSED`):

```
pip install -r requirements.txt
python code/smoke_test.py
```

Python 3.11 or later. Tested with Python 3.15 and the versions pinned in `requirements.txt` on Windows. Run every
script from the repository root. Scripts write to `output/`, which git ignores.

## Contents

| path | what it is | papers |
|---|---|---|
| `configurations/best/n{37..100}.json` | our best configurations. Each file holds the coordinates as exact decimal expansions of the stored IEEE doubles, the stored value, the exact minimum area as a fraction `p/q` (with 40 decimals), n²t and tie counts (exactly equal, within 1e-9 and within 1e-6, relative) | P1 §4 (value tables 37–64, 65–100), P2 App. A |
| `configurations/n97_polished.json` | n = 97 re-polished (experiment 108). Its value is above the stored best, but it is **not certified**: it is a KKT point with d = 2 | P1 §4, App. C |
| `configurations/polished/n{37..100}.json` | experiment 44 polished centres, as exact rationals about 1e-31 from the certified exact local maximum. `points_30` holds these truncated toward zero to 30 decimals, with the exact minimum area of the truncated coordinates | P1 §2.3, Data availability |
| `configurations/record_values.json` | published record values for 19 ≤ n ≤ 36: the exact minimum of the published coordinates. Values only | P1 App. A, App. C |
| `configurations/records_printed/` | only what the papers print about records: the n = 22 coordinates (P2 Appendix "Coordinates") and the rattler nudges of records 21, 35 and 36 (moved points only) | P2 §4, §5, App. B |
| `configurations/small_n/optima_small.json` | n = 5–18, rebuilt by our own search without the site's coordinates | P1 §7.1, P2 §7.1 |
| `certificates/exp44/` | certificate runs of experiment 44 (`certs*.jsonl`, one line per size and run). `certs_report.jsonl` is the full run on the current library. `certs97_exp108.jsonl` is the failed n = 97 attempt | P1 §2.3, App. C |
| `certificates/exp78/` | the independent full-KKT check of experiment 78: first run, re-check of 2026-11-12, current bests (`kkt78_current.jsonl`), polished centres at the second-order sizes (`kkt78_polished.jsonl`), n = 97, the comparison with exp 44, the conjecture linter and the nudge LP | P1 §2.3, P2 §4–6 |
| `certificates/exp103/` | branch-and-bound re-proofs for n = 7 and 8: local certificates, run statistics and progress logs, the n = 9 probe | P1 §7.2 |
| `certificates/exp104/exact_forms.json` | exact values: minimal polynomials, closed forms, Galois data, and t to 300 digits. Coordinates are omitted for n ≥ 19 (published records) | P1 §7.3 (table of exact values) |
| `certificates/appendix_data.json` | the critical-triangle counts behind paper 2. Record coordinates are omitted; ours (n ≥ 39) are included | P2 §5 (records table), App. A |
| `certificates/exp109/loo.json` | leave-one-family-out control for the n = 43 break | P1 §6.2 |
| `papers/` | the three papers (PDF and LaTeX source; `figures/` and `repro_table.tex` are inputs of the structure paper) | P1, P2, P3 |
| `certificates/exp110/` | the n = 9 run (`results/run_n9/`: `stats.json`, `progress.log`, `fails.jsonl` = empty, i.e. zero unprunable boxes), the n = 8 run, size probes and validation runs, and `EXP110_NOTES.md` | P3 |
| `certificates/exp112/` | rattler-repair runs (`repair.py`, `exact_check.py`). n = 39, 40, 40_global, 81 in full. For records 21, 22, 29, 35, 36 only summary fields and the moved points (record coordinates are not redistributed). `EXP112_NOTES.md` | P2 §4 |
| `code/` | verification scripts (see below) | |
| `records/` | empty. Put record files you download from the record site here (see `records/README.md`) | |

## Code

| script | what it does |
|---|---|
| `code/smoke_test.py` | quick end-to-end check (exact areas, one exp 44 and one exp 78 check, exact form n = 20, record 22 count, n = 9 closed form, exp 110 import) |
| `code/exact_min_area.py` | exact (integer and rational) minimum area and tie counts. Re-checks every `configurations/best` file, or any file given with `--file` |
| `code/verify_record_file.py` | verifies a record file **you downloaded yourself** from the record site: exact minimum, tie counts, and agreement with the value used in the papers |
| `code/exp44/certify_interval.py` | the certificate of P1 §2.3 (experiment 44): reduced tie system, radii-polynomial (Krawczyk-type) existence test, exact post-checks, first- and second-order modes. `python code/exp44/certify_interval.py --sizes 37-64` |
| `code/exp78/kkt_check.py` | the independent check (experiment 78): full KKT system and an exact Newton–Kantorovich test. `--sizes 37,40,61` |
| `code/exp78/kkt78_polished.py` | exp 78 on the exp 44 polished centres (second-order sizes 65, 76, 81, 89, 100) |
| `code/exp78/appendix.py`, `nudge.py`, `lint_conjecture.py` | critical-triangle counts, rattler nudges (LP), and the 2n−7 linter (P2) |
| `code/exp103/bb.py`, `run_par.py`, `probe.py`, `get_certs.py`, `test_sound.py` | global branch and bound for n = 7, 8 (P1 §7.2). Example: `python code/exp103/run_par.py 7 exact --workers 4 --tag n7`. This takes about 73 minutes on one core; n = 8 takes several hours |
| `code/exp104/exact_core.py`, `run.py` | exact values: high-precision Newton, LLL, exact check in Q[x]/(p), Krawczyk. Example: `python code/exp104/run.py --sizes 5` |
| `code/check_exact_forms.py` | checks that every stored minimal polynomial vanishes at the stored 300-digit value |
| `code/make_repro_table.py` | regenerates the reproducibility table of P1 Appendix C (`output/repro_table.tex`, identical to the paper's) |
| `code/make_paper_figures.py` | regenerates the four figures of P1 (`output/figures/`) from `code/figure_data/` |
| `code/exp110/` | faster branch and bound `bb2.py`, driver `run_par2.py`, probes, soundness tests, and `check_closed_form.py` (closed form of H(9)). Run commands in `code/exp110/README.md`: n = 7 about 10 min, n = 8 about 9 min, n = 9 about 16 h on 14 workers | P3 |
| `code/exp112/repair.py`, `exact_check.py` | rattler repair: single-point moves that keep t and add critical triangles, with an exact check. Writes to `output/exp112/` | P2 §4 |
| `code/hlib.py` | shared loaders and exact arithmetic |

Records 19–36 are used by several scripts: exp 44 and 78 at n ≤ 36, `appendix.py`, and `lint_conjecture.py` from
n = 20. They need the published coordinates in `records/`. Without them, those scripts skip n ≤ 36 or tell you which
file is missing. The figure script uses precomputed scalar observables (`code/figure_data/observables_records.json`)
when the records are absent, and recomputes them when the records are present. Both ways give identical output.

## What is rigorous and what is floating point

From paper 1, §2.3:

> In experiment 44, floating point only selects the active (loaded) sets, the symmetry group, the stress vertex and
> the preconditioners R. Existence and uniqueness of the exact zero (the Krawczyk test), the exact ties, the bound that
> every other triangle exceeds t on the ball, the positivity of the multipliers and the basin radius are computed in
> exact rational or interval arithmetic; in second-order mode so are the Krawczyk test of the KKT system, λ > 0,
> μ > 0 and the margin in the second-order sufficient condition. One step is not of this kind: at the symmetric
> second-order sizes 22 and 24 (both records) the KKT system is certified in the symmetry-reduced variables, and the
> lift of the reduced KKT point to a KKT point of the full system is a hand argument (invariance of the full Lagrangian
> gradient under the symmetry group), checked numerically (residual ≤ 2·10^-33). Our n = 40 is also certified in
> reduced variables (group of order 4) and rests on the same argument (full stationarity residual 2·10^-35).
> In experiment 78 every step is exact except the second-order eigenvalues, which are computed in floating point on
> an exactly computed tangent space (margins ≥ 2.4·10^-5, at n = 100) and are not interval-enclosed.
> Neither code is formally verified, and neither says anything about global optimality.

Also note:
- Every reported minimum area is recomputed in exact rational arithmetic on the stored coordinates (`exact_min_area.py`).
- Counts "within 1e-6" are relative tolerance counts. The exact count is the number within 1e-9 at the stored
  coordinates, which matches the exact ties at the certified optimum (P1 §2.1, P2 Definition "counting conventions").
- The certificates concern the polished point. The stored double-precision configuration differs from it by at most
  1e-11 at first-order sizes and by 1e-7 to 6e-6 at second-order sizes (P1 §2.3).
- Exact values (exp 104): "proved" means as defined in P1 §7.3. `check_exact_forms.py` only checks the stored
  polynomial against the stored digits.
- n = 97 is not certified (d = 2).

## Independent verification of the n = 9 proof

The proof has not yet been independently checked; checks are welcome. What can be checked, and how:

| what | how | time |
|---|---|---|
| the optimum is exactly C in Q(√65): 11 triangles of area exactly (9√65 − 55)/320, the other 73 strictly larger, C inside the certificate's existence box with the same ties | `python code/exp110/check_closed_form_exact.py` (all exact; nsimplify only proposes C) | 2 s |
| the saved run is complete and consistent: done, 0 failures, leaves − internal nodes = 36 roots, T ≤ t_lower(1 − 10⁻⁹) | `python code/exp110/verify_run.py certificates/exp110/results/run_n9 certificates/exp103/local_certs.json` | 1 s |
| P2 certificates, by an independently written checker (written from the paper's Appendix A, without the search code) | `python code/exp110/clean_room/check_p2.py certificates/exp110/p2dump/*.gz` (8 970 sampled n = 9 certificates; tests: `clean_room/test_check_p2.py`). Dump your own with `code/exp110/dump_p2.py`. In our runs it accepted all 292 120 P2 certificates of a full n = 7 run, 136 972 from n = 8 and the 8 970 from n = 9, and rejected none | 15 s |
| the search itself | re-run it: `code/exp110/README.md` (n = 7 and 8 in minutes, n = 9 about 16 h on 14 workers) | 16 h |
| the files are the ones used | `MANIFEST-n9.sha256` (`sha256sum -c MANIFEST-n9.sha256`) | 1 s |

Trusted components, and their status:
- **The search code** (`code/exp110/bb2.py`, `code/exp103/bb.py`): every discard is decided in exact integer arithmetic. The LP solver (HiGHS) only proposes multipliers. P2 has been cross-checked by the clean-room checker on samples (all of n = 7; a sample at n = 9); P0, P1 and the cover test have not been re-implemented independently. `verify_run.py` checks the run's recorded counters for consistency; it does not re-derive the boxes.
- **The local certificate** (`code/exp44/certify_interval.py`): this closes the neighbourhood of the optimum. The appendices of the n = 9 paper (`papers/heilbronn_n9.pdf`) derive the P2 certificate and the basin radius; an external review of that derivation is welcome.
- **Python, numpy, numba and the OS:** standard.

## Not included

- **Record coordinates 19 ≤ n ≤ 36.** These belong to the record site and its contributors and are not
  redistributed here. Download them from https://math.tejstead.com/heilbronn and check them with
  `code/verify_record_file.py`. The only exception is the n = 22 coordinates that paper 2 prints, which are included
  with attribution. For the same reason, exp 44 polished centres for n ≤ 36 and exact record coordinates from
  exp 104 are omitted. Paper 2 says the coordinates of record 29 are in the data repository. They are the published
  coordinates, so get them from the site.
- The search engine and the exploratory experiments behind the configurations are not included (P1 §2.2 describes them).
- Experiment 109's script depends on experiment-86 code and data that are not part of this repository. Only its
  output (`loo.json`, `loo.log`) is included.

## Attribution

- Records n ≤ 36: *The Heilbronn Problem*, record tables, exact verification and attributions, maintained by
  Tej Stead, https://math.tejstead.com/heilbronn, and the contributors credited there. The n = 22 record is due to
  N. Sudermann-Merx (October 2026).
- Exact value at n = 12: F. Comellas and J. L. A. Yebra, *New lower bounds for Heilbronn numbers*, Electron. J.
  Combin. 9 (2002), #R6.
- Exact realisation at n = 14: R. Chappell, https://github.com/rhyschappell/heilbronn-n14-exact. The configuration is
  due to M. Beyleveld (2006).
- The affine wall identity (every side carries total wall force t; opposite sides have equal first moments) and the
  existence of KKT multipliers are D. Trela's: *KKT stresses, affine moments, and separator flux in the Heilbronn
  triangle problem*, arXiv:2609.25035.
- The 2n−7 conjecture and the exact values for n ≤ 10: N. Sudermann-Merx, arXiv:2603.11107.

## Use of AI tools

The author used Claude, an AI system developed by Anthropic, for computation (writing and running search,
certification and analysis code) and for drafting text. All claims were checked against the computational records;
the author takes responsibility for the content. AI systems are not authors of this work.

## Citation and licence

Cite the papers above (see `CITATION.cff`). Code: MIT (`LICENSE`). Data (`configurations/`, `certificates/`,
`code/figure_data/`): CC BY 4.0 (`LICENSE-DATA`). The licence covers our own data only. Record data from the record
site stays under the terms of its source.
