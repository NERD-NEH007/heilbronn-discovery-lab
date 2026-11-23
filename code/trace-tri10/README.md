# Certificate trace: Heilbronn's problem for ten points in a triangle

**Zenodo:** this certificate trace, DOI [10.5281/zenodo.24426026](https://doi.org/10.5281/zenodo.24426026); the paper, DOI [10.5281/zenodo.24426068](https://doi.org/10.5281/zenodo.24426068). A SHA-256 commitment to the manifests and statements of this trace was deposited first, on 2026-11-22: DOI [10.5281/zenodo.24447262](https://doi.org/10.5281/zenodo.24447262).

This archive holds the complete certificate trace for the computer-assisted proof that the optimal normalized value for ten points in a triangle is **δ₁₀ = 0.0433767406780570017601…**. That is twice the raw optimum in the domain x ≥ 0, y ≥ 0, x + y ≤ 1. The optimum is attained by a unique configuration up to the triangle's symmetries and relabeling: an isolated critical configuration with 14 tied triangles and 6 wall contacts. The archive also holds the separate checker that replays the trace.

## Contents

| File | Contents |
|---|---|
| `tri10-checker.zip` | checker sources, the n = 10 and n = 8 local certificates, the archived KKT centre `kkt_centre_u0.json`, the n = 8 release receipt, the enclosure check, the release receipt |
| `tri10-trace-<FAMILY>.tar.gz` (8 files) | manifest, statement, pair-exclusion files and every trace shard of one contact family (the per-family results are in the release receipt) |
| `SHA256SUMS-files.txt` | SHA-256 of every file inside the archives |
| `SHA256SUMS-archives.txt` | SHA-256 of the archives themselves |

| Family | Shards | Support contractions | Pair-exclusion files | Closures | Covers |
|---|---:|---:|---:|---:|---:|
| MMM | 721 | 0 | 0 | 2,431,706 | 0 |
| MMS | 364 | 20 | 129 | 2,280,639 | 0 |
| MSS0 | 997 | 123 | 1,019 | 4,114,986 | 0 |
| MSS1 | 138 | 11 | 95 | 764,042 | 0 |
| SSS0 | 10,559 | 1,275 | 6,153 | 59,471,057 | 0 |
| SSS1 | 2,362 | 158 | 1,508 | 21,459,534 | 2,285 |
| SSS2 | 164 | 16 | 99 | 945,747 | 0 |
| SSS3 | 861 | 0 | 0 | 535,522 | 0 |
| **Total** | **16,166** | | | **92,003,233** | **2,285** |

## Reproduce the release receipt

**Requirements:**
- Python 3 with numpy (tested with Python 3.15 and NumPy 2.5). The exact cell-graph replay in `graph_replay10.py` uses numpy integer arithmetic; everything else is the standard library.
- About 6 GB of free disk space: about 2.6 GB for the extracted files, plus the downloaded archives.
- At most about 112 CPU-hours. Our clean-room replay took 7 h 58 min of wall time with 14 worker processes on an 8-core laptop (AMD Ryzen 9 4900HS, 16 GB RAM); with 4 workers, expect up to about a day (an estimate).

1. Extract every archive into one empty directory:
   ```
   unzip tri10-checker.zip -d run
   for f in tri10-trace-*.tar.gz; do tar -xzf "$f" -C run; done
   ```
2. Optionally, check the files against `SHA256SUMS-files.txt`. From inside `run/`:
   ```
   sha256sum -c ../SHA256SUMS-files.txt
   ```
3. Run the strict eight-family replay:
   ```
   cd run
   python -B experiments/232-tri10-release/verify_release10.py --archive archive --workers 4 > receipt.json
   ```
4. Check the receipt:
   - The output must contain `"complete": true` and `"conclusion": "optimal value and S3 uniqueness modulo labels"`.
   - Compare it with the shipped receipt as JSON, which ignores line endings:
     `python -c "import json,sys; a,b=(json.load(open(f)) for f in sys.argv[1:]); print(a==b)" receipt.json experiments/232-tri10-release/release-receipt.json`
   - The shipped receipt's SHA-256 is `48fd330a87556549d1b98d0e65419879a1357b7dd456225a5ef4dc2fb4072202` with LF line endings (as shipped), or `c95297bb568f01480cd5c87bb0e24dd67de1f5bd634c2045bbb0a8058b7872b4` with CRLF.
   - On Windows, run step 3 in cmd.exe or PowerShell 7: Windows PowerShell 5 writes UTF-16 when redirecting.
5. Things to know:
   - Run without `-O`: the dependency assertions must stay active.
   - Leave `REPLAY_MASK_CACHE` unset; the checker refuses to run with it set.
   - Only the full replay produces a theorem receipt (`"complete": true`). (`--preflight` exists only for the lab's working tree; it needs pipeline receipts that this package does not ship.)

## What the checker does

**Re-derives the statement from scratch:**
- the eight canonical support roots;
- Q = 770048 = 47 · 2¹⁴;
- the raw cutoff 108441851695034062548779343914599525513581 / 5·10⁴²;
- the side-gap rows, from the n = 8 theorem's certified global upper bound U_8. The checker pins the n = 8 release receipt by SHA-256, takes U_8 from it, and requires its own re-verified n = 8 basin bound to be at least U_8;
- the optimum's rational upper endpoint and the cover radius, from the n = 10 basin checker.

It requires every manifest's statement to equal the one it derived.

**Re-verifies the local pieces:**
- the n = 10 basin at the cover radius 1/200, by the wide second-order basin lemma (`246-basin-wider/basin10w.py`, on `214-basin10/basin10.py`'s KKT root; paper Appendix A). This covers existence of the KKT root, the value checks, and the second-order lemma.
- uniqueness of the KKT root in the sup-norm ball of radius 1/100000 about the ARCHIVED rational centre `kkt_centre_u0.json` (pinned by SHA-256), by an exact contraction bound, with existence of the root within that ball by Kantorovich about the same centre. The 41 centre coordinates are written into the receipt, with a flag saying whether this machine's own float recomputation reproduced them;
- the numbers stated in Theorem 1: the root ball radius is at most 1.6e-31 and the value enclosure lies strictly inside the stated decimal interval.
- the enclosure-uniqueness check, run over the root ball the basin check has just certified.

**Replays every family:**
- walks each manifest's split tree and re-derives every shard root;
- checks every split, support contraction and cell-pair exclusion;
- checks the SHA-256 of every shard and pair file;
- re-enumerates every clique itself;
- re-verifies every closure witness in rational arithmetic;
- requires every shard to close.

It never imports the search engine or the trace emitters; it asserts this at start-up.

## Trust base

**Checker sources the replay loads:**
- `232-tri10-release/verify_release10.py`;
- `216-trace10/verify_manifest10.py`, `stream_trace10.py`, `graph_replay10.py` and `support10.py`;
- `215-trace10-local/stream_local10.py`;
- `213-close-mcc/mcc_check.py`;
- `214-basin10/basin10.py`, `245-basin-sharp/basin10s.py` and `246-basin-wider/basin10w.py`;
- `202-tri8-trace/basin8.py`;
- `152-geometry-trace/verify_trace.py`;
- `140-triangle-n10-enclosure/verify.py`.

**The hand-proved mathematics described in the paper:**
- the normalization and contact-family completeness, including the canonical choice in SSS0;
- the eight-point side-gap lemma, which uses the n = 8 theorem;
- the cutoff-or-cover invariant (the soundness of each record type);
- the basin lemma (Appendix A), the Newton–Kantorovich existence and contraction-uniqueness arguments, and the enclosure-uniqueness case argument. The checker evaluates their inequalities exactly; their validity as mathematics is hand-proved.

**Windows / git note:** the pinned JSON files are LF. If you take the checker from the git repository instead of the Zenodo zip, disable CRLF conversion (`git config core.autocrlf false` before cloning), or the digest pins will refuse the files.

The search code that produced the trace is not needed to verify it and is not included.
