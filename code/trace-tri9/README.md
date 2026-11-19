# Certificate trace: Heilbronn's problem for nine points in a triangle

This archive holds the complete certificate trace for the computer-assisted proof that the optimal value for nine points in a triangle is **43/784** (normalized; raw 43/1568 in the domain x ≥ 0, y ≥ 0, x + y ≤ 1). It also holds the separate checker that replays the trace.

## Contents

| File | Contents |
|---|---|
| `tri9-checker.zip` | checker sources, the basin certificate, the enclosure check, the release receipt |
| `tri9-trace-<FAMILY>.tar.gz` (8 files) | the manifest, the per-family replay receipt and every trace shard of one contact family |
| `SHA256SUMS-files.txt` | SHA-256 of every file inside the archives |
| `SHA256SUMS-archives.txt` | SHA-256 of the archives themselves |

| Family | Shards | Closures | Covers |
|---|---:|---:|---:|
| MMM | 1 | 7,089 | 0 |
| MMS | 44 | 792,769 | 0 |
| MSS0 | 702 | 14,058,921 | 0 |
| MSS1 | 57 | 132,008 | 0 |
| SSS0 | 1,981 | 35,490,128 | 32 |
| SSS1 | 1,030 | 18,073,130 | 0 |
| SSS2 | 38 | 627,454 | 0 |
| SSS3 | 1 | 12,888 | 0 |
| **Total** | **3,854** | **69,194,387** | **32** |

SSS0 is replayed over canonical representatives of the S3 action on its six vertex–side incidences.

## Reproduce the release receipt

Requirements: Python 3.12 or later with numpy (the cell-graph replay in `graph_replay.py` uses numpy int64 arithmetic with a static overflow guard; everything else is the standard library), about 12 GB of free disk space, and a few CPU-hours (1 worker). With 8 workers on an 8-core Apple M3 the replay takes about 15 minutes; with 15 workers on our Windows machine, about 30 minutes.

1. Extract every archive into one empty directory:
   ```
   unzip tri9-checker.zip -d run
   for f in tri9-trace-*.tar.gz; do tar -xzf "$f" -C run; done
   ```
2. Optionally, check the files against `SHA256SUMS-files.txt` (run from inside `run/`):
   ```
   sha256sum -c ../SHA256SUMS-files.txt
   ```
3. Run the strict eight-family replay:
   ```
   cd run
   python -B experiments/197-tri9-release-v2/verify_release.py --workers 4 > receipt.json
   ```
   - The output must contain `"complete": true` and `"conclusion": "optimal value and S3 uniqueness modulo labels"`.
   - The family counts must match `experiments/197-tri9-release-v2/release-receipt.json`.
   - Compare the receipts as JSON, which ignores line endings:
     `python -c "import json,sys; a,b=(json.load(open(f)) for f in sys.argv[1:]); print(a==b)" receipt.json experiments/197-tri9-release-v2/release-receipt.json`
   - The shipped receipt uses LF line endings. Its SHA-256 is `f6090b7a461f18d1e7fb4646c3b260ffa490cfcee5381c20912b2f7ada29b9a6` (LF), or `08bbf836d30f92313ea0b2cf9a048069f1139071e067b50df7e8cc63ae6a7b14` for a receipt written with CRLF line endings (for example by redirecting the output in cmd.exe). Windows PowerShell 5 writes UTF-16 when redirecting, so on Windows use cmd.exe or PowerShell 7 for step 3.
   - Run it without `-O`, because the dependency assertions must stay active.
   - `--preflight` gives a fast check of every shard header and hash, without the replay.

## What the checker does

- **Pins the statement:** the eight canonical support roots, Q = 606208, the raw cutoff 42999999957/1568000000000 and the raw optimum 43/1568.
- **Re-verifies the local pieces:** both basin radii, with the exact basin checker, and the enclosure-uniqueness check.
- **Replays every family:**
  - walks each manifest's split tree and re-derives every shard root;
  - checks every split and contraction;
  - checks the SHA-256 of every shard;
  - re-verifies every closure witness in rational arithmetic;
  - requires every shard to close.
- **Never imports the search engine or the trace emitters.** The side-gap contraction for MMS, MSS0 and SSS1 is in the standalone module `experiments/197-tri9-release-v2/contract9.py`.

## Trust base

- the checker sources that the replay loads: `197-tri9-release-v2/verify_release.py` and `contract9.py`, `186-tri9-sss0-canonical-production/stream_trace.py`, `verify_manifest.py` and `canonical.py`, `160-tri9-sss3-production-trace/graph_replay.py`, `153-exact-tri9-basin/verify_basin.py`, `152-geometry-trace/verify_trace.py` and `136-tri9-enclosure-uniqueness/check.py`. The other sources in `tri9-checker.zip` are the search and emission code, included for reference; the replay does not import them;
- the hand-proved reductions described in the paper: the normalization and contact-family completeness, the exact n = 7 side-gap bound, and the orbit reduction behind canonical SSS0.

The search code that produced the trace is not needed to verify it.

## Changes in v1.1 (2026-11-19)

- The release receipt is shipped with LF line endings; SHA-256 values for both conventions are given above. Its content is unchanged.
- numpy is listed as a requirement.
- The checker takes the side-gap contraction from the standalone `contract9.py` (a verbatim copy of `contract` from `163-tri9-trace-queue/queue_traces.py`) instead of importing `queue_traces`, whose module-level imports loaded the emitter modules. The receipt it produces is identical.
- The trace files are unchanged.
