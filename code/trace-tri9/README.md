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

Requirements: Python 3.12 or later (standard library only), about 12 GB of free disk space, and a few CPU-hours (1 worker). With 15 workers, the replay takes about 30 minutes.

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
- **Never imports the search engine.**

## Trust base

- the checker sources in `tri9-checker.zip`;
- the hand-proved reductions described in the paper: the normalization and contact-family completeness, the exact n = 7 side-gap bound, and the orbit reduction behind canonical SSS0.

The search code that produced the trace is not needed to verify it.
