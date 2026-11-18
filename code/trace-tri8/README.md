# Certificate trace: Heilbronn's problem for eight points in a triangle

This archive holds the complete certificate trace for the computer-assisted proof of the optimal value for **eight points in a triangle**, with the separate checker that replays it.
- **The value:** t* is the common raw area of the 11 exactly tied triangles of the optimal configuration P*. Its normalized value is 2t* ≈ 0.0677892. t* is the unique real root of 1216512 t^7 + 1022976 t^6 − 2745456 t^5 + 2218730 t^4 + 32206304 t^3 − 19148615 t^2 + 3156103 t − 86233.
- **The domain:** x ≥ 0, y ≥ 0, x + y ≤ 1.

## Contents

| File | Contents |
|---|---|
| `tri8-checker.zip` | the checker sources, the local certificate (exp 113) and the release receipt |
| `tri8-trace-<FAMILY>.tar.gz` (8 files) | the manifest and every trace shard of one contact family |
| `SHA256SUMS-files.txt` | SHA-256 of every file inside the archives |
| `SHA256SUMS-archives.txt` | SHA-256 of the archives themselves |

| Family | Shards | Closures | Covers |
|---|---:|---:|---:|
| MMM | 1 | 192 | 0 |
| MMS | 15 | 53,196 | 0 |
| MSS0 | 203 | 527,976 | 0 |
| MSS1 | 16 | 49,132 | 0 |
| SSS0 | 2,471 | 2,481,837 | 0 |
| SSS1 | 227 | 1,164,661 | 2 |
| SSS2 | 16 | 102,009 | 0 |
| SSS3 | 1 | 627 | 0 |
| **Total** | **2,950** | **4,379,630** | **2** |

## Reproduce the release receipt

Requirements: Python 3.12 or later (standard library only) and about 1 GB of free disk space. With 15 workers the replay takes about 3 minutes.

1. Extract every archive into one empty directory:
   ```
   unzip tri8-checker.zip -d run
   for f in tri8-trace-*.tar.gz; do tar -xzf "$f" -C run; done
   ```
2. Optionally, check the files against the inner checksums (run from inside `run/`):
   ```
   sha256sum -c ../SHA256SUMS-files.txt
   ```
3. Run the strict eight-family replay:
   ```
   cd run
   python -B experiments/202-tri8-trace/verify_release8.py --workers 4 > receipt.json
   ```
   - The output must contain `"complete": true`.
   - The family counts must match `experiments/202-tri8-trace/release-receipt.json`.
   - Run it without `-O`, because the dependency assertions must stay active.
   - `python -B experiments/202-tri8-trace/test_mutations8.py` runs the mutation tests, which check that the replay rejects six kinds of corrupted traces.

## What the checker does

- **Pins the statement:** the eight canonical support roots, Q = 30 · 2^14, the raw cutoff t*(1 − 10^-9) as a rational, the certificate hash and the cover radius.
- **Checks the optimum in exact rational arithmetic** (`basin8.py`):
  - **existence:** a Newton–Kantorovich argument shows that the 11 ties and 6 wall contacts have an exact solution P* within 7 · 10^-32 of the certificate's rational centre;
  - **the value:** at P*, every other triangle is strictly larger;
  - **the basin:** a first-order basin shows that within the cover radius (about 2.78 · 10^-4) of P*, no configuration exceeds t*.
- **Replays every family:**
  - walks the split tree and checks every split and side-gap contraction;
  - checks the SHA-256 of every shard;
  - re-verifies every closure witness in rational arithmetic;
  - requires every shard to close.
- **Never imports the search engine.**

## Trust base

- the checker sources in `tri8-checker.zip`;
- the hand-proved reductions: the minimum-enclosing-triangle normalization and contact-family completeness (shared with the n = 9 proof) and the side-gap contraction. The contraction uses the proven raw bound 1/16 for six points (Yang, Zhang and Zeng).

The search code that produced the trace is not needed to verify it.
