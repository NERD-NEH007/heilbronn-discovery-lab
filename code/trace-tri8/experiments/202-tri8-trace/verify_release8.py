"""Exp 202 (Claude, 2026-11-18): strict eight-family release receipt for triangle n = 8, modelled on the n = 9
release (exp 169/197).

What it pins and checks:
- the eight distinct canonical support roots, Q = 30 * 2^14, the raw cutoff (exp 127 n = 8 exact run), the
  certificate hash (exp 113), and the cover radius;
- the exact n = 8 basin (basin8.verify): Kantorovich existence of the tied optimum P* within rho of the rational
  centre, the value at P*, and the first-order basin;
- every family manifest replayed from scratch by verify_manifest8 (exp 186 copy, n = 8), with the side-gap
  contraction for MMS, MSS0, SSS1 and SSS0. Each family's bound must be the cutoff (no covers) or the optimum
  (covers), and at least one cover must close.
It imports no search code. Run without -O."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
from functools import partial
from hashlib import file_digest
import json
from pathlib import Path
import sys

if not __debug__:
    raise RuntimeError("Run without -O to preserve the dependency assertions")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stream_trace8, verify_manifest8 as manifests, basin8, support8   # noqa: E402
sys.path.insert(0, str(HERE.parent / "160-tri9-sss3-production-trace"))
from graph_replay import graph_boxes                                      # noqa: E402
from verify_trace import require, unique_object                           # noqa: E402

Q = 491520
T = "1694730329660199977615339815376729875653/50000000000000000000000000000000000000000"
BASIN_HASH = "172a3d221b9ea5fed2b591d9e971dd8d911ababf9f293521660c4dbe348f5dc2"
FAMILIES = ("MMM", "MMS", "MSS0", "MSS1", "SSS0", "SSS1", "SSS2", "SSS3")
CONTRACTED = ("MMS", "MSS0", "SSS1", "SSS0")
WORKERS = 1
ORIGINAL_REPLAY = manifests.replay_shard


def checked_replay(task):
    require(stream_trace8.graph_boxes is graph_boxes, "graph override missing in replay worker")
    return ORIGINAL_REPLAY(task)


def install_override():
    stream_trace8.graph_boxes = graph_boxes
    manifests.verify = stream_trace8.verify
    manifests.replay_shard = checked_replay


def pinned():
    require(Q == support8.Q and T == support8.CUTOFF, "support module drifted from the pinned statement")
    require(file_digest(basin8.SOURCE.open("rb"), "sha256").hexdigest() == BASIN_HASH, "certificate hash mismatch")
    points, ties, walls, t0 = basin8.load()
    radius = basin8.production_radius()
    basin = basin8.verify(points, ties, walls, t0, radius)
    require(F(T) < basin["t_lower"], "cutoff not below the optimum")
    return basin, radius


def expected(name, basin, radius):
    return dict(op="HEADER", version=2, n=8, Q=Q, root=[list(r) for r in support8.ROOTS[name]],
                cutoff_raw_t=T, optimum_raw_t=str(basin["t_upper"]), cover_radius=str(radius),
                basin_sha256=BASIN_HASH)


def build_receipt(workers):
    basin, radius = pinned()
    statements = {}
    for name in FAMILIES:
        m = json.loads((HERE / name / "manifest.json").read_text(), object_pairs_hook=unique_object)
        statements[name] = m["statement"]
    roots = [tuple(map(tuple, s["root"])) for s in statements.values()]
    require(len(set(roots)) == 8, "duplicated canonical root")
    for name, s in statements.items():
        require(s == expected(name, basin, radius), f"{name} statement mismatch")
    install_override()
    manifests.ProcessPoolExecutor = partial(ProcessPoolExecutor, initializer=install_override)
    results = {}
    for name in FAMILIES:
        print(f"Checking {name}", file=sys.stderr, flush=True)
        path = HERE / name / "manifest.json"
        manifest = json.loads(path.read_text(), object_pairs_hook=unique_object)
        rule = partial(support8.contract, name, threshold=F(T)) if name in CONTRACTED else None
        counts = manifests.verify_manifest(manifest, expected(name, basin, radius), path.parent,
                                           workers=workers, contract=rule)
        bound = F(counts["root_upper_bound"])
        require(bound in (F(T), basin["t_upper"]) and (bound == basin["t_upper"]) == bool(counts["cover_closures"]),
                "family conclusion mismatch")
        results[name] = dict(status="replayed", **counts, artifact_sha256=file_digest(path.open("rb"), "sha256").hexdigest())
    require(any(r["cover_closures"] for r in results.values()), "full replay unexpectedly excludes the optimum")
    return dict(complete=True, mode="replay",
                theorem=dict(n=8, domain="x>=0,y>=0,x+y<=1", domain_area="1/2",
                             units="raw triangle area; normalized value is twice raw",
                             cutoff_raw=T,
                             optimum="t* = common raw area of the 11 exact ties at P* (exp 113: the unique real root of "
                                     "1216512 t^7 + 1022976 t^6 - 2745456 t^5 + 2218730 t^4 + 32206304 t^3 "
                                     "- 19148615 t^2 + 3156103 t - 86233)",
                             optimum_bracket=[str(basin["t_lower"]), str(basin["t_upper"])],
                             canonical_roots={k: [list(r) for r in support8.ROOTS[k]] for k in FAMILIES}),
                dependencies=dict(basin_sha256=BASIN_HASH, centre_error=str(basin["rho"]),
                                  kantorovich_h=str(basin["kantorovich_h"]), cover_radius=str(radius),
                                  side_gap_bound_n6_raw="1/16"),
                families=results,
                conclusion="optimal value t* (outside the certified basin every configuration is at most the cutoff)")


if __name__ == "__main__":
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    print(json.dumps(build_receipt(a.workers), indent=2, sort_keys=True))
