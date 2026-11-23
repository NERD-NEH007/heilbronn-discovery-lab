"""Exp 232 (Claude, 2026-11-20): strict eight-family release receipt for the triangle n = 10 certificate trace.

What it does (checker modules only; the search engine and emitters are never imported, asserted below):
  1. pins the statement constants (Q, cutoff, family roots) as literals and checks the statement builder
     (216 support10) agrees with them;
  2. re-derives the eight family statements from scratch: support10.statement() re-verifies the n = 10 basin at the
     cover radius 1/200 (246 basin10w, the wide basin lemma, on 214 basin10's KKT root) and the n = 8 basin behind
     the side gap (202 basin8);
  3. re-verifies the n = 10 basin (exp 246 wide lemma) at the cover radius, and the enclosure uniqueness check (140);
  4. replays every family manifest with the 216 verify_manifest10 (clean room: no mask cache, no saved receipts);
  5. requires: every family complete; root bound = cutoff for families without cover closures, = the pinned
     optimum upper endpoint for families with cover closures; at least one family with covers (the attained case).

Usage (from an empty extraction, or the lab tree):
    python -B experiments/232-tri10-release/verify_release10.py --archive <dir with MMM/ ... SSS3/> --workers 4 > receipt.json
    --preflight : lab tree only: statements plus the pipeline's saved counts (no replay; proves nothing; the
                  released package does not ship the pipeline receipts, so it is unavailable there)
Run without -O (dependency assertions must stay active). Status: final release checker (2026-11-22); its end-to-end
and tamper tests are in tools/release_tests/.
"""
from argparse import ArgumentParser
from fractions import Fraction as F
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

if not __debug__:
    raise RuntimeError("Run without -O to preserve the dependency assertions")
if os.environ.get("REPLAY_MASK_CACHE"):
    raise RuntimeError("Unset REPLAY_MASK_CACHE: the release replay must recompute every graph mask")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "216-trace10"))
import support10                                   # statement pieces (imports basin8, basin10, stream_local10)
import verify_manifest10 as manifests              # manifest + shard replay (imports stream_trace10, graph_replay10)
import basin10
import basin10w
from verify_trace import require, unique_object

# The checker closure: every module loaded from the experiments tree must be one of these files (an allowlist, so no
# search or emitter code can be imported, whatever its name). The enclosure check (140) is imported later and is
# checked again in dependencies().
CLOSURE = {"152-geometry-trace/verify_trace.py", "202-tri8-trace/basin8.py", "213-close-mcc/mcc_check.py",
           "214-basin10/basin10.py", "245-basin-sharp/basin10s.py", "246-basin-wider/basin10w.py",
           "215-trace10-local/stream_local10.py", "216-trace10/support10.py",
           "216-trace10/verify_manifest10.py", "216-trace10/stream_trace10.py", "216-trace10/graph_replay10.py",
           "140-triangle-n10-enclosure/verify.py", "232-tri10-release/verify_release10.py"}


def outside_closure():
    bad = []
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if ROOT in p.parents and p.relative_to(ROOT).as_posix() not in CLOSURE:
            bad.append(p.relative_to(ROOT).as_posix())
    return bad


assert not outside_closure(), f"a module outside the checker closure was imported: {outside_closure()}"

FAMILIES = ("MMM", "MMS", "MSS0", "MSS1", "SSS0", "SSS1", "SSS2", "SSS3")
Q = 770048
CUTOFF = "108441851695034062548779343914599525513581/5000000000000000000000000000000000000000000"
BASIN_SHA = "46ef3f6e8f76224cba2a726f15607800fc904439bf0398cf6d236ac38776e49c"
COVER_RADIUS = "1/200"                                                                 # exp 246 (paper: basin lemma)
TRI8_RECEIPT_SHA = "f735bc895bca87549ece77abc00b9544e76612103be741a22b17f522336d6fa9"   # n = 8 release v1.1 (LF)
PAPER_GAP = 166946                                                                     # Section 2 (side gap, grid units)
PAPER_RHO = F(16, 10 ** 32)                                                                     # Theorem 1(1)
PAPER_DELTA = (F("0.04337674067805700176018979456740"), F("0.04337674067805700176018979456784"))  # Theorem 1(2)
H = Q // 2
O, X, Y = ["R", 0, 0, 0, 0], ["R", Q, Q, 0, 0], ["R", 0, 0, Q, Q]
MB, ML, MH = ["R", H, H, 0, 0], ["R", 0, 0, H, H], ["R", H, H, H, H]
B1, B2 = ["R", 0, H, 0, 0], ["R", H, Q, 0, 0]
L1, L2 = ["R", 0, 0, 0, H], ["R", 0, 0, H, Q]
H1, H2 = ["H", 0, H, H, Q], ["H", H, Q, 0, H]
ROOTS = dict(MMM=[MB, ML, MH], MMS=[B1, B2, ML, MH], MSS0=[B1, B2, L1, L2, MH], MSS1=[O, B2, L2, MH],
             SSS0=[B1, B2, L1, L2, H1, H2], SSS1=[O, B2, L2, H1, H2], SSS2=[O, X, L2, H1], SSS3=[O, X, Y])


def digest(path):
    h = sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def statements():
    require(support10.Q == Q and support10.CUTOFF == CUTOFF, "statement builder constants differ from the pinned ones")
    require(digest(basin10.SOURCE) == BASIN_SHA, "local certificate digest mismatch")
    out = {}
    for name in FAMILIES:
        s = support10.statement(name)
        require(s["root"] == ROOTS[name], f"{name}: root differs from the pinned canonical root")
        require(s["cutoff_raw_t"] == CUTOFF and s["basin_sha256"] == BASIN_SHA and s["slice"] is None
                and s["cover_radius"] == COVER_RADIUS, f"{name}: header")
        out[name] = s
    require(len({json.dumps(s["root"]) for s in out.values()}) == 8, "duplicated family root")
    require(len({(s["optimum_raw_t"], s["cover_radius"]) for s in out.values()}) == 1, "families disagree on the basin")
    return out


UNIQUENESS_RADIUS = F(1, 10 ** 5)


U0_FILE = Path(__file__).resolve().parent / "kkt_centre_u0.json"
U0_SHA = "96e422a5cffb60c18ae1a8a207eadf7cfb8887868ba6b7009d6a76868fff0b35"   # the archived centre (paper: Theorem 1(1))


def kkt_uniqueness(points, ties, walls, t0, radius=UNIQUENESS_RADIUS):
    """Theorem 1(1): the KKT root is the only solution within `radius` (sup norm, all 41 KKT variables) of the
    ARCHIVED refined centre u0 (kkt_centre_u0.json, pinned by U0_SHA; API panel 2026-11-21: the centre must not be
    one recomputed through floats, which could differ between machines). J is affine in u with Lipschitz constant L
    (basin10 docstring), so x -> x - Y G(x) has derivative norm <= d + |Y| L radius on that ball; below 1 it is a
    contraction there, and two roots in the ball would be two fixed points. Y is any matrix (floats only propose it;
    d is computed exactly). The receipt also reports whether this machine's float recomputation of u0 (two Newton
    steps from the certificate, as basin10.existence) reproduces the archived centre; that is informational."""
    signs = [1 if basin10.determinant(*(points[i] for i in tr)) > 0 else -1 for tr in ties]
    require(digest(U0_FILE) == U0_SHA, "archived KKT centre digest mismatch")
    doc = json.loads(U0_FILE.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    require(doc.get("local_certificate_sha256") == BASIN_SHA, "archived KKT centre is for another local certificate")
    require(type(doc.get("u0")) is list and len(doc["u0"]) == 41 and all(type(s) is str for s in doc["u0"]),
            "archived KKT centre must be 41 rational strings")
    u = [F(s) for s in doc["u0"]]
    r = [c for p in points for c in p] + [t0] + [F(1, 28)] * 14 + [F(0)] * 6
    for _ in range(2):
        G, J = basin10.kkt(r, ties, walls, signs)
        Y = basin10.float_inverse(J)
        r = [F(round(v * 10 ** 60), 10 ** 60) for v in (a - b for a, b in zip(r, basin10.matvec(Y, G)))]
    recomputed_matches = r == u
    # basin10.existence certifies its root within ~1e-31 of the recomputed centre r; that root must lie in this
    # uniqueness ball about u, so both name the same root (holds with equality r == u on IEEE double machines)
    require(basin10.maxabs([a - b for a, b in zip(r, u)]) < radius / 2, "recomputed centre far from the archived one")
    G, J = basin10.kkt(u, ties, walls, signs)
    Y = basin10.float_inverse(J)
    I_YJ = basin10.matmul(Y, J)
    d = max(sum(abs(F(i == j) - I_YJ[i][j]) for j in range(41)) for i in range(41))
    bound = d + basin10.norm(Y) * basin10.LIP * radius
    require(bound < 1, "KKT uniqueness: no contraction on the stated ball")
    # existence about the SAME centre u (Kantorovich, as in basin10.existence), so the unique root of the ball is the
    # certified one: it lies within 2 eta of u, inside the ball (audit 0024, finding 2)
    require(d < 1, "approximate inverse too poor")
    eta = basin10.maxabs(basin10.matvec(Y, G)) / (1 - d)
    require(2 * (basin10.norm(Y) / (1 - d)) * basin10.LIP * eta <= 1, "Kantorovich condition fails at the centre")
    require(2 * eta < radius, "existence ball not inside the uniqueness ball")
    require(min(u[21:41]) - 2 * eta > 0, "multipliers not robustly positive")
    centre = [str(v) for v in u]
    return dict(radius=str(radius), contraction_bound=float(bound), existence_radius=str(2 * eta), centre=centre,
                centre_sha256=sha256(json.dumps(centre).encode()).hexdigest(), centre_file_sha256=U0_SHA,
                centre_recomputed_matches=recomputed_matches)


def tri8_upper():
    """U_8 of the side-gap lemma: the certified GLOBAL upper bound of the published n = 8 theorem (release receipt of
    the triangle n = 8 trace, Zenodo 10.5281/zenodo.24337873, independently replayed). The checker's local
    recomputation (support10.delta8_upper) must not be smaller than it (audit 0024, finding 1)."""
    path = ROOT / "202-tri8-trace" / "release-receipt.json"
    require(digest(path) == TRI8_RECEIPT_SHA, "n = 8 release receipt digest mismatch")
    r = json.loads(path.read_text(encoding="utf-8"))
    require(r.get("complete") is True and len(r["families"]) == 8, "n = 8 receipt is not a complete theorem receipt")
    u8 = max(F(f["root_upper_bound"]) for f in r["families"].values())
    require(F(support10.delta8_upper()) >= u8, "side-gap U_8 is below the published global n = 8 upper bound")
    return u8


def dependencies(stmts):
    points, ties, walls, t0 = basin10.load()
    wide = basin10w.verify_wide(points, ties, walls, t0, F(COVER_RADIUS))      # the lemma behind every cover
    uniq = kkt_uniqueness(points, ties, walls, t0)
    u8 = tri8_upper()
    require(support10.gap_grid(CUTOFF) == PAPER_GAP, "side gap differs from the value stated in the paper")
    # the paper's stated numbers, checked exactly (audit 0024, finding 2)
    require(wide["rho"] <= PAPER_RHO, "root ball larger than Theorem 1(1) states")
    require(PAPER_DELTA[0] < 2 * wide["t_lower"] and 2 * wide["t_upper"] < PAPER_DELTA[1],
            "value enclosure is not inside Theorem 1(2)'s interval")
    sys.path.insert(0, str(ROOT / "140-triangle-n10-enclosure"))
    import verify as enclosure
    # Enclosure uniqueness over the root ball basin10 has just certified (not the Exp 138 radius).
    enc = enclosure.verify(rho=wide["rho"])
    require(not outside_closure(), f"a module outside the checker closure was imported: {outside_closure()}")
    require(enc.get("unique_enclosure_area") == "1/2", "enclosure check did not conclude")
    any_s = next(iter(stmts.values()))
    return dict(basin_sha256=BASIN_SHA, optimum_upper_raw=any_s["optimum_raw_t"], cover_radius=any_s["cover_radius"],
                optimum_lower_raw=str(wide["t_lower"]), root_ball_rho=str(wide["rho"]), kkt_uniqueness=uniq,
                delta10_interval=[str(2 * wide["t_lower"]), str(2 * wide["t_upper"])],
                wide_basin_radius=COVER_RADIUS, basin_cells=wide["cells"],
                side_gap_grid=support10.gap_grid(CUTOFF), delta8_upper_raw=str(support10.delta8_upper()),
                tri8_global_upper_raw=str(u8), tri8_receipt_sha256=TRI8_RECEIPT_SHA,
                paper_rho=str(PAPER_RHO), paper_delta10=[str(v) for v in PAPER_DELTA],
                enclosure=enc, enclosure_sha256=digest(Path(enclosure.__file__)),
                checker_sha256={Path(m.__file__).name: digest(m.__file__) for m in
                                (manifests, support10, basin10, basin10w, sys.modules["basin10s"],
                                 sys.modules["stream_trace10"],
                                 sys.modules["graph_replay10"], sys.modules["stream_local10"],
                                 sys.modules["mcc_check"], sys.modules["basin8"], sys.modules["verify_trace"])})


def check_family(name, folder, header, workers, preflight):
    mpath = folder / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    require(manifest.get("statement") == header, f"{name}: manifest statement differs from the re-derived statement")
    if preflight:
        require((folder / "replay-receipt.json").exists(),
                f"{name}: --preflight needs the pipeline's replay-receipt.json (lab tree only; not in the release package)")
        saved = json.loads((folder / "replay-receipt.json").read_text(encoding="utf-8"))
        counts = {k: saved[k] for k in ("shards", "support_contractions", "ordinary_closures", "cover_closures",
                                        "root_upper_bound")}
    else:
        r = manifests.verify_manifest(manifest, header, folder, workers)
        counts = {k: r[k] for k in ("shards", "support_contractions", "ordinary_closures", "cover_closures",
                                    "root_upper_bound")}
    bound = F(counts["root_upper_bound"])
    covers = bool(counts["cover_closures"])
    require(bound == (F(header["optimum_raw_t"]) if covers else F(CUTOFF)), f"{name}: family conclusion mismatch")
    return dict(status="preflight" if preflight else "replayed", manifest_sha256=digest(mpath), **counts)


def build_receipt(archive, workers, preflight):
    stmts = statements()
    deps = dependencies(stmts)
    fams = {}
    for name in FAMILIES:
        print(f"Checking {name}", file=sys.stderr, flush=True)
        fams[name] = check_family(name, archive / name, stmts[name], workers, preflight)
    complete = not preflight and all(f["status"] == "replayed" for f in fams.values())
    if complete:
        require(any(f["cover_closures"] for f in fams.values()), "replay unexpectedly excludes the attained candidate")
    return dict(complete=complete, mode="preflight" if preflight else "replay",
                theorem=dict(n=10, domain="x>=0,y>=0,x+y<=1", domain_area="1/2",
                             units="raw triangle area; normalized value is twice raw", cutoff_raw=CUTOFF,
                             optimum="unique KKT root in the certified ball of the local certificate",
                             canonical_roots=ROOTS),
                dependencies=deps, families=fams,
                totals={k: sum(f[k] for f in fams.values()) for k in ("shards", "ordinary_closures", "cover_closures")},
                conclusion="optimal value and S3 uniqueness modulo labels" if complete else None)


if __name__ == "__main__":
    ap = ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()
    print(json.dumps(build_receipt(a.archive.resolve(), a.workers, a.preflight), indent=2, sort_keys=True))
