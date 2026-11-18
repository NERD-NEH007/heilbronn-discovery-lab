"""Exp 202: mutation tests for the n = 8 replay checker (like exp 165 for n = 9). Each test copies the MMS
family into a temporary folder, applies one corruption, fixes the manifest hash where needed (so that only the
semantic check can catch it), and requires the replay to raise. A clean copy must pass."""
import json, shutil, tempfile, hashlib
from fractions import Fraction as F
from functools import partial
from pathlib import Path
import verify_release8 as R

basin, radius = R.pinned()
R.install_override()
NAME = "MMS"
SRC = R.HERE / NAME


def replay(folder, expected=None):
    m = json.loads((folder / "manifest.json").read_text())
    rule = partial(R.support8.contract, NAME, threshold=F(R.T))
    return R.manifests.verify_manifest(m, expected or R.expected(NAME, basin, radius), folder, workers=1, contract=rule)


def rehash(folder, shard):
    m = json.loads((folder / "manifest.json").read_text())
    digest = hashlib.sha256((folder / shard).read_bytes()).hexdigest()
    def walk(node):
        if node.get("path") == shard: node["sha256"] = digest
        for c in node.get("children", []): walk(c)
        if node.get("next"): walk(node["next"])
    walk(m["partition"]); (folder / "manifest.json").write_text(json.dumps(m))


def mutate(edit):
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d) / NAME; shutil.copytree(SRC, folder)
        edit(folder)
        try:
            replay(folder); return False
        except Exception:
            return True


def biggest_shard(folder):
    return max((p for p in folder.glob("shard-*.jsonl")), key=lambda p: p.stat().st_size).name


def zero_p2(folder):
    s = biggest_shard(folder); lines = (folder / s).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if '"CLOSE_P2"' in l)
    rec = json.loads(lines[i]); rec["weights"] = rec["weights"][:1]   # one triangle alone cannot certify this box
    lines[i] = json.dumps(rec, separators=(",", ":")); (folder / s).write_text("\n".join(lines) + "\n"); rehash(folder, s)


def truncate(folder):
    s = biggest_shard(folder); lines = (folder / s).read_text().splitlines()
    (folder / s).write_text("\n".join(lines[:-1]) + "\n"); rehash(folder, s)


def flip_bound(folder):
    s = biggest_shard(folder); lines = (folder / s).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if '"CLOSE_BOUND"' in l)
    rec = json.loads(lines[i]); rec["triple"] = (rec["triple"] + 1) % 56
    lines[i] = json.dumps(rec, separators=(",", ":")); (folder / s).write_text("\n".join(lines) + "\n"); rehash(folder, s)


def unhashed_edit(folder):
    s = biggest_shard(folder); (folder / s).write_text((folder / s).read_text() + " ")


results = {
    "clean copy passes": not mutate(lambda f: None),
    "wrong cutoff rejected": None,
    "P2 certificate cut to one weight rejected": mutate(zero_p2),
    "truncated shard rejected": mutate(truncate),
    "wrong bound triple rejected": mutate(flip_bound),
    "edited shard without rehash rejected": mutate(unhashed_edit),
}
try:
    bad = dict(R.expected(NAME, basin, radius), cutoff_raw_t=str(F(R.T) * 2))
    replay(SRC, bad); results["wrong cutoff rejected"] = False
except Exception:
    results["wrong cutoff rejected"] = True
print(json.dumps(results, indent=1))
assert all(results.values()), results
print("ALL MUTATION TESTS PASSED")
