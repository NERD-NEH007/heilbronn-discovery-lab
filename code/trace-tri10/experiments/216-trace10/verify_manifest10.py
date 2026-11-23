"""Exp 216: verify a sharded triangle n = 10 trace (copy of exp 202 verify_manifest8; n = 10, support-level
narrowing with the statement's pinned rows through contract10, shards replayed by stream_trace10).
Exp 223: partition nodes {"M", "pair_exclusions": {path, sha256}, "next"} carry cell-pair exclusions verified here at
that node's support box (graph_replay10.pair_exclusions); every shard below inherits the union of its ancestors'
pairs, which its graph replay removes as edges."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "154-streaming-cover"))
sys.path.insert(0, str(HERE))
from stream_trace10 import read_root, read_rows, verify, contract10, graph_cells
from verify_trace import keys, read_record, require, split, unique_object
from graph_replay10 import pair_exclusions
from functools import lru_cache
import gzip


def _read_bytes(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as f:
        return f.read()


def pair_records(path, digest):
    """the records of one pair-exclusion file (JSON lines), after checking its digest (of the decompressed bytes)"""
    data = _read_bytes(path)
    require(sha256(data).hexdigest() == digest, "pair exclusion digest mismatch")
    out = []
    for line in data.decode("utf-8").splitlines():
        try:
            out.append(json.loads(line, object_pairs_hook=unique_object))
        except ValueError as exc:
            raise AssertionError("invalid pair exclusion JSON") from exc
    return out


@lru_cache(maxsize=8)
def _loaded_pairs(spec):
    """union of the pair sets of the files in spec; each file was verified by verify_manifest (or, in the pipelined
    run, by the driver with the same pair_exclusions) at its own support box before any shard uses it"""
    out = set()
    for path, digest in spec:
        for rec in pair_records(Path(path), digest):
            out.add(tuple(tuple(c) for c in rec["cells"]))
    return frozenset(out)


def replay_shard(task):
    path, expected, digest = task[:3]
    spec = tuple(task[3]) if len(task) > 3 else ()
    pairs = _loaded_pairs(spec) if spec else frozenset()
    hash_value = sha256()
    import gzip
    opener = gzip.open if path.suffix == ".gz" else open      # archived shards: digest is of the decompressed bytes
    with opener(path, "rb") as source:
        def lines():
            for raw in source:
                hash_value.update(raw)
                yield raw
        receipt = verify(lines(), expected, pairs=pairs, pairs_key="|".join(d for _, d in spec))
    require(hash_value.hexdigest() == digest, "trace digest mismatch")
    require(receipt["complete"], "a shard has pending obligations")
    if spec: receipt["pairs"] = [d for _, d in spec]
    return receipt


def _task(base, used, node, expected, spec=()):
    relative, digest = node["path"], node["sha256"]
    require(type(relative) is str and not Path(relative).is_absolute(), "invalid trace path")
    path = (base / relative).resolve()
    require(path.is_relative_to(base) and path not in used, "unsafe or duplicate trace path")
    require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "invalid trace digest")
    used.add(path)
    return (path, expected, digest, spec)


def verify_manifest(manifest, expected, base, workers=1, done=None):
    """done: optional {(path name, sha256): receipt} of shards already replayed by THIS checker (the pipelined run
    verifies each shard as it is written); those are not replayed again, everything else (structure, tiling,
    narrowing, remaining shards) is checked in full. A clean-room replay passes done=None."""
    require(type(workers) is int and workers >= 1, "invalid worker count")
    keys(manifest, "version statement partition")
    require(type(manifest["version"]) is int and manifest["version"] == 1, "unsupported manifest")
    require(manifest["statement"] == expected and expected.get("slice") is None, "manifest statement mismatch")
    q, n = expected["Q"], expected["n"]
    require(type(q) is int and q > 0 and n == 10, "invalid pinned dimensions")
    root = read_root(expected["root"], q, n)
    ordr = read_rows(expected["ord"], 5, 10); sumr = read_rows(expected["sum"], 4, 10)
    contract = lambda box: contract10(box, ordr, sumr, q)
    base = base.resolve()
    queue = [(root, manifest["partition"], ())]
    threshold = Fraction(expected["cutoff_raw_t"]); pair_files = 0
    tasks, used, contractions = [], set(), 0
    while queue:
        box, node, spec = queue.pop()
        require(type(node) is dict, "invalid partition node")
        if "pair_exclusions" in node:
            keys(node, "M pair_exclusions next")
            ref = node["pair_exclusions"]
            require(type(ref) is dict and type(node["M"]) is int and node["next"] is not None, "invalid pair exclusion node")
            keys(ref, "path sha256")
            path, _, digest = _task(base, used, ref, None)[:3]
            pair_exclusions(box, q, node["M"], threshold, pair_records(path, digest))
            pair_files += 1
            queue.append((box, node["next"], spec + ((str(path), digest),)))
        elif "contract" in node:
            keys(node, "contract next")
            require(node["contract"] == "SUPPORT_ROWS", "unsupported contraction")
            narrowed = contract(box)
            require(narrowed != box, "vacuous contraction")
            require((narrowed is None) == (node["next"] is None), "invalid contraction continuation")
            contractions += 1
            if narrowed is not None:
                queue.append((narrowed, node["next"], spec))
        elif "graph_slices" in node:
            keys(node, "M graph_slices")
            M, slices = node["M"], node["graph_slices"]
            require(type(M) is int and type(slices) is list and slices, "invalid graph slices")
            count = len(graph_cells(box, q, M, Fraction(expected["cutoff_raw_t"])))
            prev = 0
            for sl in slices:
                keys(sl, "range path sha256")
                a, b = sl["range"]
                require(type(a) is int and type(b) is int and a == prev and a < b, "slices must tile the cells in order")
                prev = b
                tasks.append(_task(base, used, sl, dict(expected, root=[list(r) for r in box], slice=[M, a, b]), spec))
            require(prev == count, "slices do not cover every cell")
        elif "children" in node:
            keys(node, "point axis cut children")
            boxes = split(box, node["point"], node["axis"], node["cut"], q)
            children = node["children"]
            require(type(children) is list and len(children) == len(boxes), "partition child missing or extra")
            queue.extend(reversed([(b_, c_, spec) for b_, c_ in zip(boxes, children)]))
        else:
            keys(node, "path sha256")
            tasks.append(_task(base, used, node, dict(expected, root=[list(row) for row in box]), spec))
    require(tasks or contractions, "manifest has no proof obligations")
    reused = []
    if done:
        keep = []
        for t in tasks:
            r = done.get((t[0].name, t[2]))
            if r is not None and t[0].exists() and r.get("pairs", []) == [d for _, d in t[3]]:
                require(r.get("complete") is True, "a recorded receipt is incomplete"); reused.append(r)
            else:
                keep.append(t)
        tasks = keep
    if workers == 1 or not tasks:
        receipts = reused + list(map(replay_shard, tasks))
    else:
        import os
        # sibling workers share the forbidden-cell masks they compute (graph_replay10 shared cache, local disk);
        # a clean-room replay can unset REPLAY_MASK_CACHE to recompute everything independently
        # (review 2026-11-18: no default cache folder any more; the cache is used only if the caller sets it to
        # local scratch, as the pipelined driver does. A clean-room replay leaves it unset.)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # consecutive tasks share a support box (DFS order): chunks keep them on one worker, whose graph cache
            # then serves every slice of that support
            receipts = reused + list(pool.map(replay_shard, tasks, chunksize=max(1, len(tasks) // (4 * workers))))
    upper = max([Fraction(expected["cutoff_raw_t"]), *(Fraction(r["root_upper_bound"]) for r in receipts)])
    out = dict(shards=len(receipts), support_contractions=contractions,
                ordinary_closures=sum(r["ordinary_closures"] for r in receipts),
                cover_closures=sum(r["cover_closures"] for r in receipts), root_upper_bound=str(upper))
    if pair_files: out["pair_exclusion_files"] = pair_files
    return out


if __name__ == "__main__":
    file = Path(sys.argv[1])
    manifest = json.loads(file.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    expected = read_record(Path(sys.argv[2]).read_text(encoding="utf-8"))
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    print(json.dumps(verify_manifest(manifest, expected, file.parent, workers)))
