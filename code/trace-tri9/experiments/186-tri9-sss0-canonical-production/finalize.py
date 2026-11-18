"""Exp 186 final step (Claude, 2026-11-17), equivalent to `production.py --resume` but with a parallel replay.
1. Emission with resume=True (original run_family.emit): reuses every shard written by the parallel helpers,
   computes anything missing, and writes manifest.json.
2. Independent replay: the same checks as pilot.replay, with verify_manifest over W worker processes. Each
   worker installs the graph_replay.graph_boxes override before any shard is checked (as the single-worker
   production path does in-process)."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json, sys
from canonical import DIGEST, Q
from pilot import HERE, expected
from run_family import graph_boxes
from run_family_sub import emit   # same emitter plus the known-failed split (see run_family_sub.KNOWN_FAILED)
import stream_trace, verify_manifest as vm
from verify_trace import keys, require, unique_object


ORIGINAL_REPLAY = vm.replay_shard


def checked_replay(task):
    require(stream_trace.graph_boxes is graph_boxes, "graph override missing in replay worker")
    return ORIGINAL_REPLAY(task)


def install_override():
    stream_trace.graph_boxes = graph_boxes
    vm.verify = stream_trace.verify
    vm.replay_shard = checked_replay


def main():
    W = int(sys.argv[1]); output = HERE / "SSS0"
    install_override()
    header, rule = expected("", True)
    if not (output / "manifest.json").exists():
        emit("SSS0", output, lambda: header, cap=16384, contract=rule, use_cover=True, resume=True,
             canonical_contract=rule, reduction_digest=DIGEST)
    manifest = json.loads((output / "manifest.json").read_bytes(), object_pairs_hook=unique_object)
    keys(manifest, "scope reduction_sha256 proof")
    require(manifest["scope"] == "canonical SSS0" and manifest["reduction_sha256"] == DIGEST, "canonical reduction mismatch")
    vm.ProcessPoolExecutor = partial(ProcessPoolExecutor, initializer=install_override)
    receipt = vm.verify_manifest(manifest["proof"], header, output, workers=W, contract=rule, canonical_contract=rule)
    receipt.update(reduction_sha256=DIGEST, Q=Q, scope="canonical SSS0", replay_workers=W)
    tmp = output / "replay-receipt.tmp"; tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    tmp.replace(output / "replay-receipt.json"); print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
