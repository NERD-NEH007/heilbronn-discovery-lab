"""Exp 186 parallel helper: emit one support subtree of canonical SSS0 into SSS0/ (resume semantics).
usage: emit_subtree.py PATH [--check]   (--check only walks to PATH and exits)"""
import sys
from canonical import DIGEST
from pilot import HERE, expected
import run_family_sub as rf
import stream_trace
from run_family import graph_boxes
path = sys.argv[1]; check = "--check" in sys.argv
header,rule = expected("",True)
stream_trace.graph_boxes = graph_boxes
if check:
    __import__("os").environ["WALK_ONLY"] = "1"
rf.emit("SSS0",HERE/"SSS0",lambda:header,cap=16384,contract=rule,use_cover=True,resume=True,
        canonical_contract=rule,reduction_digest=DIGEST,subtree=path)
