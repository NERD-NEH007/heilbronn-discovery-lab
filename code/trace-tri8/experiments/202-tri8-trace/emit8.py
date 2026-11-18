"""Exp 202: emit one triangle n = 8 family trace into ./<FAMILY>/ (resume semantics). usage: emit8.py FAMILY"""
import sys
from fractions import Fraction as F
from pathlib import Path
HERE = Path(__file__).resolve().parent
import run_family8 as rf
name = sys.argv[1]; out = HERE / name; out.mkdir(exist_ok=True)
rule = (lambda box: rf.contract(name, box, F(rf.CUTOFF))) if name in rf.PAIRS else None
rf.emit(name, out, lambda: rf.statement(name), cap=16384, contract=rule, use_cover=True, resume=True)
