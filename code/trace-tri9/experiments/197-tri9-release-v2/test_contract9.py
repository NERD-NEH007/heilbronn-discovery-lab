"""Test (not part of the checker): contract9.contract equals the original
163-tri9-trace-queue/queue_traces.contract (read from its source, not imported) on boundary and random boxes, for the release threshold and others.
Also checks that importing the release checker loads no emitter module. usage: python -B test_contract9.py"""
import random, subprocess, sys
from fractions import Fraction as F
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "152-geometry-trace"))
import contract9
import ast, math, types
# the ORIGINAL definitions, read from queue_traces.py without importing it (its imports load the emitters)
src = (ROOT / "163-tri9-trace-queue" / "queue_traces.py").read_text(encoding="utf-8")
keep = [nd for nd in ast.parse(src).body if (isinstance(nd, ast.FunctionDef) and nd.name in ("gap_grid", "contract"))
        or (isinstance(nd, ast.Assign) and all(isinstance(t, ast.Name) and t.id not in ("HERE",) for t in nd.targets)
            and not any(isinstance(t, ast.Tuple) for t in nd.targets) and nd.targets[0].id in ("ORDER", "H", "O", "ROOTS", "PAIRS"))
        or (isinstance(nd, ast.Assign) and isinstance(nd.targets[0], ast.Tuple))]
QT = types.SimpleNamespace()
ns = dict(F=F, isqrt=math.isqrt, require=contract9.require, Q=606208)
exec(compile(ast.Module(body=keep, type_ignores=[]), "queue_traces.py", "exec"), ns)
QT.Q, QT.ROOTS, QT.PAIRS, QT.gap_grid, QT.contract = ns["Q"], ns["ROOTS"], ns["PAIRS"], ns["gap_grid"], ns["contract"]
qline = next(l for l in (ROOT / "161-tri9-mss1-trace" / "run_family.py").read_text().splitlines() if l.startswith("GRID,Q ="))
assert qline == "GRID,Q = 37,37*2**14" and 37 * 2**14 == 606208, qline   # run_family.Q, which queue_traces imports

T = F("42999999957/1568000000000")
assert contract9.Q == QT.Q and contract9.ROOTS == QT.ROOTS and contract9.PAIRS == QT.PAIRS
rng = random.Random(9)
def boxes(root, k):
    for _ in range(k):
        out = []
        for r in root:
            x = sorted(rng.randint(r[1], r[2]) for _ in range(2)); y = sorted(rng.randint(r[3], r[4]) for _ in range(2))
            if r[0] == "H": y = [contract9.Q - x[1], contract9.Q - x[0]]
            out.append((r[0], *x, *y))
        yield tuple(out)
    yield tuple(root)
n = 0
for th in (T, F(1, 100), F(1, 30), F(7, 145)):
    assert contract9.gap_grid(th) == QT.gap_grid(th)
    for name, root in contract9.ROOTS.items():
        for b in boxes(root, 3000):
            assert contract9.contract(name, b, th) == QT.contract(name, b, th), (name, b, th); n += 1
print("contract9 agrees with queue_traces.contract on", n, "boxes")
code = ("import runpy,sys; sys.argv=['verify_release.py','--help']\n"
        "try: runpy.run_path(r'%s', run_name='not_main')\nexcept SystemExit: pass\n"
        "bad=[m for m in sys.modules if m in ('queue_traces','run_family','emit_mmm','emitter')]\n"
        "print('emitter modules loaded:', bad); sys.exit(1 if bad else 0)") % (HERE / "verify_release.py")
r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True); print(r.stdout.strip(), r.stderr[-500:])
assert r.returncode == 0
print("TEST PASSED")
