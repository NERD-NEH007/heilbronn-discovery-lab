"""Exp 186: run emit_subtree.py over a list of support paths, W at a time (each its own process and log)."""
import subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent; LOGS = HERE / "parallel-logs"; LOGS.mkdir(exist_ok=True)
paths = sys.argv[2].split(","); W = int(sys.argv[1])
def one(p):
    t = time.time()
    with open(LOGS / f"{p}.out", "w") as out, open(LOGS / f"{p}.err", "w") as err:
        rc = subprocess.run([sys.executable, "-B", "-u", str(HERE / "emit_subtree.py"), p], cwd=HERE, stdout=out, stderr=err).returncode
    line = f"{time.strftime('%H:%M:%S')} {p} rc={rc} wall={round(time.time() - t)}s"
    print(line, flush=True); return rc
with ThreadPoolExecutor(W) as ex:
    rcs = list(ex.map(one, paths))
print("ALL DONE" if not any(rcs) else f"FAILURES {rcs}", flush=True)
