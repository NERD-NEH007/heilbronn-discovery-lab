"""Run the default-byte compatibility gate before starting canonical emission."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
for script in ("default_fixture.py", "production.py"):
    subprocess.run([sys.executable,"-B","-u",str(HERE/script)],cwd=HERE,check=True)
