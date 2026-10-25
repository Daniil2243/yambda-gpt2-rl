"""One-command reproduction using the included prepared data."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for module, arguments in [
    ("recsys.verify", []),
    ("recsys.experiment", []),
    ("recsys.analysis", []),
    ("recsys.verify", ["--require-results"]),
    ("recsys.report", []),
]:
    print(f"Running {module}", flush=True)
    subprocess.run([sys.executable, "-m", module, *arguments], cwd=root, check=True)
