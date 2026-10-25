"""Prepare missing data, train all models, and regenerate metrics and figures."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
prepared = root / "data/processed"
required = ["train.npz", "validation.npz", "test.npz", "audit.json", "catalog.json"]
if not all((prepared / name).is_file() for name in required):
    print("Preparing data from the checksum-verified Yambda source (426 MB).", flush=True)
    subprocess.run([sys.executable, "-m", "recsys.data", "--download"], cwd=root, check=True)

for module, arguments in [
    ("recsys.verify", []),
    ("recsys.experiment", []),
    ("recsys.analysis", []),
    ("recsys.verify", ["--require-results"]),
]:
    print(f"Running {module}", flush=True)
    subprocess.run([sys.executable, "-m", module, *arguments], cwd=root, check=True)
