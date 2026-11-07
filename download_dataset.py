"""
Optional helper: download the dataset from Kaggle using the Kaggle CLI.
First configure your Kaggle API token (see README.md), then run:
    python download_dataset.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

command = [
    sys.executable, "-m", "kaggle", "datasets", "download",
    "-d", "clmentbisaillon/fake-and-real-news-dataset",
    "-p", str(DATA_DIR), "--unzip"
]
print("Downloading dataset from Kaggle...")
try:
    subprocess.run(command, check=True)
except FileNotFoundError:
    print("Kaggle CLI not found. Install it with: pip install kaggle")
    raise
except subprocess.CalledProcessError:
    print("Download failed. Check Kaggle API credentials and dataset access.")
    raise

required = [DATA_DIR / "Fake.csv", DATA_DIR / "True.csv"]
missing = [p.name for p in required if not p.exists()]
if missing:
    print("Download finished, but expected file(s) were not found:", ", ".join(missing))
else:
    print("Dataset ready: data/Fake.csv and data/True.csv")
