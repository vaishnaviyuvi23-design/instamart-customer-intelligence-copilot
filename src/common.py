"""Shared helpers: every script is run from the repo root; this makes that true regardless of the caller's cwd."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # so `import mock_llm` works from src/ scripts

RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"
for d in (CLEAN, OUT, STATE):
    d.mkdir(parents=True, exist_ok=True)
