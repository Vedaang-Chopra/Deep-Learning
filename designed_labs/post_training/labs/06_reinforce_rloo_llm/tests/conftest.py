"""Make starter.py importable from tests without installing anything."""

import sys
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))
