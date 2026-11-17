"""scripts/ isn't a package (it's a collection of entry-point CLIs, not library code), so
make it importable for tests without restructuring it into src/."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
