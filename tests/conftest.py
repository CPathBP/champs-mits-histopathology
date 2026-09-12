import sys
from pathlib import Path

# The scripts are plain files; make them importable by module name.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "slides"))
