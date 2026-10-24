"""PrintPick real-hardware toolkit: Creality printer status + G-code part extraction + SO-101 hover.

Self-contained: nothing here imports the simulation packages (farm/, plan/, sim/, perceive/).
See real/README.md for the hardware-day checklist.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ASSETS = ROOT / "assets"
RUNS = REPO / "runs"
