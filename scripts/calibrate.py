#!/usr/bin/env python3
"""
Standalone calibration runner — safe to call from cron when bot is stopped.

Reads resolved trade history, updates calibration parameters, and writes
the self-report. No trading, no network calls.
"""

import sys
from pathlib import Path

POLYCLAW_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(POLYCLAW_DIR))

from dotenv import load_dotenv
load_dotenv(POLYCLAW_DIR / ".env")

from lib.calibrator import CalibrationEngine, REPORT_FILE

engine = CalibrationEngine()
params = engine.calibrate()

print(f"Calibration complete: {params.total_resolved} resolved trades")
print(f"Maturity boost: +{params.maturity_edge_boost:.2%}")
print(f"Edge min: {params.edge_min_edge:.2%}  Longshot min: {params.longshot_min_edge:.2%}")
print(f"Category factors: crypto={params.category_kelly_crypto:.2f}  "
      f"sports={params.category_kelly_sports:.2f}  "
      f"politics={params.category_kelly_politics:.2f}")

if REPORT_FILE.exists():
    print("\n" + REPORT_FILE.read_text())
