from __future__ import annotations

import json
from pathlib import Path

from rigby_v2.release_ops import run_fresh_machine_preflight


ROOT = Path(__file__).resolve().parents[2]
report = run_fresh_machine_preflight(project_root=ROOT)
print(json.dumps(report.to_dict(), indent=2, default=str))
raise SystemExit(0 if report.passed else 1)
