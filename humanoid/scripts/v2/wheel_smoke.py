from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rigby_v2.release_ops import verify_wheel_smoke


ROOT = Path(__file__).resolve().parents[2]
report = verify_wheel_smoke(ROOT)
print(json.dumps(asdict(report), indent=2))
raise SystemExit(0 if report.passed else 1)
