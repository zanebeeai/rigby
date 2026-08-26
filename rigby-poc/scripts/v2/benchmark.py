from __future__ import annotations

import json

from rigby_v2.release_ops import run_local_reference_benchmark


report = run_local_reference_benchmark()
print(json.dumps(report.to_dict(), indent=2))
raise SystemExit(0 if report.passed else 1)
