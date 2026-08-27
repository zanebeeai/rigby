from __future__ import annotations

import argparse
import json
from pathlib import Path

from rigby_v2.release_ops import ReleaseEvidence, evaluate_release_checklist


parser = argparse.ArgumentParser()
parser.add_argument("evidence", type=Path, help="JSON array of release evidence records")
parser.add_argument("--evidence-root", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
records = tuple(
    ReleaseEvidence(**value)
    for value in json.loads(args.evidence.read_text(encoding="utf-8"))
)
decision = evaluate_release_checklist(
    root / "docs" / "v2" / "release-checklist.json",
    records,
    evidence_root=args.evidence_root,
)
print(json.dumps(decision.to_dict(), indent=2))
raise SystemExit(0 if decision.can_ship else 1)
