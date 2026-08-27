from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from rigby_v2.release_ops import run_production_renderer_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure five production 10-second MuJoCo evidence candidates."
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.artifact_root is None:
        with tempfile.TemporaryDirectory(prefix="rigby-production-benchmark-") as root:
            report = run_production_renderer_benchmark(Path(root))
    else:
        report = run_production_renderer_benchmark(arguments.artifact_root)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

