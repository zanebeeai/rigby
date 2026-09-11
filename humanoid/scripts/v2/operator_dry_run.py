from __future__ import annotations

import json

from rigby_v2.config import RuntimeSettings
from rigby_v2.release_ops.operator_dry_run import (
    OperatorDryRunConfig,
    run_operator_dry_run,
)


def main() -> int:
    settings = RuntimeSettings.from_env()
    evidence = run_operator_dry_run(
        OperatorDryRunConfig(
            project_root=settings.project_root,
            evidence_root=settings.project_root / "artifacts-v2" / "release-evidence",
            artifact_root=settings.artifact_root,
            database_url=settings.database_url,
        )
    )
    print(
        json.dumps(
            {
                "passed": True,
                "evidence": evidence.artifact_path,
                "evidence_sha256": evidence.artifact_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
