from __future__ import annotations

import json

from rigby_v2.config import RuntimeSettings
from rigby_v2.release import (
    RIGBY_V2_SCHEMA_MIGRATION_SET,
    apply_rigby_v2_postgres_migrations,
)


def main() -> int:
    settings = RuntimeSettings.from_env()
    applied = apply_rigby_v2_postgres_migrations(settings.database_url)
    print(
        json.dumps(
            {
                "migration_set": RIGBY_V2_SCHEMA_MIGRATION_SET,
                "newly_applied": [
                    {
                        "version": value.version,
                        "name": value.name,
                        "checksum": value.checksum,
                    }
                    for value in applied
                ],
                "status": "migrated" if applied else "already_current",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
