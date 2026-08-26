from __future__ import annotations

import argparse
import json
from pathlib import Path

from rigby_v2.artifacts import ContentAddressedArtifactStore
from rigby_v2.config import RuntimeSettings
from rigby_v2.library import (
    DeferredLegacyIndexImporter,
    LegacyArchiveImporter,
    LegacyRepositorySink,
    PostgresLibraryRepository,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import v1 result folders only as non-retrievable legacy candidates."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--mode",
        choices=("full", "metadata-only"),
        default="full",
        help=(
            "full verifies and copies source payloads into CAS; metadata-only reads index.json "
            "and creates non-promotable deferred records without copying payloads"
        ),
    )
    arguments = parser.parse_args()
    settings = RuntimeSettings.from_env()
    if arguments.mode == "metadata-only":
        importer = DeferredLegacyIndexImporter(arguments.archive)
    else:
        importer = LegacyArchiveImporter(
            arguments.archive,
            artifacts=ContentAddressedArtifactStore(settings.artifact_root),
        )
    sink = LegacyRepositorySink(PostgresLibraryRepository(settings.database_url))
    records = importer.import_all(sink, limit=arguments.limit)
    print(
        json.dumps(
            {
                "imported_or_already_present": len(records),
                "status": "legacy_candidate",
                "certified": 0,
                "release_id": None,
                "mode": arguments.mode,
                "requires_hydration": arguments.mode == "metadata-only",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
