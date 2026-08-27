from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def atomic_write_json(
    path: Path,
    value: dict[str, Any],
    *,
    replace_attempts: int = 12,
) -> None:
    """Persist JSON without exposing a partial file to readers.

    Windows search/indexing and OneDrive can briefly hold a just-written file.
    A unique sibling temporary avoids writer collisions, while bounded replace
    retries tolerate those transient locks without weakening atomicity.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, indent=2, ensure_ascii=False)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(max(1, replace_attempts)):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt + 1 >= replace_attempts:
                    raise
                time.sleep(min(0.02 * (2**attempt), 0.25))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # A failed final cleanup must not hide the write/replace outcome.
            pass
