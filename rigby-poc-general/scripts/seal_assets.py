"""Write .sha256 sidecars for the sealed assets.

Sealing is deliberate. The inventory and the benchmark are the system's fixed
points, and a digest that regenerates itself silently is not a seal at all -- so
this is a script someone runs, never something the loader does on the fly.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "general"

SEALED = (
    ASSETS / "schema_inventory.v1.json",
    ASSETS / "benchmark" / "robots.json",
    ASSETS / "benchmark" / "prompts.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without writing")
    arguments = parser.parse_args()

    failures = 0
    for path in SEALED:
        if not path.is_file():
            print(f"skip   {path.name} (absent)")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if arguments.check:
            current = (
                sidecar.read_text(encoding="ascii").strip().split()[0].lower()
                if sidecar.is_file()
                else ""
            )
            status = "ok" if current == digest else "MISMATCH"
            failures += 0 if status == "ok" else 1
            print(f"{status:<8} {path.name}")
        else:
            # Bytes, not text. ``write_text`` translates the newline on
            # Windows, so re-sealing there writes a CRLF sidecar and puts back
            # the exact platform split this seal was just fixed for. The file
            # is pinned -text in .gitattributes, so whatever is written here is
            # what gets committed.
            sidecar.write_bytes(f"{digest}  {path.name}\n".encode("ascii"))
            print(f"sealed {path.name}  {digest[:16]}...")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
