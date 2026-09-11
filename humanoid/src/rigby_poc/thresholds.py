"""The single versioned home for every numeric gate.

``config/thresholds.v1.json`` is the source of truth. Nothing in this module
invents a value: every entry is ported from a config file or from the code site
that already enforced it, and every entry carries a structured ``source`` that
says where the number came from.

Plan: ``docs/plans/08-threshold-consolidation.md``. 08a ships the file, this
loader and its schema test; 08b repoints the Python call sites onto it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_FILE = PROJECT_ROOT / "config" / "thresholds.v1.json"

SourceKind = Literal["measured", "invariant", "external", "provisional"]

#: A source that claims empirical derivation must be able to state its n.
#: Recording a ceiling as a bare float is how the current global kinematic
#: ceilings came to be derived from six clips of one gesture.
#:
#: It must also state its ``reference_frame``. A number can carry a correct n,
#: be computed correctly, and still be wrong because it was measured against the
#: wrong zero -- lane ``anatomy`` published and then retracted an elbow
#: hyperextension figure that compared rig-rest-relative deltas against clinical
#: references assuming anatomical neutral. No n-and-baseline check catches that;
#: stating the reference next to the number does.
EMPIRICAL_KINDS: frozenset[str] = frozenset({"measured"})

#: Required keys per source kind, beyond ``kind`` and ``cites``.
REQUIRED_SOURCE_FIELDS: dict[str, frozenset[str]] = {
    "measured": frozenset({"n", "derived_from", "date", "reference_frame"}),
    "invariant": frozenset({"rationale"}),
    "external": frozenset({"derived_from"}),
    "provisional": frozenset({"rationale"}),
}


class ThresholdError(LookupError):
    """A threshold was requested that the config does not define."""


@dataclass(frozen=True)
class Threshold:
    """One entry of ``config/thresholds.v1.json``."""

    key: str
    unit: str
    applies_to: tuple[str, ...]
    source: dict[str, Any]
    value: Any = None
    default: Any = None
    by_family: dict[str, Any] | None = None

    @property
    def is_per_family(self) -> bool:
        return self.by_family is not None

    def for_family(self, family: str | None = None) -> Any:
        """Resolve the value, preferring a per-family override when one exists.

        A per-family entry with an empty ``by_family`` resolves to ``default``
        everywhere. That is the deliberate 08 §6.2 state, not an error: the
        per-family numbers are derived from corpus percentiles in plan 10.
        """

        if not self.is_per_family:
            return self.value
        assert self.by_family is not None
        if family is not None and family in self.by_family:
            return self.by_family[family]
        return self.default

    def applies(self, family: str) -> bool:
        return "*" in self.applies_to or family in self.applies_to


@lru_cache(maxsize=1)
def _document() -> dict[str, Any]:
    return json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def thresholds() -> dict[str, Threshold]:
    """Every threshold, keyed by its dotted name."""

    raw = _document()["thresholds"]
    return {
        key: Threshold(
            key=key,
            unit=entry["unit"],
            applies_to=tuple(entry["applies_to"]),
            source=entry["source"],
            value=entry.get("value"),
            default=entry.get("default"),
            by_family=entry.get("by_family") if "by_family" in entry else None,
        )
        for key, entry in raw.items()
    }


def threshold(key: str) -> Threshold:
    """Look up one threshold, failing loudly on an unknown key."""

    try:
        return thresholds()[key]
    except KeyError as error:
        raise ThresholdError(
            f"{key!r} is not defined in {THRESHOLDS_FILE.name}; "
            f"add it there rather than hardcoding the number"
        ) from error


def value_of(key: str, family: str | None = None) -> Any:
    """The resolved numeric value of ``key`` for ``family``."""

    return threshold(key).for_family(family)


def schema_version() -> str:
    return _document()["schema_version"]
