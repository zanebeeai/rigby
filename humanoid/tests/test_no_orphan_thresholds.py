"""Every threshold must reach a production call site, and the gap must shrink.

Plan 08 §3.5, G1. A config file whose keys nothing reads is decoration that
looks authoritative -- the exact failure that let 17 keys sit in
``acceptance_criteria.yaml`` unread while appearing to be the contract.

The guard is not "every key has a reader" on day one, because that would be red
for the whole of 08b and get an allowlist. It is a **ledger**: keys not yet
repointed are recorded individually with the file that still reads their old
source, and the ledger may only ever get shorter. A new key cannot be added
without a reader, and a repointed key cannot regress.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = PROJECT_ROOT / "config" / "thresholds.v1.json"
PRODUCTION_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT / "evals")

#: Keys still served by their pre-consolidation source, each naming what reads
#: it today and the PR that repoints it. This list may only ever get SHORTER --
#: `test_the_migration_ledger_only_shrinks` asserts the count, so adding a key
#: here to silence a failure fails a different test. Dated 2026-08-24, owned by
#: lane `infra`, retired by 08c.
UNMIGRATED: dict[str, str] = {
    "anatomy.forearm_twist_max_rad": "config/motion_quality_reference.json hard_limits",
    "anatomy.wrist_swing_max_rad": "config/motion_quality_reference.json hard_limits",
    "anatomy.wrist_twist_max_rad": "config/motion_quality_reference.json hard_limits",
    "signal.angular_velocity_max_rad_s": "config/motion_quality_reference.json hard_limits",
    "signal.angular_acceleration_max_rad_s2": "config/motion_quality_reference.json hard_limits",
    "signal.angular_jerk_max_rad_s3": "config/motion_quality_reference.json hard_limits",
    "signal.min_active_hand_visibility_fraction": "config/motion_quality_reference.json hard_limits",
    "anatomy.upper_arm_length_m": "src/rigby_poc/primitives.py UPPER_ARM_LENGTH_M -- 08d",
    "anatomy.lower_arm_length_m": "src/rigby_poc/primitives.py LOWER_ARM_LENGTH_M -- 08d",
    "physics.min_lift_m": "acceptance_criteria.yaml physical_proof",
    "physics.min_hold_s": "acceptance_criteria.yaml physical_proof",
    "physics.max_vertical_drift_m": "acceptance_criteria.yaml physical_proof",
    "physics.max_palm_relative_slip_m": "acceptance_criteria.yaml physical_proof",
    "physics.max_coordinate_roundtrip_error_m": "acceptance_criteria.yaml structural_physics",
    "safety.joint_limit_violations_max": "acceptance_criteria.yaml safety",
    "safety.nan_count_max": "acceptance_criteria.yaml safety",
    "safety.unresolved_non_hand_collisions_max": "acceptance_criteria.yaml safety",
    "signal.discontinuity_count_max": "acceptance_criteria.yaml safety",
    "semantic.reach_tolerance_m": "acceptance_criteria.yaml semantic_forward_space",
    "semantic.min_forward_projection_m": "acceptance_criteria.yaml semantic_forward_space",
    "camera.max_angle_from_rig_forward_deg": "acceptance_criteria.yaml egocentric_camera_orientation",
}

#: The ledger length when it was recorded. Never raise this.
#:
#: 24 -> 22 in 10b. `physics.foot_drift_max_m` and `safety.penetration_max_m` both
#: acquired production readers in the physics layer -- `physics.contact.foot_skate`
#: and `physics.ground.penetration` respectively. The foot-drift entry is the more
#: pointed of the two: its ledger note said "the gate is vacuous until 08c measures
#: it", and it was retired by being measured rather than by 08c.
#:
#: 22 -> 21 in 08d. `anatomy.shoulder_origin_m` gains a production reader in
#: `primitives.py shoulder_position`. Counted from the real module rather than by
#: arithmetic on either branch's old constant: 10b and 08d were both written
#: against a ledger of 24 and lowered it to 22 and 23, so neither figure survives
#: both landing, and a ceiling above the real length is a ledger that cannot catch
#: the next entry.
LEDGER_HIGH_WATER_MARK = 21


def _keys() -> set[str]:
    return set(json.loads(THRESHOLDS.read_text(encoding="utf-8"))["thresholds"])


def _production_readers() -> dict[str, list[str]]:
    sources = [
        path
        for root in PRODUCTION_ROOTS
        for path in root.rglob("*.py")
        if ".venv" not in path.parts
    ]
    texts = [(path, path.read_text(encoding="utf-8")) for path in sources]
    readers: dict[str, list[str]] = {}
    for key in _keys():
        for path, text in texts:
            if f'"{key}"' in text or f"'{key}'" in text:
                readers.setdefault(key, []).append(
                    path.relative_to(PROJECT_ROOT).as_posix()
                )
    return readers


def test_no_threshold_is_read_by_nothing_at_all() -> None:
    """A key with no reader anywhere -- not even a test -- is dead on arrival."""

    readers = _production_readers()
    schema_test = (PROJECT_ROOT / "tests").rglob("*.py")
    test_text = "\n".join(path.read_text(encoding="utf-8") for path in schema_test)
    orphans = [
        key
        for key in sorted(_keys())
        if not readers.get(key)
        and f'"{key}"' not in test_text
        and f"'{key}'" not in test_text
    ]
    assert not orphans, f"thresholds nothing reads at all: {orphans}"


def test_every_threshold_reaches_production_or_is_on_the_ledger() -> None:
    """The gate. A new key must have a production reader or be repointed."""

    readers = _production_readers()
    unaccounted = sorted(
        key for key in _keys() if not readers.get(key) and key not in UNMIGRATED
    )
    assert not unaccounted, (
        "these thresholds have no production reader and are not on the migration "
        f"ledger: {unaccounted}\n"
        "Repoint the call site, or -- only if it is genuinely pre-existing work -- add "
        "it to UNMIGRATED with what reads its old source. Adding to the ledger raises "
        "its length, which fails test_the_migration_ledger_only_shrinks."
    )


def test_the_migration_ledger_only_shrinks() -> None:
    """A ledger that can grow is an allowlist, and allowlists are unmaintained."""

    assert len(UNMIGRATED) <= LEDGER_HIGH_WATER_MARK, (
        f"the migration ledger grew to {len(UNMIGRATED)} from {LEDGER_HIGH_WATER_MARK}. "
        "It records pre-existing unrepointed call sites and may only get shorter."
    )


def test_a_repointed_key_is_removed_from_the_ledger() -> None:
    """Keeping a migrated key on the ledger hides that the work is done."""

    readers = _production_readers()
    stale = sorted(key for key in UNMIGRATED if readers.get(key))
    assert not stale, (
        f"these are on the migration ledger but now have production readers: {stale}. "
        "Delete them from UNMIGRATED and lower LEDGER_HIGH_WATER_MARK."
    )


def test_every_ledger_entry_names_what_still_reads_it() -> None:
    for key, source in UNMIGRATED.items():
        assert key in _keys(), f"{key} is on the ledger but not in the config"
        assert len(source) > 20, f"{key}'s ledger entry must name its current source"


@pytest.mark.parametrize(
    "key",
    ["signal.discontinuity_rad", "physics.root_drift_max_m"],
)
def test_the_keys_08b_repointed_have_production_readers(key: str) -> None:
    """Pins 08b's own work so it cannot silently regress."""

    assert _production_readers().get(key), f"{key} lost its production reader"
