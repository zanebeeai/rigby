"""Prompt to certified motion, and the four different ways of saying no.

Builds its own small library rather than reading the baked one, so the suite
stays self-contained and does not silently pass because someone happened to bake
recently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rigby_general.bake.enumerate import build_candidate
from rigby_general.bake.runner import _attempt
from rigby_general.config import base_tree_fingerprint
from rigby_general.pipeline import ingest_robot
from rigby_general.primitives import PrimitiveLibrary, PrimitiveRecord, retrieve
from rigby_general.run import answer, unsupported_reason
from rigby_general.schema.inventory import load_inventory
from rigby_general.schema.program import Remove


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"

# Enough to answer a couple of prompts without baking a whole robot. Note that
# "as far as you can" is *not* covered here on purpose: that phrase names
# ``reach_to_edge``, a different schema from a long ``reach_to_point``, and one of
# the tests below relies on the distinction.
LIBRARY_SLICE = (
    ("reach_to_point", Remove.MEDIAL),
    ("reach_to_point", Remove.DISTAL),
    ("retract_from_point", Remove.MEDIAL),
    ("hold_still", Remove.MEDIAL),
)


@pytest.fixture(scope="module")
def inventory():
    return load_inventory()


@pytest.fixture(scope="module")
def robot():
    return ingest_robot(ZOO_ROOT / "zoo_jaw_arm" / "robot.urdf", robot_id="zoo_jaw_arm")


@pytest.fixture(scope="module")
def library(robot, inventory):
    """Bake a handful of bindings so the run path has something to bind to."""

    fingerprint = base_tree_fingerprint().sha256
    records, failures = [], []
    for entry_id, remove in LIBRARY_SLICE:
        candidate = build_candidate(inventory.by_id(entry_id), remove)
        outcome = _attempt(
            candidate,
            robot.manifest,
            robot.finalized.model,
            inventory,
            policy=None,
            fingerprint=fingerprint,
        )
        (records if isinstance(outcome, PrimitiveRecord) else failures).append(outcome)
    return tuple(records), tuple(failures)


def run(robot, inventory, library, prompt: str):
    records, failures = library
    return answer(
        prompt,
        robot.manifest,
        robot.finalized.model,
        inventory,
        records,
        failures,
    )


def test_the_slice_certified_something(library) -> None:
    records, _ = library
    assert records, "no primitive certified; the rest of this file proves nothing"


def test_a_plain_request_produces_a_certified_motion(
    robot, inventory, library
) -> None:
    result = run(robot, inventory, library, "reach out to a far point")

    assert result.accepted
    assert result.duration_s > 0.0
    assert result.certification is not None
    assert result.certification.certified
    assert not result.certification.violations


def test_the_motion_is_replay_identical(robot, inventory, library) -> None:
    """Three simulations of the same trajectory must agree exactly."""

    result = run(robot, inventory, library, "reach out to a far point")
    assert len(set(result.certification.replay_hashes)) == 1


def test_the_summary_records_the_body_neutral_reading(
    robot, inventory, library
) -> None:
    summary = run(robot, inventory, library, "reach out").summary()
    assert summary["accepted"]
    assert len(summary["role_normalized_hash"]) == 64
    assert summary["schema_keys"]


def test_a_two_step_request_runs_both_steps(robot, inventory, library) -> None:
    result = run(robot, inventory, library, "reach out and then come back")
    assert result.accepted
    assert len(result.schema_program.segments) == 2
    assert result.bound.used_certified_primitives == 2


def test_pace_words_change_the_motion(robot, inventory, library) -> None:
    quick = run(robot, inventory, library, "reach out quickly")
    slow = run(robot, inventory, library, "reach out slowly")
    assert quick.accepted and slow.accepted
    assert quick.duration_s < slow.duration_s


# --------------------------------------------------------------------------
# The four refusals
# --------------------------------------------------------------------------


def test_unrecognised_wording_refuses_at_the_planner(
    robot, inventory, library
) -> None:
    result = run(robot, inventory, library, "make me a sandwich")
    assert not result.accepted
    assert result.failure_stage == "planning"


def test_an_unbaked_schema_refuses_at_the_binder(robot, inventory, library) -> None:
    """The library here holds four primitives; a circle is not among them."""

    records, failures = library
    result = answer(
        "trace a circle",
        robot.manifest,
        robot.finalized.model,
        inventory,
        records,
        failures,
    )
    assert not result.accepted
    assert result.failure_stage == "binding"


def test_each_refusal_says_something_different(robot, inventory, library) -> None:
    """Collapsing them into one "unsupported" would throw the useful part away."""

    unknown = run(robot, inventory, library, "solve the halting problem")
    unbaked = run(robot, inventory, library, "trace a circle")

    assert unsupported_reason(unknown) != unsupported_reason(unbaked)
    assert unknown.failure_stage != unbaked.failure_stage


def test_a_refusal_still_reports_what_it_understood(
    robot, inventory, library
) -> None:
    """A binding refusal happened *after* planning, so the reading survives."""

    result = run(robot, inventory, library, "trace a circle")
    assert result.schema_program is not None
    assert "circular" in result.schema_program.canonical_keys[0]


# --------------------------------------------------------------------------
# The library itself
# --------------------------------------------------------------------------


def test_retrieval_is_exact_match(library) -> None:
    """A schema binding is a symbol; looking one up needs no model."""

    records, _ = library
    assert retrieve(records, records[0].segment_key) is records[0]
    assert retrieve(records, "path:nonsense|a->b|distal.point|absolute") is None


def test_a_library_round_trips_through_disk(tmp_path: Path, library) -> None:
    from rigby_general.primitives.library import BakeSummary

    records, failures = library
    store = PrimitiveLibrary(tmp_path)
    summary = BakeSummary(
        robot_id="zoo_jaw_arm",
        attempted=len(records) + len(failures),
        certified=len(records),
        elapsed_seconds=1.0,
        budget_seconds=1200,
        max_attempts=500,
        complete=True,
        inventory_sha256="0" * 64,
        base_tree_sha256="0" * 64,
        afforded_entry_ids=("reach_to_point",),
        covered_entry_ids=("reach_to_point",),
    )
    store.write("zoo_jaw_arm", list(records), list(failures), summary)

    reloaded = store.load("zoo_jaw_arm")
    assert len(reloaded) == len(records)
    assert reloaded[0].segment_key == records[0].segment_key
    assert store.load_summary("zoo_jaw_arm").certified == len(records)
