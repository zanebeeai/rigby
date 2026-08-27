"""Every run leaves a record, including the ones that refused.

A pipeline that reports only its verdict cannot be debugged, and one that records
only its failures cannot be trusted about its successes. These tests hold both
halves: a successful run explains itself stage by stage, and a refused one says
which stage refused and why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rigby_general.bake.enumerate import build_candidate
from rigby_general.bake.runner import _attempt
from rigby_general.config import base_tree_fingerprint
from rigby_general.pipeline import ingest_robot
from rigby_general.primitives import PrimitiveRecord
from rigby_general.run import answer
from rigby_general.schema.inventory import load_inventory
from rigby_general.schema.program import Remove
from rigby_general.trace import RunTrace, TraceStore


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"

LIBRARY_SLICE = (
    ("reach_to_point", Remove.MEDIAL),
    ("reach_to_point", Remove.DISTAL),
)


@pytest.fixture(scope="module")
def inventory():
    return load_inventory()


@pytest.fixture(scope="module")
def robot():
    return ingest_robot(ZOO_ROOT / "zoo_jaw_arm" / "robot.urdf", robot_id="zoo_jaw_arm")


@pytest.fixture(scope="module")
def library(robot, inventory):
    fingerprint = base_tree_fingerprint().sha256
    records, failures = [], []
    for entry_id, remove in LIBRARY_SLICE:
        outcome = _attempt(
            build_candidate(inventory.by_id(entry_id), remove),
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
        prompt, robot.manifest, robot.finalized.model, inventory, records, failures
    )


# --------------------------------------------------------------------------
# What a successful run records
# --------------------------------------------------------------------------


def test_a_successful_run_records_every_stage(robot, inventory, library) -> None:
    trace = run(robot, inventory, library, "reach out to a far point").trace

    assert trace is not None
    names = [stage.name for stage in trace.stages]
    assert names == [
        "firewall",
        "planning",
        "binding",
        "grounding",
        "compilation",
        "certification",
    ]
    assert all(stage.status == "ok" for stage in trace.stages)


def test_the_trace_keeps_the_exact_prompt(robot, inventory, library) -> None:
    prompt = "reach out to a far point"
    trace = run(robot, inventory, library, prompt).trace
    assert trace.prompt == prompt
    assert trace.to_json()["prompt"] == prompt


def test_the_planner_stage_records_which_cue_fired(
    robot, inventory, library
) -> None:
    """The difference between "the planner got it wrong" and "*across* matched"."""

    trace = run(robot, inventory, library, "reach out to a far point").trace
    planning = next(stage for stage in trace.stages if stage.name == "planning")

    cues = planning.detail["cues"]
    assert cues and cues[0]["entry_id"] == "reach_to_point"
    assert cues[0]["remove"] == "distal"
    assert cues[0]["clause"]


def test_the_trace_carries_the_body_neutral_reading(
    robot, inventory, library
) -> None:
    trace = run(robot, inventory, library, "reach out").trace

    assert len(trace.role_normalized_hash) == 64
    segments = trace.schema_program["segments_readable"]
    assert segments[0]["vector"] == "to"
    assert segments[0]["figure"] == "primary_effector"
    assert segments[0]["remove"] == "medial"


def test_the_trace_records_what_it_grounded_against(
    robot, inventory, library
) -> None:
    """The measurements that turned ``distal`` into metres."""

    trace = run(robot, inventory, library, "reach out").trace
    against = trace.grounded["grounded_against"]

    assert against["reach_radius_m"] == robot.morphology.scale.reach_radius_m
    assert against["neutral_speed_mps"] == robot.morphology.scale.neutral_speed_mps


def test_the_trace_records_the_gates(robot, inventory, library) -> None:
    trace = run(robot, inventory, library, "reach out").trace

    assert trace.certification["certified"] is True
    assert trace.certification["repeats"] >= 3
    assert trace.certification["replay_agreement"] is True
    assert trace.certification["violations"] == []


def test_the_trace_is_content_addressed(robot, inventory, library) -> None:
    payload = run(robot, inventory, library, "reach out").trace.to_json()
    assert len(payload["content_sha256"]) == 64
    assert payload["provenance"]["base_tree"]["package"] == "rigby_core"


# --------------------------------------------------------------------------
# What a refusal records
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt, stage",
    [
        ("drive across the room and open the door", "firewall"),
        ("make me a sandwich", "planning"),
        ("trace a circle", "binding"),
    ],
)
def test_each_refusal_records_the_stage_that_refused(
    robot, inventory, library, prompt: str, stage: str
) -> None:
    trace = run(robot, inventory, library, prompt).trace

    assert trace.accepted is False
    assert trace.failure["stage"] == stage
    assert trace.failure["detail"]
    assert trace.stages[-1].status == "refused"


def test_a_refusal_after_planning_keeps_what_was_understood(
    robot, inventory, library
) -> None:
    """A binding refusal happened downstream, so the reading survives it."""

    trace = run(robot, inventory, library, "trace a circle").trace

    assert trace.schema_program is not None
    assert trace.role_normalized_hash
    assert "circular" in trace.schema_program["canonical_keys"][0]


def test_a_firewall_refusal_records_no_reading(robot, inventory, library) -> None:
    """Nothing was read, because nothing got as far as the recognizer."""

    trace = run(robot, inventory, library, "explain your reasoning").trace

    assert trace.schema_program is None
    assert trace.stages[0].name == "firewall"
    assert trace.stages[0].detail["rule"].startswith("capability.")


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


def test_traces_round_trip_through_disk(tmp_path: Path, robot, inventory, library) -> None:
    store = TraceStore(tmp_path)
    trace = run(robot, inventory, library, "reach out").trace
    store.write(trace, clip_bytes=b"GIF89a-not-really")

    assert store.list_ids() == (trace.trace_id,)
    reloaded = store.load(trace.trace_id)
    assert reloaded["prompt"] == trace.prompt
    assert reloaded["clip"] == "clip.gif"
    assert store.clip_path(trace.trace_id).read_bytes() == b"GIF89a-not-really"


def test_the_index_exposes_the_hash_for_grouping(
    tmp_path: Path, robot, inventory, library
) -> None:
    """The listing has to carry the hash, or the invariance view cannot group."""

    store = TraceStore(tmp_path)
    store.write(run(robot, inventory, library, "reach out").trace)

    row = store.index()[0]
    assert row["role_normalized_hash"]
    assert row["accepted"] is True
    assert row["robot_id"] == "zoo_jaw_arm"


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..\\windows"])
def test_the_store_refuses_a_traversal_id(tmp_path: Path, bad: str) -> None:
    store = TraceStore(tmp_path)
    with pytest.raises(ValueError, match="unsafe trace id"):
        store.directory(bad)


def test_trace_ids_are_readable(robot, inventory, library) -> None:
    """Someone will read these in a directory listing."""

    trace = run(robot, inventory, library, "reach out to a far point").trace
    assert trace.trace_id.startswith("zoo_jaw_arm--reach-out")
