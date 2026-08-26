"""Contact probes leave the same kind of record a prompt run does.

The gap these close: the grasp path does not go through ``answer``, so for a
while it produced no trace at all and the studio simply did not show it. A
failure mode that is invisible in the viewer is one nobody will notice is
failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rigby_general.contact import PROBE_PROMPT, probe_grasp
from rigby_general.pipeline import ingest_robot
from rigby_general.trace import TraceStore


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"

CERTIFIED_GRASPER = "zoo_long_arm"
FAILING_GRASPER = "zoo_compact_arm"
NO_GRIPPER = "zoo_tool_arm"


@pytest.fixture(scope="module")
def robots():
    return {
        robot_id: ingest_robot(ZOO_ROOT / robot_id / "robot.urdf", robot_id=robot_id)
        for robot_id in (CERTIFIED_GRASPER, FAILING_GRASPER, NO_GRIPPER)
    }


def probe(robots, robot_id: str):
    robot = robots[robot_id]
    return probe_grasp(robot.manifest, robot.mjcf_xml, robot.finalized.model)


def test_a_probe_is_labelled_as_one(robots) -> None:
    """Not dressed up as a prompt run: it deliberately bypasses the planner."""

    trace = probe(robots, CERTIFIED_GRASPER).trace
    assert trace.kind == "contact_probe"
    assert trace.prompt == PROBE_PROMPT
    assert trace.schema_program is None


def test_a_held_grasp_records_every_stage(robots) -> None:
    outcome = probe(robots, CERTIFIED_GRASPER)

    assert outcome.trace.accepted
    assert [stage.name for stage in outcome.trace.stages] == [
        "effector",
        "scene",
        "closure",
        "grasp_gates",
    ]
    assert all(stage.status == "ok" for stage in outcome.trace.stages)


def test_the_scene_stage_records_what_it_was_sized_from(robots) -> None:
    """A scene in absolute metres is the same mistake as a motion in them."""

    trace = probe(robots, CERTIFIED_GRASPER).trace
    scene = next(stage for stage in trace.stages if stage.name == "scene")

    assert scene.detail["sized_from"] == "measured gripper aperture"
    assert scene.detail["placed_from"] == "measured reach envelope"
    assert scene.detail["block_size_m"] > 0.0


def test_the_grasp_record_carries_its_measurements(robots) -> None:
    grasp = probe(robots, CERTIFIED_GRASPER).trace.grasp

    assert grasp["certified"] is True
    assert grasp["opposition_achieved"] is True
    assert grasp["lift_height_m"] > grasp["required_lift_m"]
    assert grasp["equality_constraints"] == 0, "a weld would make any grasp look perfect"
    assert grasp["violations"] == []


def test_a_failed_grasp_names_the_claim_that_failed(robots) -> None:
    """The whole point of recording failures: which claim, not just "failed"."""

    trace = probe(robots, FAILING_GRASPER).trace

    assert trace.accepted is False
    assert trace.failure["stage"] == "grasp_gates"
    assert trace.failure["code"] in {
        "object_not_lifted",
        "object_dropped",
        "excessive_penetration",
        "grasp_not_achieved",
    }
    assert trace.grasp["violations"]


def test_a_robot_with_no_gripper_refuses_at_the_effector_stage(robots) -> None:
    outcome = probe(robots, NO_GRIPPER)

    assert outcome.result is None
    assert outcome.scene is None
    assert outcome.trace.failure["code"] == "no_gripper"
    assert outcome.trace.stages[0].name == "effector"


def test_probes_and_prompt_runs_share_one_store(tmp_path: Path, robots) -> None:
    """Splitting the record in two is how failed grasps stop being visible."""

    store = TraceStore(tmp_path)
    store.write(probe(robots, CERTIFIED_GRASPER).trace)
    store.write(probe(robots, FAILING_GRASPER).trace)

    kinds = {row["kind"] for row in store.index()}
    assert kinds == {"contact_probe"}
    assert len(store.index()) == 2


def test_a_rebuild_that_skips_rendering_keeps_the_clip(tmp_path: Path, robots) -> None:
    """Writing a trace without new frames must not orphan the recording.

    ``build_studio.py --no-render`` rewrites every trace with ``clip_bytes=None``.
    That used to leave ``clip`` unset while ``clip.gif`` sat right there beside
    the trace, so the views that fall back to a rendered clip -- contact probes
    above all -- went blank after a rebuild that changed nothing about them.
    """

    store = TraceStore(tmp_path)
    trace = probe(robots, CERTIFIED_GRASPER).trace

    store.write(trace, clip_bytes=b"GIF89a-pretend")
    assert trace.clip == "clip.gif"

    reloaded = probe(robots, CERTIFIED_GRASPER).trace
    assert reloaded.clip is None, "a fresh trace starts with no clip"
    store.write(reloaded)

    assert reloaded.clip == "clip.gif"
    assert store.load(reloaded.trace_id)["clip"] == "clip.gif"
    assert (tmp_path / reloaded.trace_id / "clip.gif").read_bytes() == b"GIF89a-pretend"
