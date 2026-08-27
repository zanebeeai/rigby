"""What real, third-party robot descriptions broke, and what now holds instead.

Every case here is a defect that the generated zoo could not have surfaced,
because the zoo is authored by this repository and is therefore tidy in exactly
the ways real URDFs are not. The models come from an outside project, are pinned
by commit, and are fetched rather than committed -- so each test skips when the
fetch has not been run, and none of them is allowed to be the only thing standing
between a regression and a release.

    uv run python scripts/fetch_exotic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rigby_general.contracts import EffectorKind
from rigby_general.errors import ModelIngestError, MorphologyError
from rigby_general.grounding.grounder import figure_site_for
from rigby_general.grounding.workspace import build_workspace_frame
from rigby_general.ingest.integrity import scan_source_text
from rigby_general.morphology.measure import _penetration_depth
from rigby_general.pipeline import ingest_robot


EXOTIC = Path(__file__).resolve().parents[1] / "assets" / "general" / "exotic"

MESH_ARM = "iiwa7"
FOLDED_ARM = "kuka_lwr"
JAW_ARM = "panda"
BAD_INERTIA = "xarm6"
NO_ARM = "pr2_gripper"
NOT_A_MANIPULATOR = "cartpole"


def source(robot_id: str) -> Path:
    path = EXOTIC / robot_id / "robot.urdf"
    if not path.is_file():
        pytest.skip(f"{robot_id} not fetched; run scripts/fetch_exotic.py")
    return path


@pytest.fixture(scope="module")
def arms():
    return {
        robot_id: ingest_robot(source(robot_id), robot_id=robot_id)
        for robot_id in (MESH_ARM, FOLDED_ARM, JAW_ARM)
    }


# -- ingest ----------------------------------------------------------------


def test_a_mesh_bearing_model_survives_the_second_compile(arms) -> None:
    """The staged copy has to outlive measurement.

    Ingest compiles twice: once as delivered, once after sites and actuators are
    written. The upload is staged in a scratch directory so a stray relative path
    cannot escape it, and deleting that directory after the first compile worked
    for every model built from primitives and failed on the first one with
    meshes, because the spec resolves mesh paths against where it was read.
    """

    robot = arms[MESH_ARM]
    assert robot.finalized.model.nmesh > 0
    assert robot.manifest.rest_qpos


def test_a_folded_zero_pose_is_relaxed_rather_than_used(arms) -> None:
    """Clamping zero into the joint limits is not enough to make a rest pose.

    Both of these arms self-intersect at their zero configuration -- one joint's
    range has zero at its endpoint, and the arm folds back through itself. Rest
    has to mean a pose the robot could actually hold.
    """

    for robot_id in (FOLDED_ARM, JAW_ARM):
        model = arms[robot_id].finalized.model
        rest = np.asarray(arms[robot_id].manifest.rest_qpos, dtype=float)
        raw = np.array(model.qpos0, dtype=float)

        assert not np.allclose(rest, raw), f"{robot_id} rest pose was not relaxed"
        assert _penetration_depth(model, rest) < _penetration_depth(model, raw)

    # One of the two comes out completely clear. The other cannot: its residual
    # overlap is the authored-hull pair, which no configuration parts -- that is
    # the whole reason it is recorded instead of refused. Demanding a clear rest
    # pose from it would be demanding something the model does not contain.
    panda_model = arms[JAW_ARM].finalized.model
    panda_rest = np.asarray(arms[JAW_ARM].manifest.rest_qpos, dtype=float)
    assert _penetration_depth(panda_model, panda_rest) <= 0.0


def test_hulls_that_always_overlap_are_recorded_not_refused(arms) -> None:
    """Two defects look identical at rest and are not the same problem.

    An arm slumped against its torso can rotate out of it; two wrist links whose
    convex hulls are authored intersecting cannot, at any configuration. Refusing
    both turned a real arm away for an ordinary modelling habit.
    """

    robot = arms[FOLDED_ARM]
    assert robot.integrity.notes, "the overlapping wrist hulls were not recorded"
    assert any("every configuration" in note.detail for note in robot.integrity.notes)


def test_a_recorded_overlap_is_excluded_from_the_collision_gate(arms) -> None:
    """Recording it and then gating it anyway would be worse than not noticing.

    A pair that overlaps in every configuration is in contact in every frame of
    every motion, so gating it does not detect a collision -- it rejects the
    robot. This robot's bake yield was zero until the exclusion was honoured.
    """

    robot = arms[FOLDED_ARM]
    recorded = {
        tuple(sorted(note.subject.split("|")))
        for note in robot.integrity.notes
        if "|" in note.subject
    }
    declared = {
        tuple(sorted(pair))
        for pair in robot.manifest.adjacent_collision_exclusions
    }
    assert recorded, "nothing was recorded to exclude"
    assert recorded <= declared


# -- morphology ------------------------------------------------------------


def test_a_frame_marker_between_the_jaws_is_not_a_finger(arms) -> None:
    """Real URDFs mark the tool centre point with a massless, geomless link.

    That marker sits exactly between the fingers, and it has no surface. Counting
    it as a member meant the closure test refused to run at all -- one member
    with no geometry abandoned the whole measurement -- and once that was fixed,
    counting it as a digit made a two-finger hand a three-fingered one.
    """

    morphology = arms[JAW_ARM].morphology
    effector = morphology.effectors[0]

    assert effector.kind is EffectorKind.PARALLEL_JAW
    assert effector.can_grasp
    assert effector.max_aperture_m and effector.max_aperture_m > 0.05
    assert len(effector.member_bodies) == 3, "the marker is still part of the effector"


def test_the_reachable_set_has_an_inner_surface(arms) -> None:
    """A serial arm cannot fold its tool onto its own shoulder.

    On these arms the hole in the middle is 15-25% of maximum reach. Treating the
    reachable set as a ball put the closest degrees of remove inside the machine,
    and a third of every bake came back ungroundable.
    """

    robot = arms[MESH_ARM]
    chain = robot.morphology.chains[0]
    assert chain.reach_inner_m, "no inner surface was measured"

    frame = build_workspace_frame(
        robot.finalized.model,
        robot.morphology,
        chain,
        figure_site=figure_site_for(robot.manifest, chain.chain_id),
    )
    near = frame.inner_reach(0.0, 0.0)
    far = frame.directional_reach(0.0, 0.0)
    assert 0.0 < near < far

    closest = frame.point(radius_fraction=0.0, azimuth_deg=0.0, elevation_deg=0.0)
    radius = float(np.linalg.norm(closest - frame.origin))
    assert radius == pytest.approx(near, abs=1e-6), "remove 0 is not the near surface"


def test_scene_placement_still_spans_the_maximum_reach(arms) -> None:
    """Two fraction-of-reach questions that are not the same question.

    A degree of remove spans the shell. Where to stand a block spans the arm's
    outward extent, which is what those constants were calibrated against;
    re-basing them on the shell moved every block 60 mm out and closed the jaws
    inside it.
    """

    robot = arms[MESH_ARM]
    chain = robot.morphology.chains[0]
    frame = build_workspace_frame(
        robot.finalized.model,
        robot.morphology,
        chain,
        figure_site=figure_site_for(robot.manifest, chain.chain_id),
    )
    point = frame.point_at_reach_fraction(radius_fraction=0.5, elevation_deg=-22.0)
    radius = float(np.linalg.norm(point - frame.origin))
    assert radius == pytest.approx(0.5 * frame.directional_reach(0.0, -22.0), rel=1e-6)


# -- refusals, which are results -------------------------------------------


def test_an_impossible_inertia_is_named_before_the_parser_dies(arms) -> None:
    """MuJoCo refuses this model, and its refusal reads as a parse error.

    One link's principal moments violate A + B >= C by 13%, which describes no
    rigid body. The check runs on the source text so the failure names the link
    and the size of the violation instead of surfacing as "unreadable model".
    """

    violations = scan_source_text(source(BAD_INERTIA).read_bytes())
    inertia = [v for v in violations if "inertial" in v.rule.value]
    assert inertia, "the impossible inertia was not caught"
    assert any("A + B >= C" in v.detail for v in inertia)

    with pytest.raises(ModelIngestError) as caught:
        ingest_robot(source(BAD_INERTIA), robot_id=BAD_INERTIA)
    assert caught.value.details["subject"]


@pytest.mark.parametrize(
    "robot_id, positioning_dof", [(NO_ARM, 0), (NOT_A_MANIPULATOR, 1)]
)
def test_a_mechanism_that_moves_is_not_one_that_positions(
    robot_id: str, positioning_dof: int
) -> None:
    """Placing an effector takes two independent axes; one traces a line.

    Without this the cart-and-pole ingested happily as a fixed-base arm whose
    tool tip was a freely swinging pole and whose measured reach was fifteen
    metres -- every number technically correct, the whole description
    meaningless.
    """

    with pytest.raises(MorphologyError) as caught:
        ingest_robot(source(robot_id), robot_id=robot_id)
    assert caught.value.details["positioning_dof"] == positioning_dof
    assert caught.value.details["required"] == 2
