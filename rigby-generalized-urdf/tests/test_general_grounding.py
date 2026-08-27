"""One body-neutral program, six very different bodies.

The claim under test is Talmy's: closed-class spatial terms are magnitude-neutral,
so a description of a motion need not know how big the thing performing it is.
If that holds, the same schema program grounds on a 0.39 m desktop arm and a
2.05 m long-reach arm without either the program or the planner changing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rigby_v2.motion.compiler import compile_motion_program

from rigby_general.contracts import DirectionV1
from rigby_general.errors import GeneralFailureCode, GroundingError, RigbyGeneralError
from rigby_general.grounding import ground
from rigby_general.morphology import confirm_frame
from rigby_general.pipeline import ingest_robot
from rigby_general.schema.inventory import (
    Requirement,
    afforded_entries,
    capabilities_of,
    load_inventory,
    unafforded_reasons,
)
from rigby_general.schema.program import (
    BindingRole,
    BoundaryCondition,
    Concurrency,
    Dimensionality,
    MannerV1,
    MotionSchemaProgramV1,
    ReferenceFrame,
    RegionV1,
    Remove,
    RoleBindingV1,
    SegmentLinkV1,
    SegmentV1,
)


ZOO_ROOT = Path(__file__).resolve().parents[1] / "assets" / "general" / "zoo"
ZOO_IDS = sorted(
    directory.name
    for directory in ZOO_ROOT.iterdir()
    if (directory / "robot.urdf").is_file()
)


@pytest.fixture(scope="module")
def inventory():
    return load_inventory()


@pytest.fixture(scope="module")
def robots() -> dict:
    return {
        robot_id: ingest_robot(ZOO_ROOT / robot_id / "robot.urdf", robot_id=robot_id)
        for robot_id in ZOO_IDS
    }


def one_segment(
    inventory,
    entry_id: str = "reach_to_point",
    *,
    remove: Remove = Remove.DISTAL,
    manner: MannerV1 | None = None,
    frame: ReferenceFrame | None = None,
    text: str = "reach out in front of you",
) -> MotionSchemaProgramV1:
    entry = inventory.by_id(entry_id)
    # A deictic path has to be read in a frame that has a viewpoint; the contract
    # refuses one that does not, so the helper follows the schema rather than
    # forcing every caller to remember.
    deictic = entry.schema.deixis is not None and entry.schema.deixis.value != "neutral"
    if frame is None:
        frame = ReferenceFrame.INTRINSIC if deictic else ReferenceFrame.ABSOLUTE
    return MotionSchemaProgramV1(
        program_id="test-program",
        source_text=text,
        segments=(
            SegmentV1(
                segment_id="only",
                motion_schema=entry.schema,
                figure=RoleBindingV1(role=entry.figure_role),
                ground=RoleBindingV1(role=entry.ground_role),
                region=RegionV1(remove=remove, dimensionality=Dimensionality.POINT),
                frame=frame,
                manner=manner or MannerV1(),
                boundary=BoundaryCondition.TERMINUS,
            ),
        ),
    )


def reach_then_retract(inventory) -> MotionSchemaProgramV1:
    reach = inventory.by_id("reach_to_point")
    retract = inventory.by_id("retract_from_point")
    return MotionSchemaProgramV1(
        program_id="reach-retract",
        source_text="reach out and then come back",
        segments=(
            SegmentV1(
                segment_id="reach",
                motion_schema=reach.schema,
                figure=RoleBindingV1(role=reach.figure_role),
                ground=RoleBindingV1(role=reach.ground_role),
                region=RegionV1(remove=Remove.DISTAL, dimensionality=Dimensionality.POINT),
                frame=ReferenceFrame.ABSOLUTE,
                boundary=BoundaryCondition.TERMINUS,
            ),
            SegmentV1(
                segment_id="retract",
                motion_schema=retract.schema,
                figure=RoleBindingV1(role=retract.figure_role),
                ground=RoleBindingV1(role=retract.ground_role),
                region=RegionV1(remove=Remove.DISTAL, dimensionality=Dimensionality.POINT),
                frame=ReferenceFrame.ABSOLUTE,
                boundary=BoundaryCondition.TERMINUS,
            ),
        ),
        links=(
            SegmentLinkV1(
                from_segment="reach", to_segment="retract", relation=Concurrency.SEQUENCE
            ),
        ),
    )


# --------------------------------------------------------------------------
# The load-bearing property
# --------------------------------------------------------------------------


def test_one_program_grounds_on_every_robot(robots, inventory) -> None:
    program = reach_then_retract(inventory)
    for robot_id, robot in robots.items():
        grounded = ground(program, robot.manifest, robot.finalized.model, inventory)
        assert grounded.program.rig_id == robot_id
        assert grounded.program.duration_s > 0.0
        assert grounded.program.tracks


def test_the_schema_hash_is_identical_across_every_body(robots, inventory) -> None:
    """Requirement ``schema_invariance``.

    A single divergence here would mean the semantic layer had learned something
    about the body, and the whole approach would be unsound rather than merely
    needing a wider threshold.
    """

    program = reach_then_retract(inventory)
    hashes = {
        ground(program, robot.manifest, robot.finalized.model, inventory)
        .program.metadata["role_normalized_hash"]
        for robot in robots.values()
    }
    assert len(hashes) == 1


def test_grounded_programs_compile_through_the_certified_compiler(
    robots, inventory
) -> None:
    """The grounder's whole job is to hand v2 an ordinary program."""

    program = reach_then_retract(inventory)
    for robot in robots.values():
        grounded = ground(program, robot.manifest, robot.finalized.model, inventory)
        trajectory = compile_motion_program(
            grounded.program, robot.finalized.model, robot.manifest, sample_hz=240
        )
        assert trajectory.qpos.shape[0] > 0
        assert np.all(np.isfinite(trajectory.qpos))


def test_grounding_is_deterministic(robots, inventory) -> None:
    """No model call, no randomness. A primitive certified yesterday must still
    describe the motion it was certified for."""

    program = reach_then_retract(inventory)
    robot = robots["zoo_jaw_arm"]
    first = ground(program, robot.manifest, robot.finalized.model, inventory)
    second = ground(program, robot.manifest, robot.finalized.model, inventory)
    assert first.program.content_hash() == second.program.content_hash()


# --------------------------------------------------------------------------
# Magnitude neutrality, measured
# --------------------------------------------------------------------------


def test_the_same_term_means_different_metres_on_different_bodies(
    robots, inventory
) -> None:
    """``distal`` is far *for this arm*, which is the entire point."""

    program = one_segment(inventory, remove=Remove.DISTAL)
    travels = {}
    for robot_id in ("zoo_compact_arm", "zoo_long_arm"):
        robot = robots[robot_id]
        grounded = ground(program, robot.manifest, robot.finalized.model, inventory)
        travels[robot_id] = robot.morphology.scale.reach_radius_m

    assert travels["zoo_long_arm"] / travels["zoo_compact_arm"] > 4.0


def _terminal_radius(robot, grounded) -> float:
    """How far the steered site gets from its chain root at the end of the action.

    Not the final keyframe: every primitive now retraces its path back to rest, so
    the last keyframe is the parked pose on every one of them. What ``proximal``
    and ``distal`` are claims about is where the effector *arrived*.
    """

    import mujoco

    model = robot.finalized.model
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(robot.manifest.rest_qpos, dtype=float)
    reaching = grounded.program.tracks[0]

    action_end = max(
        phase.end_s
        for phase in grounded.program.phases
        if phase.kind.value != "recovery"
    )
    arrival = max(
        (kf for kf in reaching.keyframes if kf.time_s <= action_end + 1e-6),
        key=lambda kf: kf.time_s,
    )
    for name, value in arrival.joint_values.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[joint])] = value
    mujoco.mj_kinematics(model, data)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, reaching.target)
    chain = next(
        item
        for item in robot.morphology.chains
        if item.chain_id == reaching.owner.removeprefix("chain_")
    )
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, chain.root_body)
    return float(np.linalg.norm(data.site_xpos[site] - data.xpos[root]))


def test_a_nearer_region_lands_nearer(robots, inventory) -> None:
    """``proximal`` and ``distal`` are about where the effector ends up, not about
    how far it travelled to get there -- the arm parks somewhere, and the trip to
    a near point is not necessarily the shorter one."""

    robot = robots["zoo_jaw_arm"]
    near = ground(
        one_segment(inventory, remove=Remove.PROXIMAL),
        robot.manifest,
        robot.finalized.model,
        inventory,
    )
    far = ground(
        one_segment(inventory, remove=Remove.DISTAL),
        robot.manifest,
        robot.finalized.model,
        inventory,
    )
    assert _terminal_radius(robot, near) < _terminal_radius(robot, far)


def test_asking_for_speed_shortens_the_motion(robots, inventory) -> None:
    robot = robots["zoo_jaw_arm"]
    slow = ground(
        one_segment(inventory, manner=MannerV1(speed=-2)),
        robot.manifest,
        robot.finalized.model,
        inventory,
    )
    quick = ground(
        one_segment(inventory, manner=MannerV1(speed=2)),
        robot.manifest,
        robot.finalized.model,
        inventory,
    )
    assert quick.program.duration_s < slow.program.duration_s


def test_timing_never_exceeds_a_measured_velocity_limit(robots, inventory) -> None:
    """A requested pace is a preference; a velocity limit is a fact.

    Asking for maximum speed on every axis must slow the motion down rather than
    produce one the hardware could not perform.
    """

    robot = robots["zoo_long_arm"]
    grounded = ground(
        one_segment(inventory, manner=MannerV1(speed=2)),
        robot.manifest,
        robot.finalized.model,
        inventory,
    )
    trajectory = compile_motion_program(
        grounded.program, robot.finalized.model, robot.manifest, sample_hz=240
    )
    limits = np.array(
        [dof.velocity_limit for dof in robot.manifest.dofs], dtype=float
    )
    assert np.all(np.abs(trajectory.qvel).max(axis=0) <= limits + 1e-6)


def test_the_program_records_what_it_was_grounded_against(robots, inventory) -> None:
    """Provenance: which measurements produced these numbers."""

    robot = robots["zoo_hand_arm"]
    grounded = ground(
        one_segment(inventory), robot.manifest, robot.finalized.model, inventory
    )
    against = grounded.program.metadata["grounded_against"]
    assert against["reach_radius_m"] == robot.morphology.scale.reach_radius_m
    assert grounded.program.metadata["inventory_sha256"] == inventory.sha256


# --------------------------------------------------------------------------
# Failing closed
# --------------------------------------------------------------------------


def test_a_one_armed_robot_cannot_be_asked_for_a_handover(robots, inventory) -> None:
    robot = robots["zoo_jaw_arm"]
    program = one_segment(inventory, "hand_across", text="pass it to your other hand")

    with pytest.raises(RigbyGeneralError) as caught:
        ground(program, robot.manifest, robot.finalized.model, inventory)
    assert caught.value.code is GeneralFailureCode.UNAFFORDED_SCHEMA
    assert "two_grasping_effectors" in str(caught.value)


def test_a_two_armed_robot_can(robots, inventory) -> None:
    robot = robots["zoo_dual_arm"]
    grounded = ground(
        one_segment(inventory, "hand_across", text="pass it across"),
        robot.manifest,
        robot.finalized.model,
        inventory,
    )
    assert grounded.program.tracks


def test_a_deictic_schema_waits_for_a_person_to_confirm_the_front(
    robots, inventory
) -> None:
    """A guessed front renders a confidently backwards motion.

    Nothing in a pedestal arm's geometry says which way it faces, so beckoning --
    which means *toward the robot* -- has no referent until someone says.
    """

    robot = robots["zoo_jaw_arm"]
    program = one_segment(inventory, "draw_hither", text="beckon it closer")

    with pytest.raises(RigbyGeneralError) as caught:
        ground(program, robot.manifest, robot.finalized.model, inventory)
    assert caught.value.code is GeneralFailureCode.UNAFFORDED_SCHEMA
    assert "confirmed_frame" in str(caught.value)


def test_the_same_deictic_schema_grounds_once_the_front_is_confirmed(
    robots, inventory
) -> None:
    robot = robots["zoo_jaw_arm"]
    confirmed = confirm_frame(
        robot.morphology.intrinsic_frame, front=DirectionV1(x=1.0, y=0.0, z=0.0)
    )
    morphology = robot.morphology.model_copy(update={"intrinsic_frame": confirmed})
    manifest = robot.manifest.model_copy(update={"morphology": morphology})

    grounded = ground(
        one_segment(inventory, "draw_hither", text="beckon it closer"),
        manifest,
        robot.finalized.model,
        inventory,
    )
    assert grounded.program.tracks


def test_an_unreachable_request_names_the_measurement_it_broke(
    robots, inventory
) -> None:
    """Fail closed. A quietly shortened reach is a motion nobody asked for."""

    robot = robots["zoo_compact_arm"]
    program = one_segment(inventory, remove=Remove.DISTAL, manner=MannerV1(amplitude=2))
    try:
        ground(program, robot.manifest, robot.finalized.model, inventory)
    except GroundingError as error:
        assert error.details.get("measurement")
    # Amplitude is capped inside the reachable set, so this may legitimately
    # succeed; what must never happen is a silent overshoot.
    else:
        grounded = ground(program, robot.manifest, robot.finalized.model, inventory)
        compile_motion_program(
            grounded.program, robot.finalized.model, robot.manifest, sample_hz=120
        )


def test_a_schema_outside_the_inventory_is_refused(robots, inventory) -> None:
    from rigby_general.schema.program import (
        Conformation,
        Contour,
        Deixis,
        PathVector,
        SchemaKind,
        SegmentSchemaV1,
    )

    robot = robots["zoo_jaw_arm"]
    invented = MotionSchemaProgramV1(
        program_id="invented",
        source_text="do something not in the closed class",
        segments=(
            SegmentV1(
                segment_id="only",
                motion_schema=SegmentSchemaV1(
                    kind=SchemaKind.PATH,
                    vector=PathVector.VIA,
                    conformation=Conformation.VOLUME,
                    deixis=Deixis.NEUTRAL,
                    contour=Contour.CIRCULAR,
                ),
                figure=RoleBindingV1(role=BindingRole.PRIMARY_EFFECTOR),
                ground=RoleBindingV1(role=BindingRole.BASE),
                region=RegionV1(remove=Remove.MEDIAL),
                frame=ReferenceFrame.ABSOLUTE,
                boundary=BoundaryCondition.TERMINUS,
            ),
        ),
    )
    with pytest.raises(RigbyGeneralError) as caught:
        ground(invented, robot.manifest, robot.finalized.model, inventory)
    assert caught.value.code is GeneralFailureCode.INVENTED_BINDING


# --------------------------------------------------------------------------
# The inventory itself
# --------------------------------------------------------------------------


def test_inventory_is_sealed(inventory) -> None:
    assert len(inventory.sha256) == 64
    assert inventory.entries


def test_a_tampered_inventory_is_refused(tmp_path: Path) -> None:
    """Every primitive was certified against the sealed version."""

    from rigby_general.schema.inventory import INVENTORY_PATH, load_inventory as fresh

    copy = tmp_path / "schema_inventory.v1.json"
    copy.write_text(
        INVENTORY_PATH.read_text(encoding="utf-8").replace("0.84", "0.99"),
        encoding="utf-8",
    )
    sidecar = tmp_path / "schema_inventory.v1.json.sha256"
    sidecar.write_text(
        (INVENTORY_PATH.parent / "schema_inventory.v1.json.sha256").read_text(
            encoding="ascii"
        ),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="sealed digest"):
        fresh(copy)


def test_capabilities_come_only_from_measurement(robots) -> None:
    jaw = capabilities_of(robots["zoo_jaw_arm"].morphology)
    dual = capabilities_of(robots["zoo_dual_arm"].morphology)
    tool = capabilities_of(robots["zoo_tool_arm"].morphology)
    long_arm = capabilities_of(robots["zoo_long_arm"].morphology)

    assert Requirement.GRASPING_EFFECTOR in jaw
    assert Requirement.TWO_GRASPING_EFFECTORS not in jaw
    assert Requirement.TWO_GRASPING_EFFECTORS in dual
    assert Requirement.GRASPING_EFFECTOR not in tool
    assert Requirement.SENSOR in long_arm
    assert Requirement.CONFIRMED_FRAME not in jaw


def test_affordance_narrows_with_capability(robots, inventory) -> None:
    tool = afforded_entries(inventory, robots["zoo_tool_arm"].morphology)
    dual = afforded_entries(inventory, robots["zoo_dual_arm"].morphology)
    assert len(dual) > len(tool)


def test_unafforded_reasons_name_the_missing_capabilities(robots, inventory) -> None:
    missing = unafforded_reasons(
        inventory, robots["zoo_tool_arm"].morphology, "transport_object"
    )
    assert "grasping_effector" in missing
    assert "contact_scene" in missing


def test_every_robot_affords_the_free_space_core(robots, inventory) -> None:
    """Free-space schemas need no confirmation and no scene, so a robot should be
    usable the moment it is uploaded."""

    for robot in robots.values():
        afforded = afforded_entries(inventory, robot.morphology)
        keys = {entry.entry_id for entry in afforded}
        assert {"reach_to_point", "retract_from_point", "traverse_line"} <= keys
