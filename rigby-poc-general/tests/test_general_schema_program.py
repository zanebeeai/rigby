"""The schema IR must be magnitude-neutral and body-independent.

These tests are the executable form of the design claim. If any of them can be
made to fail by an ordinary-looking change, the semantic layer has started to
depend on the body and the whole approach stops working for a new robot.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rigby_v2.contracts import Contract

from rigby_general.schema.program import (
    BindingRole,
    BoundaryCondition,
    Concurrency,
    Conformation,
    Contour,
    Deixis,
    Dimensionality,
    MannerV1,
    MotionSchemaProgramV1,
    PathVector,
    ReferenceFrame,
    RegionV1,
    Remove,
    RoleBindingV1,
    SchemaKind,
    SegmentLinkV1,
    SegmentSchemaV1,
    SegmentV1,
    Stative,
    assert_metric_free,
)


def path_schema(
    *,
    vector: PathVector = PathVector.TO,
    conformation: Conformation = Conformation.SURFACE,
    deixis: Deixis = Deixis.NEUTRAL,
    contour: Contour = Contour.STRAIGHT,
) -> SegmentSchemaV1:
    return SegmentSchemaV1(
        kind=SchemaKind.PATH,
        vector=vector,
        conformation=conformation,
        deixis=deixis,
        contour=contour,
    )


def segment(
    segment_id: str = "s1",
    *,
    schema: SegmentSchemaV1 | None = None,
    figure_site: str | None = None,
    ground_site: str | None = None,
    remove: Remove = Remove.DISTAL,
    manner: MannerV1 | None = None,
    boundary: BoundaryCondition = BoundaryCondition.TERMINUS,
) -> SegmentV1:
    return SegmentV1(
        segment_id=segment_id,
        motion_schema=schema or path_schema(),
        figure=RoleBindingV1(role=BindingRole.PRIMARY_EFFECTOR, site=figure_site),
        ground=RoleBindingV1(role=BindingRole.BASE, site=ground_site),
        region=RegionV1(remove=remove, dimensionality=Dimensionality.POINT),
        frame=ReferenceFrame.ABSOLUTE,
        manner=manner or MannerV1(),
        boundary=boundary,
    )


# --------------------------------------------------------------------------
# Magnitude neutrality
# --------------------------------------------------------------------------


def test_schema_program_cannot_hold_a_metric_value() -> None:
    """A planner cannot guess a reach it has no field to write it in."""

    assert_metric_free(MotionSchemaProgramV1)


def test_metric_free_check_actually_catches_a_float() -> None:
    """Negative control: the guarantee is only worth its detector."""

    class Leaky(Contract):
        distance_m: float

    with pytest.raises(TypeError, match="magnitude-neutral"):
        assert_metric_free(Leaky)


def test_metric_free_check_descends_into_nested_contracts() -> None:
    class Inner(Contract):
        duration_s: float

    class Outer(Contract):
        inner: Inner

    with pytest.raises(TypeError, match="Inner.duration_s"):
        assert_metric_free(Outer)


def test_manner_axes_are_bounded_ordinals() -> None:
    with pytest.raises(ValidationError):
        MannerV1(speed=3)
    with pytest.raises(ValidationError):
        MannerV1(amplitude=-3)
    assert MannerV1(speed=2, effort=-2).distance(MannerV1()) == 4


def test_repetition_count_is_an_exact_cardinal() -> None:
    """Cardinality is not magnitude-neutral: "three times" is exact everywhere."""

    assert MannerV1(repetition_count=3).repetition_count == 3
    with pytest.raises(ValidationError):
        MannerV1(repetition_count=0)


# --------------------------------------------------------------------------
# Body independence
# --------------------------------------------------------------------------


def test_same_roles_hash_identically_across_different_bodies() -> None:
    """Requirement ``schema_invariance``, in miniature.

    Two robots bind the same roles to entirely different site names. The
    semantic content is identical, so the hashes must be too.
    """

    franka_like = MotionSchemaProgramV1(
        program_id="run-a",
        source_text="reach out",
        segments=(segment(figure_site="arm_tip", ground_site="robot_base"),),
    )
    aloha_like = MotionSchemaProgramV1(
        program_id="run-b",
        source_text="Reach out.",
        segments=(segment(figure_site="left_gripper_center", ground_site="torso_base"),),
    )

    assert franka_like.role_normalized_hash() == aloha_like.role_normalized_hash()


def test_a_different_region_changes_the_hash() -> None:
    """The hash must still be sensitive to real semantic differences."""

    near = MotionSchemaProgramV1(
        program_id="run-a",
        source_text="reach",
        segments=(segment(remove=Remove.PROXIMAL),),
    )
    far = MotionSchemaProgramV1(
        program_id="run-a",
        source_text="reach",
        segments=(segment(remove=Remove.DISTAL),),
    )

    assert near.role_normalized_hash() != far.role_normalized_hash()


def test_a_different_manner_changes_the_hash() -> None:
    slow = MotionSchemaProgramV1(
        program_id="run",
        source_text="reach",
        segments=(segment(manner=MannerV1(speed=-2)),),
    )
    fast = MotionSchemaProgramV1(
        program_id="run",
        source_text="reach",
        segments=(segment(manner=MannerV1(speed=2)),),
    )

    assert slow.role_normalized_hash() != fast.role_normalized_hash()


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_path_schema_requires_every_path_member() -> None:
    with pytest.raises(ValidationError):
        SegmentSchemaV1(kind=SchemaKind.PATH, vector=PathVector.TO)


def test_stative_schema_rejects_path_members() -> None:
    with pytest.raises(ValidationError):
        SegmentSchemaV1(
            kind=SchemaKind.STATIVE,
            stative=Stative.HOLD,
            contour=Contour.ARCED,
        )


def test_canonical_keys_are_stable_discrete_symbols() -> None:
    """Exact-match retrieval depends on this; no embedding model is involved."""

    assert path_schema().canonical_key == "path:to.surface.neutral.straight"
    assert (
        SegmentSchemaV1(kind=SchemaKind.STATIVE, stative=Stative.ORIENT).canonical_key
        == "stative:orient"
    )


def test_figure_cannot_be_its_own_ground() -> None:
    with pytest.raises(ValidationError, match="cannot locate itself"):
        SegmentV1(
            segment_id="s1",
            motion_schema=path_schema(),
            figure=RoleBindingV1(role=BindingRole.PRIMARY_EFFECTOR),
            ground=RoleBindingV1(role=BindingRole.PRIMARY_EFFECTOR),
            region=RegionV1(remove=Remove.DISTAL),
            frame=ReferenceFrame.ABSOLUTE,
            boundary=BoundaryCondition.TERMINUS,
        )


def test_deictic_path_requires_a_frame_with_a_viewpoint() -> None:
    """HITHER is meaningless without a front to point away from."""

    with pytest.raises(ValidationError, match="intrinsic or viewer frame"):
        SegmentV1(
            segment_id="s1",
            motion_schema=path_schema(deixis=Deixis.HITHER),
            figure=RoleBindingV1(role=BindingRole.PRIMARY_EFFECTOR),
            ground=RoleBindingV1(role=BindingRole.TARGET_OBJECT),
            region=RegionV1(remove=Remove.ADJACENT),
            frame=ReferenceFrame.ABSOLUTE,
            boundary=BoundaryCondition.TERMINUS,
        )


def test_non_object_roles_cannot_bind_a_scene_object() -> None:
    with pytest.raises(ValidationError, match="cannot bind a scene object"):
        RoleBindingV1(role=BindingRole.PRIMARY_EFFECTOR, object_id="block")


def test_multi_segment_program_must_state_a_temporal_relation() -> None:
    with pytest.raises(ValidationError, match="how its segments relate"):
        MotionSchemaProgramV1(
            program_id="run",
            source_text="reach then retract",
            segments=(segment("s1"), segment("s2")),
        )


def test_sequence_cycles_are_rejected() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        MotionSchemaProgramV1(
            program_id="run",
            source_text="loop",
            segments=(segment("s1"), segment("s2")),
            links=(
                SegmentLinkV1(
                    from_segment="s1", to_segment="s2", relation=Concurrency.SEQUENCE
                ),
                SegmentLinkV1(
                    from_segment="s2", to_segment="s1", relation=Concurrency.SEQUENCE
                ),
            ),
        )


def test_coupled_relation_requires_a_constraint() -> None:
    with pytest.raises(ValidationError, match="coupling constraint"):
        SegmentLinkV1(
            from_segment="s1",
            to_segment="s2",
            relation=Concurrency.CONCURRENT_COUPLED,
        )


def test_uncoupled_relation_rejects_a_constraint() -> None:
    from rigby_general.schema.program import Coupling

    with pytest.raises(ValidationError, match="cannot carry a coupling"):
        SegmentLinkV1(
            from_segment="s1",
            to_segment="s2",
            relation=Concurrency.SEQUENCE,
            coupling=Coupling.MIRROR,
        )


def test_segment_key_excludes_manner() -> None:
    """A primitive is baked neutral and modulated at bind time."""

    assert segment(manner=MannerV1(speed=2)).canonical_key == segment().canonical_key
