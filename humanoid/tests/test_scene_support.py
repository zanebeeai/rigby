"""The support surface is a scene object, and never the thing being manipulated."""

from __future__ import annotations

import pytest
from rigby_poc.models import (
    AffordanceRole,
    AffordanceSocket,
    Intent,
    PlanRequest,
    SceneObject,
    Transform,
    Vec3,
    default_scene,
)
from rigby_poc.planner import OfflinePlanner, OpenAIPlanner, plan_motion

pytestmark = pytest.mark.fast


def test_default_scene_carries_exactly_one_table_appended_last() -> None:
    scene = default_scene()
    tables = [item for item in scene.objects if item.kind == "table"]
    assert len(tables) == 1
    assert scene.objects[-1] is tables[0]
    assert scene.support_surface() is tables[0]
    # Several consumers fall back to objects[0] as the manipulated object.
    assert scene.objects[0].kind == "block"


def test_block_rests_on_the_table_top() -> None:
    scene = default_scene()
    block = scene.object_by_id("block")
    table = scene.support_surface()
    assert block is not None and table is not None
    underside = block.transform.translation.y - block.dimensions_m.y / 2.0
    top = table.transform.translation.y + table.dimensions_m.y / 2.0
    assert underside == pytest.approx(top, abs=1e-9)
    # The block's footprint sits inside the table's.
    for axis in ("x", "z"):
        offset = abs(getattr(block.transform.translation, axis) - getattr(table.transform.translation, axis))
        assert offset + getattr(block.dimensions_m, axis) / 2.0 < getattr(table.dimensions_m, axis) / 2.0


def test_table_carries_a_support_socket_and_no_grasp_socket() -> None:
    table = default_scene().support_surface()
    assert table is not None
    roles = {socket.role for socket in table.sockets}
    assert AffordanceRole.SUPPORT in roles
    assert AffordanceRole.GRASP not in roles
    assert [item.id for item in default_scene().graspable_objects()] == ["block"]


def test_a_table_without_a_support_socket_is_rejected() -> None:
    with pytest.raises(ValueError, match="support socket"):
        SceneObject(
            id="table",
            kind="table",
            transform=Transform(translation=Vec3(x=0.0, y=0.98, z=0.5)),
            dimensions_m=Vec3(x=1.0, y=0.05, z=0.7),
            sockets=[
                AffordanceSocket(
                    id="edge",
                    transform=Transform(translation=Vec3(x=0.0, y=0.0, z=0.0)),
                    approach_normal=Vec3(x=0.0, y=0.0, z=-1.0),
                    grasp_span_m=0.05,
                )
            ],
        )


def test_the_table_is_never_resolved_as_the_manipulated_object() -> None:
    scene = default_scene()
    assert OfflinePlanner._resolve_object("grab the table", scene) is None
    assert OfflinePlanner._resolve_object("place the block on the table", scene) == "block"
    assert OfflinePlanner._resolve_object("throw the box onto the table", scene) == "block"


def test_grabbing_the_table_is_unsupported_offline() -> None:
    outcome = plan_motion(PlanRequest(text="grab the table", scene=default_scene(), provider="offline"))
    assert outcome.program.intent == Intent.UNSUPPORTED


def test_a_program_naming_the_table_as_its_object_is_invalid() -> None:
    scene = default_scene()
    grab = plan_motion(PlanRequest(text="grab the block", scene=scene, provider="offline")).program
    assert grab.intent == Intent.GRAB
    renamed = grab.model_copy(
        update={
            "primitives": [
                primitive.model_copy(update={"object_id": "table"}) if primitive.object_id else primitive
                for primitive in grab.primitives
            ]
        }
    )
    with pytest.raises(ValueError, match="support surface"):
        OpenAIPlanner._validate_semantics(renamed, scene)
