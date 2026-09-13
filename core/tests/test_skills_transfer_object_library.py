"""The transfer_object library: its shape is what G10 promises."""

from __future__ import annotations

from rigby_core.skills import NodeKind, SkillLibraryV1
from rigby_core.skills.examples import EPISODE_CAP_S, RETRY_BUDGET, transfer_object_library


def test_library_is_valid_and_expands_to_the_five_stage_tree():
    library = transfer_object_library()
    assert isinstance(library, SkillLibraryV1)
    tree = library.expand("transfer_object", {"object": "cube", "destination": "platform", "effector": "palm"})
    leaves = [node.skill_id for node in tree.root.walk() if node.kind in (NodeKind.PRIMITIVE, NodeKind.OBSERVE)]
    assert leaves == ["observe_object", "acquire", "verify_hold", "transport", "release", "verify_placement"]
    assert tree.max_depth == 6


def test_every_loop_is_bounded_at_the_retry_budget_and_the_root_carries_the_cap():
    library = transfer_object_library()
    loops = [s for s in library.skills if s.kind is NodeKind.REPEAT_UNTIL]
    assert {s.skill_id for s in loops} == {"place_until_placed", "acquire_until_held"}
    assert all(s.loop.max_attempts == RETRY_BUDGET == 3 for s in loops)
    assert library.skill("transfer_object").timeout_s == EPISODE_CAP_S == 120.0
    assert all(s.recovery.max_attempts == 1 for s in library.skills), "retries live in the bounded loops, never in per-node recovery on top of them"


def test_verifications_are_observations_the_primitives_cannot_fake():
    library = transfer_object_library()
    for name in ("verify_hold", "verify_placement", "observe_object"):
        assert library.skill(name).kind is NodeKind.OBSERVE
    # The loops ask predicates that only the observations establish.
    assert library.skill("acquire_until_held").loop.until.name == "object_held"
    assert library.skill("place_until_placed").loop.until.name == "object_placed"
    assert {ref.name for ref in library.skill("acquire").initiation} == {"object_known", "object_in_reach"}
    assert {ref.name for ref in library.skill("transport").initiation} == {"object_held"}
    for name in ("acquire", "transport", "release"):
        assert library.skill(name).termination.require_effects is False, "a primitive's own certificate settles nothing; the verifications do"
