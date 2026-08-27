from __future__ import annotations

import json

from evals.criteria import supported_cases
from evals.heldout import (
    PROFILE_VALUES,
    heldout_complex_hangten_uplift_cases,
    heldout_hangten_cases,
)
from rigby_poc.models import PrimitiveKind
from rigby_poc.models import PlanRequest, default_scene
from rigby_poc.planner import OfflinePlanner

import pytest

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


def test_thirty_heldout_hangten_profiles_are_unique_and_unseen() -> None:
    cases = heldout_hangten_cases()
    assert len(cases) == 30
    profiles = {json.dumps(case["motion_profile"], sort_keys=True) for case in cases}
    prior = {
        json.dumps(case["motion_profile"], sort_keys=True)
        for case in supported_cases()[:20]
    }
    assert len(profiles) == 30
    assert profiles.isdisjoint(prior)
    assert {case["hand"] for case in cases} == {"left", "right"}
    for key, values in PROFILE_VALUES.items():
        assert set(values) <= {case["motion_profile"][key] for case in cases}


def test_heldout_prompts_roundtrip_to_the_declared_profiles() -> None:
    planner = OfflinePlanner()
    scene = default_scene()
    for case in heldout_hangten_cases():
        program = planner.plan(
            PlanRequest(text=case["prompt"], scene=scene, provider="offline")
        ).program
        assert program.hand.value == case["hand"]
        assert program.motion_profile is not None
        assert program.motion_profile.model_dump(mode="json") == case["motion_profile"]


def test_complex_uplift_fixture_preserves_profiles_and_varies_anatomical_wording() -> None:
    cases = heldout_complex_hangten_uplift_cases()
    assert len(cases) == 30
    assert len({case["prompt"] for case in cases}) == 30
    assert {case["expected_cycles"] for case in cases} == {2.0, 3.0, 4.0}
    assert {case["shake_wording"] for case in cases} == {
        "natural_shake",
        "palm_rotation",
        "anatomical",
    }
    planner = OfflinePlanner()
    scene = default_scene()
    for case in cases:
        program = planner.plan(
            PlanRequest(text=case["prompt"], scene=scene, provider="offline")
        ).program
        assert program.hand.value == case["hand"]
        assert program.motion_profile is not None
        assert program.motion_profile.model_dump(mode="json") == case["motion_profile"]
        assert [primitive.kind for primitive in program.primitives] == [
            PrimitiveKind.PRESENT,
            PrimitiveKind.HOLD,
            PrimitiveKind.SHAKE,
            PrimitiveKind.RECOVER,
        ]
        shake = program.primitives[2]
        assert shake.parameters.wrist_shake_cycles == case["expected_cycles"]
        assert shake.parameters.index_curl == 1.0
        assert shake.parameters.middle_curl == 1.0
        assert shake.parameters.ring_curl == 1.0
