from __future__ import annotations

import pytest

from rigby_poc.compiler import compile_motion
from rigby_poc.models import CompileRequest, Intent, PlanRequest, default_scene
from rigby_poc.planner import plan_motion

#: compiles, corpus, pipeline or subprocess -- see docs/testing.md
pytestmark = pytest.mark.medium


@pytest.mark.parametrize(
    ("family", "prompt", "intent"),
    (
        ("finger gesture", "throw up a hang-ten sign with your right hand", Intent.GESTURE),
        ("strike", "throw a right jab", Intent.STRIKE),
        ("bilateral arms", "reach forward with both hands", Intent.COMPOSITE),
        ("locomotion", "walk forward three steps", Intent.FULL_BODY),
        ("concurrent body and arms", "turn left while raising both hands", Intent.FULL_BODY),
        ("exercise", "do three squats", Intent.FULL_BODY),
        ("grounded support", "hold a plank for four seconds", Intent.FULL_BODY),
        ("grounded locomotion", "crawl forward two steps", Intent.FULL_BODY),
        ("body rotation", "do a forward roll", Intent.FULL_BODY),
        ("airborne rotation", "do a front flip", Intent.FULL_BODY),
        ("choreography", "dance for six beats", Intent.FULL_BODY),
        ("climbing", "climb up the ladder for three rungs", Intent.FULL_BODY),
        ("obstacle", "step over the hurdle with your right foot", Intent.FULL_BODY),
        ("pickup", "grab the block with your right hand", Intent.GRAB),
        ("guided object", "push the block away from you", Intent.OBJECT_INTERACTION),
        ("rolling object", "roll the block forward", Intent.OBJECT_INTERACTION),
        ("spinning object", "spin the block clockwise", Intent.OBJECT_INTERACTION),
        ("placing object", "place the block to the left on the table", Intent.OBJECT_INTERACTION),
        (
            "stateful sequence",
            "grab the block then carry it forward two steps then drop it",
            Intent.SEQUENCE,
        ),
    ),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_representative_prompt_family_compiles(
    family: str,
    prompt: str,
    intent: Intent,
) -> None:
    scene = default_scene()
    program = plan_motion(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    ).program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )

    assert program.intent is intent, family
    assert program.unsupported_reason is None, family
    assert clip.success, f"{family}: {clip.failure}"
    assert clip.metrics["structural_valid"] is True, family
    assert clip.metrics["discontinuities"] == 0, family


@pytest.mark.parametrize(
    ("prompt", "reason_fragment"),
    (
        ("open the door", "scene does not contain"),
        ("kick the block forward", "persistent contact lifecycle"),
        ("step over the block", "not a traversable floor-level obstacle"),
    ),
)
def test_out_of_affordance_prompts_fail_transparently(
    prompt: str,
    reason_fragment: str,
) -> None:
    program = plan_motion(
        PlanRequest(text=prompt, scene=default_scene(), provider="offline")
    ).program

    assert program.intent is Intent.UNSUPPORTED
    assert reason_fragment in (program.unsupported_reason or "")
