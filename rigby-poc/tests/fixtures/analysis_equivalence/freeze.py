"""Freeze the current analysis output for a representative program set.

Run from the repository root::

    uv run python tests/fixtures/analysis_equivalence/freeze.py

Each case stores the *effective* scene and program plus the metrics
``rigby_poc.analysis.analyze`` produces for them today. The equivalence harness
(``tests/test_analysis_equivalence.py``) recompiles each program, re-runs the
analyzer, and asserts the output is byte-identical both to this snapshot and to
what ``compiler.py`` computes in the same run.

This fixture is a stand-in for the golden corpus being built in PR 03a. When
that lands it supersedes this directory; the harness keeps its shape.

Regenerating is only correct when a metric *definition* deliberately changed. A
diff here otherwise means the move was not behaviour-preserving.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIXTURE_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rigby_poc import analysis  # noqa: E402
from rigby_poc.compiler import compile_motion  # noqa: E402
from rigby_poc.models import CompileRequest, PlanRequest, default_scene  # noqa: E402
from rigby_poc.planner import OfflinePlanner  # noqa: E402


# One case per distinct shape of analysis output, not one per prompt. Between
# them these exercise every metric family the analysis layer owns today:
# safety on all seven intents, gesture structure, shake oscillation, intra-hand
# contact, gaze, semantic cycle on both composite and full body, parallel
# forearm, and angular kinematics on all three of its bone sets.
CASES: list[tuple[str, str]] = [
    (
        "gesture_shaka_right",
        "With your right hand, make a quick energetic shaka high and extended, "
        "held outward from the body. Pitch the wrist up, yaw it outward, and "
        "roll it clockwise.",
    ),
    ("gesture_open_palm", "show an open palm"),
    ("gesture_point_right", "point forward with your right index finger"),
    ("strike_left_hook", "throw a left hook"),
    ("grab_block_right", "grab the block with your right hand"),
    (
        "composite_travel_foul",
        'roll your forearms around eachother repeatedly, as if you are '
        'indicating the "travel" foul in basketball',
    ),
    (
        "composite_finger_count_gaze",
        "count every finger on your left hand with your left thumb while "
        "watching your hand",
    ),
    ("composite_wave", "wave your right hand"),
    ("full_body_walk", "walk forward three steps"),
    (
        "full_body_wave_while_walking",
        "wave hello with your right hand while walking forward three steps",
    ),
    ("full_body_climb", "climb the ladder for three rungs"),
    ("object_throw_forward", "throw the block forward"),
    ("object_handoff", "pass the block from your right hand to your left hand"),
    ("sequence_catch_then_turn", "catch the block then turn around"),
]


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)


def build_case(case_id: str, prompt: str) -> dict:
    scene = default_scene()
    outcome = OfflinePlanner().plan(
        PlanRequest(text=prompt, scene=scene, provider="offline")
    )
    program = outcome.program
    clip = compile_motion(
        CompileRequest(scene=scene, program=program, persist=False)
    )
    if not clip.frames:
        raise SystemExit(f"{case_id}: compiled to zero frames, cannot freeze")
    metrics = analysis.analyze(clip, program, scene)
    return {
        "id": case_id,
        "prompt": prompt,
        "intent": program.intent.value,
        "frame_count": len(clip.frames),
        "scene": scene.model_dump(mode="json"),
        "program": program.model_dump(mode="json"),
        "expected_metrics": metrics,
    }


def main() -> None:
    written = []
    for case_id, prompt in CASES:
        case = build_case(case_id, prompt)
        path = FIXTURE_DIR / f"{case_id}.json"
        path.write_text(canonical(case) + "\n", encoding="utf-8")
        written.append((case_id, case["intent"], len(case["expected_metrics"])))
    index = {
        "schema_version": "1.0",
        "note": (
            "Frozen analysis output for PR 02a. Superseded by the golden corpus "
            "(plan 03) once it lands."
        ),
        "cases": [case_id for case_id, _ in CASES],
    }
    (FIXTURE_DIR / "index.json").write_text(canonical(index) + "\n", encoding="utf-8")
    for case_id, intent, count in written:
        print(f"{case_id:34s} {intent:20s} {count:3d} owned metric keys")


if __name__ == "__main__":
    main()
