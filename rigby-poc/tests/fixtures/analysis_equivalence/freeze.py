"""Freeze the compiler's metric output, and the analysis layer's, for a case set.

Run from the repository root::

    uv run python tests/fixtures/analysis_equivalence/freeze.py

Each case stores the *effective* scene and program plus two snapshots:

``compiler_metrics``
    The whole of ``ClipResult.metrics``. This is the **move baseline**: it was
    captured before PR 02b started moving metric blocks out of ``compiler.py``
    and it must never change while the extraction is a move rather than a
    redesign. ``freeze.py`` refuses to overwrite it without
    ``--rebless-compiler``, because silently re-blessing this snapshot to make a
    test pass would destroy the only evidence that the extraction preserved
    behaviour.

``expected_metrics``
    What ``rigby_poc.analysis.analyze`` owns today. This one legitimately grows
    as each PR ports another path, so a routine ``freeze.py`` run rewrites it.
    It is not the safety net; ``compiler_metrics`` is.

The harness (``tests/test_analysis_equivalence.py``) recompiles every program
and checks both snapshots plus the agreement between the analyzer and the
compiler in the same run.

This fixture is a stand-in for the golden corpus being built in PR 03. When that
lands it supersedes this directory; the harness keeps its shape.
"""

from __future__ import annotations

import argparse
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


# One case per distinct shape of metric output, not one per prompt.
#
# The first fourteen are 02a's set: they exercise every metric family the
# analysis layer owned before the full-body port — safety on all seven intents,
# gesture structure, shake oscillation, intra-hand contact, gaze, semantic
# cycle, parallel forearm, and angular kinematics on all three of its bone sets.
#
# The rest were added by 02b. Between them they fire every gated metric block in
# ``_compile_full_body``: the eight label-keyed exercise families, both rotation
# variants, both obstacle traversal modes, both horizontal support poses, and
# the plain locomotion actions that contribute only to the shared blocks. The
# rule is that no branch inside the moved code is left unexecuted by the
# fixture.
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
    # --- 02b: one case per gated block in the full-body metric pass ---
    ("full_body_jumping_jack", "do three jumping jacks"),
    ("full_body_burpee", "do two burpees"),
    ("full_body_squat", "do three squats"),
    ("full_body_lunge", "do two lunges"),
    ("full_body_single_leg_balance", "balance on your left leg"),
    ("full_body_sit_up", "do three sit-ups"),
    ("full_body_crawl", "crawl forward on all fours"),
    ("full_body_push_up", "do three push-ups"),
    ("full_body_dance", "dance to four beats"),
    ("full_body_cartwheel", "do a cartwheel"),
    ("full_body_floor_roll", "do a forward roll"),
    ("full_body_obstacle_over", "step over the hurdle"),
    ("full_body_obstacle_around", "walk around the hurdle"),
    ("full_body_plank", "hold a plank"),
    ("full_body_lie_supine", "lie down on your back"),
    ("full_body_kick", "kick with your right foot"),
    ("full_body_turn", "turn around"),
    ("full_body_run", "run forward four steps"),
    ("full_body_crouch", "crouch down"),
]


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def build_case(case_id: str, prompt: str) -> tuple[dict, dict]:
    """Return ``(case, compiler_metrics)`` for one prompt."""

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
    case = {
        "id": case_id,
        "prompt": prompt,
        "intent": program.intent.value,
        "frame_count": len(clip.frames),
        "scene": scene.model_dump(mode="json"),
        "program": program.model_dump(mode="json"),
        "expected_metrics": analysis.analyze(clip, program, scene),
    }
    return case, json.loads(json.dumps(clip.metrics))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebless-compiler",
        action="store_true",
        help=(
            "overwrite the frozen compiler_metrics baseline. Only correct when a "
            "metric definition deliberately changed; never to make a test pass."
        ),
    )
    args = parser.parse_args()

    written: list[tuple[str, str, int, str]] = []
    drifted: list[str] = []
    added: set[str] = set()
    for case_id, prompt in CASES:
        path = FIXTURE_DIR / f"{case_id}.json"
        previous = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
        case, compiler_metrics = build_case(case_id, prompt)
        baseline = previous.get("compiler_metrics")
        if baseline is None or args.rebless_compiler:
            case["compiler_metrics"] = compiler_metrics
            state = "blessed" if baseline is None else "REBLESSED"
        else:
            # A key may be *added* — 02b persists the commanded IK support
            # targets, which no post-hoc pass can invert out of a clip. What may
            # never happen is a frozen key changing value, so the comparison is
            # over the baseline's own key set. tests/test_analysis_equivalence.py
            # makes the same distinction and additionally requires every
            # addition to be declared.
            case["compiler_metrics"] = baseline
            changed = [
                key
                for key in baseline
                if key not in compiler_metrics
                or _compact(compiler_metrics[key]) != _compact(baseline[key])
            ]
            new_keys = set(compiler_metrics) - set(baseline)
            added |= new_keys
            if changed:
                drifted.append(f"{case_id} ({', '.join(sorted(changed)[:4])})")
                state = "DRIFTED"
            elif new_keys:
                state = f"held +{len(new_keys)}"
            else:
                state = "held"
        path.write_text(canonical(case) + "\n", encoding="utf-8")
        written.append(
            (case_id, case["intent"], len(case["expected_metrics"]), state)
        )
    index = {
        "schema_version": "2.0",
        "note": (
            "Frozen compiler and analysis output for the PR 02 extraction. "
            "compiler_metrics is the move baseline and predates 02b; "
            "expected_metrics grows as each path is ported. Superseded by the "
            "golden corpus (plan 03) once it covers these paths."
        ),
        "cases": [case_id for case_id, _ in CASES],
    }
    (FIXTURE_DIR / "index.json").write_text(canonical(index) + "\n", encoding="utf-8")
    for case_id, intent, count, state in written:
        print(f"{case_id:32s} {intent:20s} {count:3d} owned keys  {state}")
    if added:
        print("\nkeys added since the baseline: " + ", ".join(sorted(added)))
    if drifted:
        raise SystemExit(
            "compiler metrics drifted from the frozen baseline for: "
            + ", ".join(drifted)
            + "\nThe extraction is supposed to be behaviour-preserving. Find the "
            "cause; do not pass --rebless-compiler to silence this."
        )


if __name__ == "__main__":
    main()
