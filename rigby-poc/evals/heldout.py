from __future__ import annotations

import itertools
import json
import random
from typing import Any

from evals.criteria import supported_cases


PROFILE_VALUES = {
    "style": ("relaxed", "neutral", "energetic", "precise", "playful"),
    "timing": ("quick", "balanced", "slow"),
    "height": ("low", "chest", "high"),
    "depth": ("close", "natural", "extended"),
    "lateral": ("inward", "center", "outward"),
    "wrist_pitch": ("down", "neutral", "up"),
    "wrist_yaw": ("inward", "neutral", "outward"),
    "wrist_roll": ("counterclockwise", "neutral", "clockwise"),
}


def _prompt(hand: str, profile: dict[str, str]) -> str:
    height = {
        "low": "low",
        "chest": "at chest height",
        "high": "high",
    }[profile["height"]]
    depth = {
        "close": "close to the body",
        "natural": "at a natural reach",
        "extended": "fully extended",
    }[profile["depth"]]
    lateral = {
        "inward": "drawn inward",
        "center": "centered on the arm",
        "outward": "held outward",
    }[profile["lateral"]]
    pitch = {
        "down": "Pitch the wrist down",
        "neutral": "Keep wrist pitch neutral",
        "up": "Pitch the wrist up",
    }[profile["wrist_pitch"]]
    yaw = {
        "inward": "yaw it inward",
        "neutral": "keep yaw neutral",
        "outward": "yaw it outward",
    }[profile["wrist_yaw"]]
    roll = {
        "counterclockwise": "roll it counterclockwise",
        "neutral": "keep roll neutral",
        "clockwise": "roll it clockwise",
    }[profile["wrist_roll"]]
    return (
        f"With your {hand} hand, make a {profile['timing']} {profile['style']} shaka {height} "
        f"and {depth}, {lateral}. {pitch}, {yaw}, and {roll}."
    )


def heldout_hangten_cases() -> list[dict[str, Any]]:
    keys = tuple(PROFILE_VALUES)
    prior = {
        json.dumps(case["motion_profile"], sort_keys=True)
        for case in supported_cases()[:20]
    }
    profiles = [
        dict(zip(keys, values))
        for values in itertools.product(*(PROFILE_VALUES[key] for key in keys))
    ]
    profiles = [profile for profile in profiles if json.dumps(profile, sort_keys=True) not in prior]
    random.Random(20260807).shuffle(profiles)
    cases: list[dict[str, Any]] = []
    # Interleave hands so review order cannot be used to infer candidate source.
    for index in range(30):
        hand = "right" if index % 2 == 0 else "left"
        profile = profiles[index]
        cases.append(
            {
                "id": f"h{index + 1:02d}",
                "prompt": _prompt(hand, profile),
                "hand": hand,
                "motion_profile": profile,
            }
        )
    return cases


def heldout_complex_hangten_uplift_cases() -> list[dict[str, Any]]:
    """Thirty unseen profiles with the full present/shake/recover semantics.

    The wording is stratified so success cannot depend on seeing the literal
    anatomy terms in every request: one third use everyday gesture language,
    one third describe palm rotation, and one third name forearm
    pronation/supination explicitly.
    """
    cycle_words = {2: "twice", 3: "three times", 4: "four times"}
    cases: list[dict[str, Any]] = []
    for index, base in enumerate(heldout_hangten_cases()):
        cycles = (2, 3, 4)[index % 3]
        wording_index = (index // 3) % 3
        middle_three = (
            "Keep the thumb and little finger extended and the middle three fingers "
            "as fully curled as possible. "
        )
        if wording_index == 0:
            action = (
                f"Shake the hang-ten back and forth {cycle_words[cycles]}, keeping the wrist "
                "from bending, then return fully to default."
            )
            wording = "natural_shake"
        elif wording_index == 1:
            action = (
                f"Rock the palm orientation back and forth {cycle_words[cycles]} by rotating "
                "the forearm, then return fully to default."
            )
            wording = "palm_rotation"
        else:
            action = (
                f"Pronate and supinate the forearm back and forth {cycle_words[cycles]} while "
                "the wrist joint stays stable, then return fully to default."
            )
            wording = "anatomical"
        cases.append(
            {
                **base,
                "id": f"hx{index + 1:02d}",
                "prompt": f"{base['prompt']} {middle_three}{action}",
                "expected_cycles": float(cycles),
                "shake_wording": wording,
            }
        )
    return cases


def heldout_complex_hangten_cases() -> list[dict[str, Any]]:
    """Small pre-VLM pilot for phase sequencing, not the 30-prompt uplift gate."""
    return [
        {
            "id": "cx01",
            "hand": "right",
            "expected_cycles": 3.0,
            "prompt": (
                'Throw up a "hang-ten" sign with your right hand, there should be a swift motion '
                "up to the main position wherein the middle three fingers are as contracted as "
                "possible, the wrist should then shake rapidly back and forth a few times, before "
                "returning to default"
            ),
        },
        {
            "id": "cx02",
            "hand": "left",
            "expected_cycles": 2.0,
            "prompt": (
                "With your left hand, slowly raise a hang-ten sign to chest height with the middle "
                "three fingers fully curled. Shake the wrist quickly back and forth twice, then "
                "return the arm and hand to their relaxed default pose."
            ),
        },
        {
            "id": "cx03",
            "hand": "right",
            "expected_cycles": 4.0,
            "prompt": (
                "Quickly present a playful hang-ten high and held outward with your right hand. "
                "Keep the thumb and little finger extended and the middle three fingers maximally "
                "contracted. Shake the wrist rapidly back and forth four times, then recover fully "
                "to default."
            ),
        },
        {
            "id": "cx04",
            "hand": "left",
            "expected_cycles": 3.0,
            "prompt": (
                "With your left hand, present a precise hang-ten close to the body at chest height. "
                "The middle three fingers should be fully contracted. Shake the wrist back and "
                "forth three times at a controlled pace, then cleanly return to the relaxed default."
            ),
        },
    ]
