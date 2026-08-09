from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Iterator


POC_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_criteria(path: Path | None = None) -> dict[str, Any]:
    # JSON is valid YAML 1.2, so this remains dependency-free and machine-readable.
    return load_json(path or POC_ROOT / "acceptance_criteria.yaml")


def supported_cases() -> list[dict[str, Any]]:
    return load_json(FIXTURES / "planner_supported.json")


def unsupported_cases() -> list[dict[str, Any]]:
    return load_json(FIXTURES / "planner_unsupported.json")


def review_template() -> dict[str, Any]:
    return load_json(FIXTURES / "hangten_review_template.json")


def grasp_trials(criteria: dict[str, Any]) -> Iterator[dict[str, Any]]:
    grasp = criteria["grasp"]
    for shape, pose, hand, paraphrase_index in product(
        grasp["shapes"], grasp["poses"], grasp["hands"], range(len(grasp["paraphrases"]))
    ):
        prompt = grasp["paraphrases"][paraphrase_index].format(hand=hand)
        trial_id = f"g-{shape['id']}-{pose['id']}-{hand}-p{paraphrase_index + 1}"
        yield {
            "id": trial_id,
            "prompt": prompt,
            "expected": {
                "intent": "grasp",
                "hand": hand,
                "primitive": "grab",
                "object_id": "block",
            },
            "scene": {
                "schema_version": "1.0",
                "rig": {
                    "id": "mesh2motion-human-vrm1",
                    "asset_uri": "assets/models/human-male.glb",
                    "profile_uri": "config/rig_profiles/mesh2motion-human-vrm1.json",
                    "fixed_root": True
                },
                "objects": [
                    {
                        "id": "block",
                        "kind": "block",
                        "transform": {
                            "translation": dict(zip(("x", "y", "z"), pose["position_m"])),
                            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                        },
                        "dimensions_m": dict(zip(("x", "y", "z"), shape["dimensions_m"])),
                        "mass_kg": shape["mass_kg"],
                        "friction": 1.0,
                        "sockets": [{
                            "id": "front_center",
                            "transform": {
                                "translation": {"x": 0.0, "y": 0.0, "z": -shape["dimensions_m"][2] / 2.0},
                                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
                            },
                            "approach_normal": {"x": 0.0, "y": 0.0, "z": -1.0},
                            "grasp_span_m": min(shape["dimensions_m"][0], 0.09)
                        }]
                    }
                ],
                "fps": 30,
                "reachable_radius_m": 0.72
            },
        }


def fixture_errors(criteria: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    supported = supported_cases()
    unsupported = unsupported_cases()
    reviews = review_template()["records"]
    trials = list(grasp_trials(criteria))
    if len(supported) != criteria["planner"]["supported_total"]:
        errors.append(f"supported fixture count is {len(supported)}")
    if len(unsupported) != criteria["planner"]["unsupported_total"]:
        errors.append(f"unsupported fixture count is {len(unsupported)}")
    if len(reviews) != criteria["gesture"]["clip_total"]:
        errors.append(f"review fixture count is {len(reviews)}")
    review_cases = [record for record in supported if record.get("review_clip") is True]
    if [record.get("id") for record in review_cases] != [f"s{index:02d}" for index in range(1, 21)]:
        errors.append("review planner cases must be the ordered ids s01 through s20")
    profiles = [record.get("motion_profile") for record in review_cases]
    if any(not isinstance(profile, dict) for profile in profiles):
        errors.append("every review planner case must declare a motion_profile")
    elif len({json.dumps(profile, sort_keys=True) for profile in profiles}) != len(profiles):
        errors.append("review motion profiles must be pairwise unique")
    if len(trials) != criteria["grasp"]["expected_trials"]:
        errors.append(f"grasp matrix count is {len(trials)}")
    for name, records in (
        ("supported", supported),
        ("unsupported", unsupported),
        ("review", reviews),
        ("grasp", trials),
    ):
        ids = [record.get("id", record.get("clip_id")) for record in records]
        if len(ids) != len(set(ids)):
            errors.append(f"{name} fixture IDs are not unique")
    return errors
