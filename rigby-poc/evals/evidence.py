from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from itertools import combinations
from typing import Any

from .models import GateResult, Status


def lookup(value: Any, names: Iterable[str]) -> Any:
    wanted = {name.lower() for name in names}
    queue = [value]
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            for key, child in current.items():
                if key.lower() in wanted:
                    return child
                if isinstance(child, (dict, list)):
                    queue.append(child)
        elif isinstance(current, list):
            queue.extend(item for item in current if isinstance(item, (dict, list)))
    return None


def _text_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    queue = [value]
    while queue:
        current = queue.pop()
        if isinstance(current, str):
            tokens.add(current.lower().replace("-", "_"))
        elif isinstance(current, dict):
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return tokens


def program_matches(program: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    intent = lookup(program, ["intent", "action", "motion_type"])
    hand = lookup(program, ["hand", "handedness", "effector"])
    object_id = lookup(program, ["object_id", "target_object_id", "target_id"])
    primitive = lookup(program, ["primitive", "primitive_type", "hand_shape", "gesture"])
    tokens = _text_tokens(program)
    expected_intent = expected["intent"]
    if not isinstance(intent, str) or (
        intent.lower() != expected_intent and not ({intent.lower(), expected_intent} <= {"grab", "grasp"})
    ):
        failures.append(f"intent expected {expected_intent!r}, got {intent!r}")
    if not isinstance(hand, str) or expected["hand"] not in hand.lower():
        failures.append(f"hand expected {expected['hand']!r}, got {hand!r}")
    expected_primitive = expected["primitive"]
    primitive_values = set(tokens)
    if isinstance(primitive, str):
        primitive_values.add(primitive.lower().replace("-", "_"))
    aliases = {expected_primitive}
    if expected_primitive == "grab":
        aliases.add("grasp")
    if not primitive_values.intersection(aliases):
        failures.append(f"primitive expected one of {sorted(aliases)}, got {primitive!r}")
    if expected.get("object_id") is not None and object_id != expected["object_id"]:
        failures.append(f"object expected {expected['object_id']!r}, got {object_id!r}")
    expected_profile = expected.get("motion_profile")
    if isinstance(expected_profile, dict):
        actual_profile = lookup(program, ["motion_profile"])
        if not isinstance(actual_profile, dict):
            failures.append("motion_profile is missing")
        else:
            for key, value in expected_profile.items():
                if actual_profile.get(key) != value:
                    failures.append(f"motion_profile.{key} expected {value!r}, got {actual_profile.get(key)!r}")
    return not failures, failures


def is_rejected(api_result: dict[str, Any]) -> bool:
    status = api_result.get("status")
    body = api_result.get("body")
    if isinstance(status, int) and 400 <= status < 500:
        return True
    if not isinstance(body, dict):
        return False
    rejected = lookup(body, ["unsupported_reason", "rejection_reason", "rejected", "unsupported"])
    state = lookup(body, ["status", "state"])
    return bool(rejected) or (isinstance(state, str) and state.lower() in {"rejected", "unsupported"})


def planner_gate(
    supported: list[dict[str, Any]], unsupported: list[dict[str, Any]], criteria: dict[str, Any], provider: str = "openai"
) -> GateResult:
    failures: list[str] = []
    missing: list[str] = []
    supported_correct = 0
    provider_proven = 0
    for item in supported:
        body = item.get("body")
        if body is None:
            missing.append(item["id"])
            continue
        program = body.get("program", body) if isinstance(body, dict) else {}
        correct, reasons = program_matches(program, item["expected"])
        if correct:
            supported_correct += 1
        else:
            failures.append(f"{item['id']}: {'; '.join(reasons)}")
        actual_provider = lookup(body, ["provider"])
        model = lookup(body, ["model"])
        model_calls = lookup(body, ["model_calls"])
        if actual_provider == criteria.get("required_provider") and isinstance(model, str) and model and isinstance(model_calls, int) and model_calls >= criteria.get("min_model_calls_per_case", 0):
            provider_proven += 1
        else:
            failures.append(f"{item['id']}: planner provenance is not a qualifying {criteria.get('required_provider')} model call")
    unsupported_correct = sum(is_rejected(item) for item in unsupported)
    for item in unsupported:
        body = item.get("body")
        actual_provider = lookup(body or {}, ["provider"])
        model = lookup(body or {}, ["model"])
        model_calls = lookup(body or {}, ["model_calls"])
        if actual_provider == criteria.get("required_provider") and isinstance(model, str) and model and isinstance(model_calls, int) and model_calls >= criteria.get("min_model_calls_per_case", 0):
            provider_proven += 1
        else:
            failures.append(f"{item['id']}: rejection lacks qualifying planner provenance")
    missing.extend(item["id"] for item in unsupported if item.get("body") is None and item.get("status") is None)
    supported_fraction = supported_correct / len(supported) if supported else 0.0
    unsupported_fraction = unsupported_correct / len(unsupported) if unsupported else 0.0
    passed = (
        provider == criteria.get("required_provider", provider)
        and len(supported) == criteria["supported_total"]
        and len(unsupported) == criteria["unsupported_total"]
        and supported_fraction >= criteria["supported_min_fraction"]
        and unsupported_fraction >= criteria["unsupported_min_fraction"]
        and provider_proven == len(supported) + len(unsupported)
    )
    status = Status.PASS if passed else (Status.UNVERIFIED if missing else Status.FAIL)
    if missing:
        failures.append(f"missing API evidence for {len(set(missing))} cases")
    if provider != criteria.get("required_provider", provider):
        failures.append(f"planner provider {provider!r} is development-only; {criteria['required_provider']!r} is required")
        status = Status.UNVERIFIED
    return GateResult(
        "planner",
        status,
        f"{supported_correct}/{len(supported)} supported and {unsupported_correct}/{len(unsupported)} unsupported correct",
        {
            "supported_correct": supported_correct,
            "supported_total": len(supported),
            "supported_fraction": supported_fraction,
            "unsupported_correct": unsupported_correct,
            "unsupported_total": len(unsupported),
            "unsupported_fraction": unsupported_fraction,
            "provider": provider,
            "provider_proven_cases": provider_proven,
            "provider_proven_required": len(supported) + len(unsupported),
        },
        criteria,
        failures=failures[:50],
    )


def _normalized_finger_state(clip: dict[str, Any]) -> dict[str, float] | None:
    state = lookup(clip, ["finger_state", "finger_states", "normalized_finger_curls"])
    if isinstance(state, list) and state:
        state = state[-1]
    if not isinstance(state, dict):
        return None
    curls: dict[str, float] = {}
    for finger in ("thumb", "index", "middle", "ring", "little", "pinky"):
        item = state.get(finger)
        if isinstance(item, (int, float)):
            curls[finger] = float(item)
        elif isinstance(item, dict):
            curl = item.get("curl")
            extension = item.get("extension")
            if isinstance(curl, (int, float)):
                curls[finger] = float(curl)
            elif isinstance(extension, (int, float)):
                curls[finger] = 1.0 - float(extension)
    if "pinky" in curls and "little" not in curls:
        curls["little"] = curls["pinky"]
    return curls or None


def _quat_angle(rotation: Any) -> float | None:
    if isinstance(rotation, dict):
        values = [rotation.get(key) for key in ("x", "y", "z", "w")]
    elif isinstance(rotation, list):
        values = rotation
    else:
        return None
    if len(values) != 4 or any(not isinstance(value, (int, float)) for value in values):
        return None
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= 1e-12:
        return None
    return 2.0 * math.acos(min(1.0, abs(float(values[3]) / norm)))


def _raw_hangten(clip: dict[str, Any], hand: str | None) -> tuple[bool | None, list[str]]:
    frames = clip.get("frames")
    if not isinstance(frames, list) or not frames:
        return None, ["missing raw clip frames"]
    hands = [hand] if hand in {"left", "right"} else ["left", "right"]
    candidates: list[tuple[float, dict[str, float]]] = []
    for frame in frames:
        bones = frame.get("bones") if isinstance(frame, dict) else None
        if not isinstance(bones, dict):
            continue
        for side in hands:
            angles: dict[str, float] = {}
            keys = {
                "thumb": f"{side}ThumbMetacarpal", "index": f"{side}IndexProximal",
                "middle": f"{side}MiddleProximal", "ring": f"{side}RingProximal",
                "little": f"{side}LittleProximal",
            }
            for finger, key in keys.items():
                pose = bones.get(key)
                angle = _quat_angle(pose.get("rotation")) if isinstance(pose, dict) else None
                if angle is not None:
                    angles[finger] = angle
            if len(angles) == 5:
                score = sum(angles[name] for name in ("index", "middle", "ring")) / 3.0
                candidates.append((score, angles))
    if not candidates:
        return None, ["finger-bone rotations are absent"]
    _, angles = max(candidates, key=lambda item: item[0])
    failures = []
    for finger in ("index", "middle", "ring"):
        if angles[finger] < 0.75:
            failures.append(f"{finger} proximal rotation {angles[finger]:.3f} rad is not curled")
    for finger in ("thumb", "little"):
        if angles[finger] > 0.45:
            failures.append(f"{finger} proximal rotation {angles[finger]:.3f} rad is not extended")
    return not failures, failures


def hangten_assertion(clip: dict[str, Any], hand: str | None = None) -> tuple[bool | None, list[str]]:
    state = _normalized_finger_state(clip)
    if state is None:
        raw_result, raw_failures = _raw_hangten(clip, hand)
        if raw_result is not None:
            return raw_result, raw_failures
        assertions = lookup(clip, ["finger_assertions"])
        computed = lookup(clip, ["finger_assertions_computed_from_clip", "computed_from_clip"])
        if isinstance(assertions, dict) and computed is True:
            failed = [str(key) for key, value in assertions.items() if value is not True]
            return not failed, failed
        return None, ["missing raw normalized finger state"]
    required = {"thumb": (0.0, 0.35), "index": (0.65, 1.0), "middle": (0.65, 1.0),
                "ring": (0.65, 1.0), "little": (0.0, 0.35)}
    failures = []
    for finger, (minimum, maximum) in required.items():
        value = state.get(finger)
        if value is None:
            failures.append(f"missing {finger} curl")
        elif not minimum <= value <= maximum:
            failures.append(f"{finger} curl {value:.3f} not in [{minimum}, {maximum}]")
    return not failures, failures


def gesture_gate(clips: list[dict[str, Any]], review: dict[str, Any] | None, criteria: dict[str, Any]) -> GateResult:
    failures: list[str] = []
    assertion_passes = 0
    assertion_missing = 0
    for item in clips:
        hand = lookup(item.get("program", {}), ["hand", "handedness"])
        passed, reasons = hangten_assertion(item.get("clip", {}), hand if isinstance(hand, str) else None)
        if passed is True:
            assertion_passes += 1
        elif passed is None:
            assertion_missing += 1
        else:
            failures.append(f"{item['id']}: {'; '.join(reasons)}")
    reviewed_passes = 0
    reviewed_complete = 0
    review_records = review.get("records", []) if isinstance(review, dict) else []
    for record in review_records:
        ego = record.get("egocentric_ratings", [])
        orbit = record.get("orbit_ratings", [])
        valid_ego = [float(x) for x in ego if isinstance(x, (int, float)) and 1 <= x <= 5]
        valid_orbit = [float(x) for x in orbit if isinstance(x, (int, float)) and 1 <= x <= 5]
        if not valid_ego or not valid_orbit:
            continue
        reviewed_complete += 1
        if sum(valid_ego) / len(valid_ego) >= criteria["review_min_score"] and sum(valid_orbit) / len(valid_orbit) >= criteria["review_min_score"]:
            reviewed_passes += 1
    review_fraction = reviewed_passes / criteria["clip_total"]
    passed = (
        len(clips) == criteria["clip_total"]
        and assertion_passes == criteria["clip_total"]
        and reviewed_complete == criteria["clip_total"]
        and review_fraction >= criteria["review_min_fraction"]
    )
    missing = assertion_missing or reviewed_complete < criteria["clip_total"] or len(clips) < criteria["clip_total"]
    status = Status.PASS if passed else (Status.UNVERIFIED if missing else Status.FAIL)
    if reviewed_complete < criteria["clip_total"]:
        failures.append(f"manual ratings complete for {reviewed_complete}/{criteria['clip_total']} prompt-aware clips")
    if assertion_missing:
        failures.append(f"raw finger evidence missing for {assertion_missing} clips")
    return GateResult(
        "gesture",
        status,
        f"{assertion_passes}/{len(clips)} finger assertions; {reviewed_passes}/{criteria['clip_total']} clips rated at least 4/5 in both views",
        {
            "clip_count": len(clips), "finger_assertion_passes": assertion_passes,
            "reviews_complete": reviewed_complete, "review_passes": reviewed_passes,
            "review_fraction": review_fraction,
        },
        criteria,
        failures=failures[:30],
    )


def _motion_hash(clip: dict[str, Any]) -> str | None:
    if not clip.get("frames"):
        return None
    canonical = {
        key: clip.get(key)
        for key in ("schema_version", "fps", "duration_s", "frames", "contacts")
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def gesture_diversity_gate(
    clips: list[dict[str, Any]],
    criteria: dict[str, Any],
) -> GateResult:
    fields = list(criteria["descriptor_ranges"])
    descriptors: dict[str, list[float]] = {}
    motion_hashes: dict[str, str] = {}
    profiles: dict[str, str] = {}
    evidence: list[str] = []
    failures: list[str] = []
    missing: list[str] = []

    for item in clips:
        clip_id = str(item.get("id", "unknown"))
        clip = item.get("clip") if isinstance(item.get("clip"), dict) else {}
        observables = clip.get("slider_observables") or clip.get("parametric_observables")
        profile = lookup(item.get("program", {}), ["motion_profile"])
        motion_hash = _motion_hash(clip)
        if not isinstance(observables, dict) or not isinstance(profile, dict) or motion_hash is None:
            missing.append(clip_id)
            continue
        raw_descriptor: list[float] = []
        normalized: list[float] = []
        for field in fields:
            value = observables.get(field)
            limits = criteria["descriptor_ranges"][field]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                missing.append(f"{clip_id}:{field}")
                break
            number = float(value)
            raw_descriptor.append(number)
            normalized.append((number - float(limits[0])) / (float(limits[1]) - float(limits[0])))
        else:
            descriptors[clip_id] = normalized
            motion_hashes[clip_id] = motion_hash
            profiles[clip_id] = json.dumps(profile, sort_keys=True, separators=(",", ":"))
            evidence.append(
                f"{clip_id}: " + ", ".join(f"{field}={value:.4f}" for field, value in zip(fields, raw_descriptor))
            )

    closest_pair: tuple[str, str] | None = None
    minimum_distance = math.inf
    minimum_changed = len(fields)
    close_pairs: list[str] = []
    for (left_id, left), (right_id, right) in combinations(descriptors.items(), 2):
        deltas = [abs(a - b) for a, b in zip(left, right)]
        distance = math.sqrt(sum(delta * delta for delta in deltas))
        changed = sum(delta >= criteria["changed_dimension_delta"] for delta in deltas)
        if distance < minimum_distance:
            minimum_distance = distance
            minimum_changed = changed
            closest_pair = (left_id, right_id)
        if distance < criteria["min_pairwise_normalized_distance"] or changed < criteria["min_changed_dimensions_per_pair"]:
            close_pairs.append(f"{left_id}/{right_id}: distance={distance:.3f}, changed_dimensions={changed}")

    unique_profiles = len(set(profiles.values()))
    unique_hashes = len(set(motion_hashes.values()))
    unique_descriptors = len({tuple(round(value, 6) for value in descriptor) for descriptor in descriptors.values()})
    expected = criteria["clip_total"]
    passed = (
        len(descriptors) == expected
        and unique_profiles == criteria["required_unique_profiles"]
        and unique_hashes == criteria["required_unique_motion_hashes"]
        and unique_descriptors == criteria["required_unique_descriptors"]
        and not close_pairs
    )
    if unique_profiles < criteria["required_unique_profiles"]:
        failures.append(f"only {unique_profiles}/{criteria['required_unique_profiles']} unique LLM motion profiles")
    if unique_hashes < criteria["required_unique_motion_hashes"]:
        failures.append(f"only {unique_hashes}/{criteria['required_unique_motion_hashes']} unique compiled motion hashes")
    if unique_descriptors < criteria["required_unique_descriptors"]:
        failures.append(f"only {unique_descriptors}/{criteria['required_unique_descriptors']} unique motion descriptors")
    failures.extend(close_pairs[:20])
    if missing:
        failures.append(f"missing diversity evidence for {len(missing)} clips or fields")
    status = Status.PASS if passed else (Status.UNVERIFIED if missing else Status.FAIL)
    return GateResult(
        "gesture_diversity",
        status,
        f"{unique_hashes}/{expected} unique motions; closest pair distance={minimum_distance:.3f}" if descriptors else "No gesture diversity evidence",
        {
            "clip_count": len(descriptors),
            "unique_motion_profiles": unique_profiles,
            "unique_motion_hashes": unique_hashes,
            "unique_motion_descriptors": unique_descriptors,
            "pair_count": len(descriptors) * (len(descriptors) - 1) // 2,
            "closest_pair": list(closest_pair) if closest_pair else None,
            "min_pairwise_normalized_distance": minimum_distance if math.isfinite(minimum_distance) else None,
            "min_changed_dimensions_in_closest_pair": minimum_changed if closest_pair else None,
        },
        criteria,
        evidence=evidence,
        failures=failures,
    )
def _number(metrics: dict[str, Any], names: list[str]) -> float | None:
    value = lookup(metrics, names)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _boolean(metrics: dict[str, Any], names: list[str]) -> bool | None:
    value = lookup(metrics, names)
    return value if isinstance(value, bool) else None


def physical_trial(metrics: dict[str, Any], criteria: dict[str, Any]) -> tuple[bool | None, list[str]]:
    fields = {
        "lift": _number(metrics, ["lift_height_m", "block_lift_m", "com_lift_m"]),
        "hold": _number(metrics, ["hold_duration_s", "physical_hold_s"]),
        "drift": _number(metrics, ["vertical_drift_m", "hold_vertical_drift_m"]),
        "slip": _number(metrics, ["palm_relative_slip_m", "relative_slip_m"]),
        "lost_table": _boolean(metrics, ["lost_table_contact", "table_contact_lost"]),
        "opposing": _boolean(metrics, ["opposing_contacts", "has_opposing_contacts"]),
        "weld": _boolean(metrics, ["weld_used", "hidden_attachment_used", "attachment_used"]),
    }
    missing = [key for key, value in fields.items() if value is None]
    if missing:
        return None, [f"missing {', '.join(missing)}"]
    failures = []
    if fields["lift"] < criteria["min_lift_m"]:
        failures.append("insufficient lift")
    if fields["hold"] < criteria["min_hold_s"]:
        failures.append("insufficient hold")
    if fields["drift"] >= criteria["max_vertical_drift_m"]:
        failures.append("vertical drift limit exceeded")
    if fields["slip"] >= criteria["max_palm_relative_slip_m"]:
        failures.append("palm-relative slip limit exceeded")
    if fields["lost_table"] is not True:
        failures.append("table contact was not lost")
    if fields["opposing"] is not True:
        failures.append("opposing contacts were not retained")
    if fields["weld"] is not False:
        failures.append("weld or hidden attachment was used")
    return not failures, failures


#: Safety fields that ``compiler._base_metrics`` **seeds** and several compile
#: paths never write, each mapped to the key that witnesses a real measurement.
#: Plan 08 §1.1 item 3.
#:
#: A seeded value is present, well-typed and finite, so ``_number`` returns it and
#: the gate compares it -- against a constant no clip can move. That is the §1.1
#: shape exactly: a gate that is read, documented, and structurally unable to fail.
#:
#: The witness is a key **only the measuring path emits**, because which members
#: are real depends on the compile path -- a grasp clip genuinely does measure
#: penetration, and reporting it unverified would be the mirror error. ``None``
#: means no path measures it at all.
#:
#: - ``foot_drift_m`` -- the literal ``0.0`` at ``compiler.py:484``,
#:   ``analysis/safety.py:29`` and ``analysis/safety.py:98``. Nothing computes it:
#:   foot world positions need forward kinematics and only ``hips`` carries a
#:   position on a clip frame.
#: - ``max_penetration_m`` -- measured only by the MuJoCo grasp trial
#:   (``physics.py:273``), which is also the only producer of ``physics_engine``.
#: - ``unresolved_non_hand_collisions`` -- ``physics.py:276`` writes a literal
#:   ``0``; the only real producer is ``analysis/hand.py:207``, which copies
#:   ``self_collision_frames`` out of the structure analysis that published it.
#:
#: The other seven members of ``analysis.equivalence.UNWRITTEN_BASE_DEFAULTS`` are
#: read by ``physical_trial``, which ``grasp_and_physical_gates`` only ever hands
#: real grasp trials, so they are measured wherever that gate can see them.
#:
#: ``tests/test_acceptance_harness.py`` fails if a producer starts measuring one,
#: so this cannot go stale into the opposite error.
SEEDED_SAFETY_FIELDS: dict[str, str | None] = {
    "foot_drift_m": None,
    "max_penetration_m": "physics_engine",
    "unresolved_non_hand_collisions": "self_collision_frames",
}


def _is_seeded(metrics: dict[str, Any], key: str) -> bool:
    """True when ``key`` holds its seeded constant rather than a measurement."""

    if key not in SEEDED_SAFETY_FIELDS:
        return False
    witness = SEEDED_SAFETY_FIELDS[key]
    return witness is None or lookup(metrics, [witness]) is None


def safety_trial(metrics: dict[str, Any], criteria: dict[str, Any]) -> tuple[bool | None, list[str]]:
    values = {
        "joint_limit_violations": _number(metrics, ["joint_limit_violations"]),
        "root_drift_m": _number(metrics, ["root_drift_m"]),
        "foot_drift_m": _number(metrics, ["foot_drift_m", "max_foot_drift_m"]),
        "unresolved_non_hand_collisions": _number(metrics, ["unresolved_non_hand_collisions"]),
        "nan_count": _number(metrics, ["nan_count"]),
        "discontinuities": _number(metrics, ["discontinuities", "discontinuity_count"]),
        "max_penetration_m": _number(metrics, ["max_penetration_m", "penetration_depth_m"]),
    }
    missing = [key for key, value in values.items() if value is None]
    unmeasured = [key for key in values if key not in missing and _is_seeded(metrics, key)]
    maximums = {
        "joint_limit_violations": criteria["max_joint_limit_violations"],
        "root_drift_m": criteria["max_root_drift_m"],
        "unresolved_non_hand_collisions": criteria["max_unresolved_non_hand_collisions"],
        "nan_count": criteria["max_nan_count"],
        "discontinuities": criteria["max_discontinuities"],
        "max_penetration_m": criteria["max_penetration_m"],
    }
    # Breaches are collected even when the verdict is UNVERIFIED. An invariant
    # that cannot be certified is not a reason to stop reporting the ones that
    # were measured and did fail -- dropping them would trade one silence for
    # another.
    failures = [
        f"{key}={values[key]} exceeds {limit}"
        for key, limit in maximums.items()
        if key not in missing and key not in unmeasured and values[key] > limit
    ]
    if missing or unmeasured:
        reasons = []
        if missing:
            reasons.append(f"missing {', '.join(missing)}")
        if unmeasured:
            reasons.append(
                f"not measured: {', '.join(unmeasured)} (seeded constant, no producer ran)"
            )
        return None, reasons + failures
    return not failures, failures


def grasp_and_physical_gates(trials: list[dict[str, Any]], grasp_cfg: dict[str, Any], physical_cfg: dict[str, Any]) -> tuple[GateResult, GateResult]:
    success = 0
    missing = 0
    failures: list[str] = []
    for trial in trials:
        result, reasons = physical_trial(trial.get("metrics", {}), physical_cfg)
        if result is True:
            success += 1
        elif result is None:
            missing += 1
        else:
            failures.append(f"{trial['id']}: {'; '.join(reasons)}")
    fraction = success / grasp_cfg["expected_trials"]
    count_ok = len(trials) == grasp_cfg["expected_trials"]
    passed = count_ok and not missing and fraction >= grasp_cfg["min_success_fraction"]
    status = Status.PASS if passed else (Status.UNVERIFIED if missing or len(trials) < grasp_cfg["expected_trials"] else Status.FAIL)
    measured = {"successes": success, "trials": len(trials), "missing_evidence": missing, "success_fraction": fraction}
    grasp = GateResult("grasp_robustness", status, f"{success}/{grasp_cfg['expected_trials']} physical pickups passed", measured, grasp_cfg, failures=failures[:50])
    physical = GateResult("physical_proof", status, "Physical pickup criteria are the authoritative grasp success definition", measured, physical_cfg, failures=failures[:50])
    return grasp, physical


def safety_gate(items: list[dict[str, Any]], criteria: dict[str, Any]) -> GateResult:
    passed = 0
    missing = 0
    failures: list[str] = []
    for item in items:
        result, reasons = safety_trial(item.get("metrics", {}), criteria)
        if result is True:
            passed += 1
        elif result is None:
            missing += 1
            failures.append(f"{item['id']}: unverified -- {'; '.join(reasons)}")
        else:
            failures.append(f"{item['id']}: {'; '.join(reasons)}")
    status = Status.PASS if items and passed == len(items) else (Status.UNVERIFIED if missing or not items else Status.FAIL)
    return GateResult("safety_and_quality", status, f"{passed}/{len(items)} generated clips passed every safety invariant", {"passed": passed, "clips": len(items), "missing_evidence": missing}, criteria, failures=failures[:50])


def structural_physics_gate(items: list[dict[str, Any]], criteria: dict[str, Any]) -> GateResult:
    failures: list[str] = []
    evidence_items = []
    required_digits = set(criteria["required_digits"])
    for item in items:
        source = item.get("response")
        body = getattr(source, "body", None) if source is not None else item
        model = lookup(body or {}, ["physics_model", "model_metadata", "hand_model"])
        coordinates = lookup(body or {}, ["coordinate_frames", "coordinate_conversion"])
        if not isinstance(model, dict) or not isinstance(coordinates, dict):
            continue
        digits = model.get("digits")
        digit_map: dict[str, Any] = {}
        if isinstance(digits, list):
            for digit in digits:
                if isinstance(digit, str):
                    digit_map[digit.lower()] = {}
                elif isinstance(digit, dict) and isinstance(digit.get("name"), str):
                    digit_map[digit["name"].lower()] = digit
        elif isinstance(digits, dict):
            digit_map = {str(key).lower(): value for key, value in digits.items()}
        item_failures = []
        if set(digit_map) != required_digits:
            item_failures.append(f"digits are {sorted(digit_map)}, expected {sorted(required_digits)}")
        for name in required_digits.intersection(digit_map):
            detail = digit_map[name]
            if not isinstance(detail, dict):
                detail = {}
            independent = detail.get("independently_actuated", model.get("independently_actuated_digits"))
            contactable = detail.get("contactable", model.get("contactable_digits"))
            if independent is not True:
                item_failures.append(f"{name} is not proven independently actuated")
            if contactable is not True:
                item_failures.append(f"{name} is not proven contactable")
        if model.get("parallel_gripper_proxy") is not False:
            item_failures.append("model is not explicitly declared non-parallel-gripper")
        app_up = str(coordinates.get("app_up_axis", coordinates.get("source_up_axis", ""))).upper()
        sim_up = str(coordinates.get("simulation_up_axis", coordinates.get("target_up_axis", ""))).upper()
        if app_up != criteria["app_up_axis"] or sim_up != criteria["simulation_up_axis"]:
            item_failures.append(f"coordinate conversion is {app_up}-up to {sim_up}-up")
        if coordinates.get("explicit_trajectory_conversion") is not True:
            item_failures.append("trajectory conversion is not explicit")
        roundtrip = coordinates.get("trajectory_roundtrip_error_m")
        if not isinstance(roundtrip, (int, float)) or roundtrip > criteria["max_coordinate_roundtrip_error_m"]:
            item_failures.append(f"coordinate round-trip error missing or too high: {roundtrip!r}")
        evidence_items.append(item["id"])
        failures.extend(f"{item['id']}: {failure}" for failure in item_failures)
    # The model is invariant, but require evidence from both a gesture and a grasp rollout.
    categories = {"gesture" if item_id.startswith("s") else "grasp" for item_id in evidence_items}
    missing = categories != {"gesture", "grasp"}
    status = Status.PASS if not failures and not missing else (Status.UNVERIFIED if missing else Status.FAIL)
    if missing:
        failures.append("structural metadata is required from at least one gesture and one grasp rollout")
    return GateResult(
        "structural_physics",
        status,
        f"Validated articulated hand and coordinate conversion metadata on {len(evidence_items)} rollouts",
        {"evidence_items": evidence_items, "categories": sorted(categories)},
        criteria,
        failures=failures[:50],
    )


def parametric_gate(cases: list[dict[str, Any]], criteria: dict[str, Any]) -> GateResult:
    failures: list[str] = []
    missing = 0
    passed = 0
    controls = criteria["controls"]
    by_slider = {case["control"]: case for case in cases}
    for control in controls:
        slider = control["name"]
        case = by_slider.get(slider)
        if case is None:
            missing += 1
            continue
        values = case.get("observed", [])
        compile_ms = case.get("compile_ms", [])
        planner_calls = case.get("planner_calls", [])
        successful_recompiles = case.get("successful_recompiles", [])
        if len(values) != len(control["levels"]) or any(not isinstance(x, (int, float)) for x in values):
            missing += 1
            continue
        reasons = []
        if control["behavior"] == "increasing":
            if not all(a <= b for a, b in zip(values, values[1:])) or values[0] == values[-1]:
                reasons.append(f"non-monotonic/no-op observations {values}")
        elif control["behavior"] == "mirrored":
            if len(values) != 2 or values[0] * values[1] >= 0 or abs(values[0] + values[1]) > criteria["mirror_tolerance_m"]:
                reasons.append(f"left/right trajectories are not mirrored: {values}")
        else:
            reasons.append(f"unknown expected behavior {control['behavior']!r}")
        if len(compile_ms) != len(values) or max(compile_ms, default=math.inf) >= criteria["max_compile_ms"]:
            reasons.append("compile time limit exceeded or missing")
        if len(planner_calls) != len(values) or any(x > criteria["max_planner_calls"] for x in planner_calls):
            reasons.append("compile invoked planner or telemetry is missing")
        if len(successful_recompiles) != len(values) or not all(successful_recompiles):
            reasons.append("one or more advertised levels did not produce a successful clip")
        if reasons:
            failures.append(f"{slider}: {'; '.join(reasons)}")
        else:
            passed += 1
    status = Status.PASS if passed == len(controls) else (Status.UNVERIFIED if missing else Status.FAIL)
    return GateResult("parametric_control", status, f"{passed}/{len(controls)} controls had the expected response, were model-free, and compiled under 3 seconds", {"passed": passed, "controls": len(controls), "missing": missing}, criteria, failures=failures)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def latency_gate(samples_ms: list[float], criteria: dict[str, Any]) -> GateResult:
    p95 = _percentile(samples_ms, criteria["percentile"])
    enough = len(samples_ms) >= criteria["min_samples"]
    passed = enough and p95 is not None and p95 < criteria["max_end_to_end_ms"]
    status = Status.PASS if passed else (Status.UNVERIFIED if not enough else Status.FAIL)
    return GateResult("overall_latency", status, f"p95={p95 if p95 is not None else 'missing'} ms over {len(samples_ms)} samples", {"samples": len(samples_ms), "p95_ms": p95}, criteria)


def export_gate(checks: list[dict[str, Any]], criteria: dict[str, Any]) -> GateResult:
    complete = len(checks) == criteria["sample_count"]
    failures: list[str] = []
    missing = 0
    passed = 0
    for check in checks:
        if check.get("compared_samples", 0) <= 0:
            missing += 1
        elif check.get("valid") and check.get("has_provenance") and not check.get("failures"):
            passed += 1
        else:
            failures.append(f"{check.get('id')}: {'; '.join(check.get('failures', ['failed']))}")
    status = Status.PASS if complete and passed == criteria["sample_count"] else (Status.UNVERIFIED if missing or not complete else Status.FAIL)
    return GateResult("glb_export", status, f"{passed}/{criteria['sample_count']} sampled exports independently reimported within tolerance with provenance", {"passed": passed, "checks": len(checks), "missing": missing}, criteria, failures=failures)
